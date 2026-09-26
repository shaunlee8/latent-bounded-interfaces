// Fused flash-JVP kernel: direction pairs (R=2 per launch), hd64, BM=BN=64,
// wgmma, occupancy 2. Entry point flash_jvp_occ2_full loops the pairs.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <tl_templates/cuda/instruction/wgmma.h>
#include "lbi_mma.cuh"
#include <tl_templates/cuda/intrin.h>
#include <tl_templates/cuda/barrier.h>

namespace flashjvp {

using bf16 = __nv_bfloat16;
constexpr int BM = 64, BN = 64, HD = 64;

__device__ inline int sw128(int r, int c) {
    // [rows][64] bf16 tile, 128-byte swizzle (SW128 layout).
    return r * 64 + ((((c >> 3) & 7) ^ (r & 7)) << 3) + (c & 7);
}
__device__ inline void acc_rc(int w, int l, int reg, int& r, int& c) {
    const int j8 = reg >> 2, e = reg & 3;
    r = w * 16 + (l >> 2) + ((e >> 1) << 3);
    c = (j8 << 3) + ((l & 3) << 1) + (e & 1);
}

constexpr int TILE = BM * HD;

__device__ __forceinline__ void cp16(bf16* dst, const bf16* src) {
    unsigned sdst = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(sdst), "l"(src));
}

// Single-buffered KV (96 KB smem, 2 CTAs/SM); the co-resident CTA supplies
// the cross-phase overlap of double buffering.
constexpr size_t SMEM_OCC2 = (size_t)12 * TILE * 2;

// Hout == 0: O/DO in [B,H,L,hd]. Hout > 0: flat [B,L,Hout*hd] so out_proj
// consumes the epilogue directly (same contiguous 64-channel store, new stride).
__global__ void __launch_bounds__(128, 2) flash_jvp_occ2_kernel(
    const bf16* __restrict__ Q, const bf16* __restrict__ K,
    const bf16* __restrict__ V, const bf16* __restrict__ DQ,
    const bf16* __restrict__ DK, const bf16* __restrict__ DV,
    bf16* __restrict__ O, bf16* __restrict__ DO,
    long lane_str, int Lctx, float scale, int Hout) {
    extern __shared__ __align__(1024) char smem[];
    bf16* q_s   = reinterpret_cast<bf16*>(smem);
    bf16* dq0_s = q_s + TILE;
    bf16* dq1_s = dq0_s + TILE;
    bf16* kvbuf = dq1_s + TILE;               // [1][6][TILE]
    bf16* p_s   = kvbuf + 6 * TILE;
    bf16* pds0_s = p_s + TILE;
    bf16* pds1_s = pds0_s + TILE;

    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int pid_m = blockIdx.x;
    const long bh = blockIdx.y;
    const long base = bh * Lctx * HD;
    const int row0 = pid_m * BM;

    for (int idx = tid; idx < BM * 8; idx += 128) {
        const int r = idx >> 3, u = idx & 7;
        const long g = base + (long)(row0 + r) * HD + u * 8;
        const int so = sw128(r, u * 8);
        *reinterpret_cast<uint4*>(q_s + so) = *reinterpret_cast<const uint4*>(Q + g);
        *reinterpret_cast<uint4*>(dq0_s + so) = *reinterpret_cast<const uint4*>(DQ + g);
        *reinterpret_cast<uint4*>(dq1_s + so) = *reinterpret_cast<const uint4*>(DQ + lane_str + g);
    }

    const int n_tiles = pid_m + 1;
    const bf16* gsrc[6] = {K, V, DK, DK, DV, DV};
    const long goff[6] = {0, 0, 0, lane_str, 0, lane_str};

    auto prefetch = [&](int t) {
        if (t >= n_tiles) return;
        bf16* buf = kvbuf;
        const long col_base = base + (long)(t * BN) * HD;
        for (int idx = tid; idx < BN * 8; idx += 128) {
            const int r = idx >> 3, u = idx & 7;
            const int so = sw128(r, u * 8);
            const long g = col_base + (long)r * HD + u * 8;
#pragma unroll
            for (int j = 0; j < 6; ++j)
                cp16(buf + j * TILE + so, gsrc[j] + goff[j] + g);
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    };


    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
    lbi::mma::Accum<32> acc_o, do0, do1;
    float as0_0 = 0.f, as0_1 = 0.f, as1_0 = 0.f, as1_1 = 0.f;
#pragma unroll
    for (int i = 0; i < 32; ++i) { acc_o[i] = 0.f; do0[i] = 0.f; do1[i] = 0.f; }

    tl::GmmaDescriptor d_q, d_dq0, d_dq1, d_p, d_pds0, d_pds1;
    tl::initialize_wgmma_descriptor<1, 1, 64>(d_q, q_s);
    tl::initialize_wgmma_descriptor<1, 1, 64>(d_dq0, dq0_s);
    tl::initialize_wgmma_descriptor<1, 1, 64>(d_dq1, dq1_s);
    tl::initialize_wgmma_descriptor<1, 1, 64>(d_p, p_s);
    tl::initialize_wgmma_descriptor<1, 1, 64>(d_pds0, pds0_s);
    tl::initialize_wgmma_descriptor<1, 1, 64>(d_pds1, pds1_s);

    for (int t = 0; t < n_tiles; ++t) {
        bf16* buf = kvbuf;
        tl::GmmaDescriptor d_k, d_v, d_dk0, d_dk1, d_dv0, d_dv1;
        tl::initialize_wgmma_descriptor<1, 1, 64>(d_k, buf);
        tl::initialize_wgmma_descriptor<1, 1024, 64>(d_v, buf + TILE);
        tl::initialize_wgmma_descriptor<1, 1, 64>(d_dk0, buf + 2 * TILE);
        tl::initialize_wgmma_descriptor<1, 1, 64>(d_dk1, buf + 3 * TILE);
        tl::initialize_wgmma_descriptor<1, 1024, 64>(d_dv0, buf + 4 * TILE);
        tl::initialize_wgmma_descriptor<1, 1024, 64>(d_dv1, buf + 5 * TILE);

        prefetch(t);
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        __syncthreads();

        lbi::mma::Accum<32> S, dS0, dS1;
        lbi::mma::fence_operand(S);
        lbi::mma::fence_operand(dS0);
        lbi::mma::fence_operand(dS1);
        lbi::mma::arrive();
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
            lbi::mma::mma_ss<64, 64, 16, false, false>(
                uint64_t(d_q + ki * 2), uint64_t(d_k + ki * 2), S, ki != 0);
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
            lbi::mma::mma_ss<64, 64, 16, false, false>(
                uint64_t(d_dq0 + ki * 2), uint64_t(d_k + ki * 2), dS0, ki != 0);
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
            lbi::mma::mma_ss<64, 64, 16, false, false>(
                uint64_t(d_q + ki * 2), uint64_t(d_dk0 + ki * 2), dS0, 1);
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
            lbi::mma::mma_ss<64, 64, 16, false, false>(
                uint64_t(d_dq1 + ki * 2), uint64_t(d_k + ki * 2), dS1, ki != 0);
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
            lbi::mma::mma_ss<64, 64, 16, false, false>(
                uint64_t(d_q + ki * 2), uint64_t(d_dk1 + ki * 2), dS1, 1);
        lbi::mma::commit();
        lbi::mma::wait<0>();
        lbi::mma::fence_operand(S);
        lbi::mma::fence_operand(dS0);
        lbi::mma::fence_operand(dS1);

        const int col0 = t * BN;
        const bool diag = (t == pid_m);
        float rmax0 = -INFINITY, rmax1 = -INFINITY;
        if (diag) {
#pragma unroll
            for (int wb = 0; wb < 32; wb += lbi::mma::ACC_WINDOW) {
                S.window(wb);
                dS0.window(wb);
                dS1.window(wb);
            #pragma unroll
                for (int wj = 0; wj < lbi::mma::ACC_WINDOW; ++wj) {
                    const int reg = wb + wj;
                    int r, c; acc_rc(warp, lane, reg, r, c);
                    const bool dead = (col0 + c) > (row0 + r);
                    float sv = dead ? -INFINITY : S[reg] * scale;
                    S[reg] = sv;
                    dS0[reg] = dead ? 0.f : dS0[reg];
                    dS1[reg] = dead ? 0.f : dS1[reg];
                    if ((reg & 2) == 0) rmax0 = fmaxf(rmax0, sv); else rmax1 = fmaxf(rmax1, sv);
                }
            }
        } else {
#pragma unroll
            for (int wb = 0; wb < 32; wb += lbi::mma::ACC_WINDOW) {
                S.window(wb);
            #pragma unroll
                for (int wj = 0; wj < lbi::mma::ACC_WINDOW; ++wj) {
                    const int reg = wb + wj;
                    const float sv = S[reg] * scale;
                    S[reg] = sv;
                    if ((reg & 2) == 0) rmax0 = fmaxf(rmax0, sv); else rmax1 = fmaxf(rmax1, sv);
                }
            }
        }
#pragma unroll
        for (int d = 1; d <= 2; d <<= 1) {
            rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xffffffff, rmax0, d));
            rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xffffffff, rmax1, d));
        }
        const float mn0 = fmaxf(m0, rmax0), mn1 = fmaxf(m1, rmax1);
        const float a0 = __expf(m0 - mn0), a1 = __expf(m1 - mn1);
        const bool rescale = (a0 != 1.f) || (a1 != 1.f);
        m0 = mn0; m1 = mn1;

        float rs0 = 0.f, rs1 = 0.f, prs00 = 0.f, prs01 = 0.f, prs10 = 0.f, prs11 = 0.f;
        __syncthreads();   // previous epilogue reads of p/pds are done (wait below)
#pragma unroll
        for (int wb = 0; wb < 32; wb += lbi::mma::ACC_WINDOW) {
            S.window(wb);
            dS0.window(wb);
            dS1.window(wb);
        #pragma unroll
            for (int wj = 0; wj < lbi::mma::ACC_WINDOW; ++wj) {
                const int reg = wb + wj;
                int r, c; acc_rc(warp, lane, reg, r, c);
                const bool hi = (reg & 2) != 0;
                const float pv = __expf(S[reg] - (hi ? m1 : m0));
                const float pd0 = pv * dS0[reg] * scale, pd1 = pv * dS1[reg] * scale;
                const int so = sw128(r, c);
                p_s[so] = __float2bfloat16(pv);
                pds0_s[so] = __float2bfloat16(pd0);
                pds1_s[so] = __float2bfloat16(pd1);
                if (hi) { rs1 += pv; prs01 += pd0; prs11 += pd1; }
                else    { rs0 += pv; prs00 += pd0; prs10 += pd1; }
            }
        }
        if (rescale) {
#pragma unroll
            for (int wb = 0; wb < 32; wb += lbi::mma::ACC_WINDOW) {
                acc_o.window(wb);
                do0.window(wb);
                do1.window(wb);
            #pragma unroll
                for (int wj = 0; wj < lbi::mma::ACC_WINDOW; ++wj) {
                    const int reg = wb + wj;
                    const float a = (reg & 2) ? a1 : a0;
                    acc_o[reg] *= a; do0[reg] *= a; do1[reg] *= a;
                }
            }
        }
        __syncthreads();

        lbi::mma::fence_operand(acc_o);
        lbi::mma::fence_operand(do0);
        lbi::mma::fence_operand(do1);
        lbi::mma::arrive();
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
            lbi::mma::mma_ss<64, 64, 16, false, true>(
                uint64_t(d_p + ki * 2), uint64_t(d_v + ki * 128), acc_o, 1);
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
            lbi::mma::mma_ss<64, 64, 16, false, true>(
                uint64_t(d_pds0 + ki * 2), uint64_t(d_v + ki * 128), do0, 1);
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
            lbi::mma::mma_ss<64, 64, 16, false, true>(
                uint64_t(d_p + ki * 2), uint64_t(d_dv0 + ki * 128), do0, 1);
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
            lbi::mma::mma_ss<64, 64, 16, false, true>(
                uint64_t(d_pds1 + ki * 2), uint64_t(d_v + ki * 128), do1, 1);
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
            lbi::mma::mma_ss<64, 64, 16, false, true>(
                uint64_t(d_p + ki * 2), uint64_t(d_dv1 + ki * 128), do1, 1);
        lbi::mma::commit();

        // Scalar reductions overlap the epilogue wgmma batch.
#pragma unroll
        for (int d = 1; d <= 2; d <<= 1) {
            rs0 += __shfl_xor_sync(0xffffffff, rs0, d);
            rs1 += __shfl_xor_sync(0xffffffff, rs1, d);
            prs00 += __shfl_xor_sync(0xffffffff, prs00, d);
            prs01 += __shfl_xor_sync(0xffffffff, prs01, d);
            prs10 += __shfl_xor_sync(0xffffffff, prs10, d);
            prs11 += __shfl_xor_sync(0xffffffff, prs11, d);
        }
        l0 = l0 * a0 + rs0; l1 = l1 * a1 + rs1;
        as0_0 = as0_0 * a0 + prs00; as0_1 = as0_1 * a1 + prs01;
        as1_0 = as1_0 * a0 + prs10; as1_1 = as1_1 * a1 + prs11;

        lbi::mma::wait<0>();
        lbi::mma::fence_operand(acc_o);
        lbi::mma::fence_operand(do0);
        lbi::mma::fence_operand(do1);
        __syncthreads();
    }

    long obase = base, ostr = HD;
    if (Hout > 0) {
        const long b = bh / Hout, h = bh % Hout;
        obase = (b * Lctx * Hout + h) * HD;
        ostr = (long)Hout * HD;
    }
#pragma unroll
    for (int wb = 0; wb < 32; wb += lbi::mma::ACC_WINDOW) {
        acc_o.window(wb);
        do0.window(wb);
        do1.window(wb);
    #pragma unroll
        for (int wj = 0; wj < lbi::mma::ACC_WINDOW; ++wj) {
            const int reg = wb + wj;
            int r, c; acc_rc(warp, lane, reg, r, c);
            const bool hi = (reg & 2) != 0;
            const float li = hi ? l1 : l0;
            const float o = acc_o[reg] / li;
            const long g = obase + (long)(row0 + r) * ostr + c;
            O[g] = __float2bfloat16(o);
            DO[g] = __float2bfloat16((do0[reg] - (hi ? as0_1 : as0_0) * o) / li);
            DO[lane_str + g] = __float2bfloat16((do1[reg] - (hi ? as1_1 : as1_0) * o) / li);
        }
    }
}

}  // namespace flashjvp

// Full-r entry: loops direction pairs inside one op (single graph node); the
// primal O is recomputed per pair, identical bits, last write wins.
std::vector<torch::Tensor> flash_jvp_occ2_full(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor DQ, torch::Tensor DK, torch::Tensor DV, double scale) {
    using namespace flashjvp;
    TORCH_CHECK(Q.is_contiguous() && DQ.is_contiguous() && DK.is_contiguous()
                && DV.is_contiguous());
    const int R = DQ.size(0);
    TORCH_CHECK(R % 2 == 0);
    const int B = Q.size(0), H = Q.size(1), L = Q.size(2);
    TORCH_CHECK(Q.size(3) == HD && L % BM == 0);
    // flat outputs: out_proj consumes these without a transpose copy
    auto O = torch::empty({B, L, (long)H * HD}, Q.options());
    auto DO = torch::empty({R, B, L, (long)H * HD}, Q.options());
    cudaFuncSetAttribute(flash_jvp_occ2_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_OCC2);
    dim3 grid(L / BM, B * H);
    const int64_t ls = DQ.stride(0);
    auto* dq = reinterpret_cast<const __nv_bfloat16*>(DQ.data_ptr());
    auto* dk = reinterpret_cast<const __nv_bfloat16*>(DK.data_ptr());
    auto* dv = reinterpret_cast<const __nv_bfloat16*>(DV.data_ptr());
    auto* dout = reinterpret_cast<__nv_bfloat16*>(DO.data_ptr());
    for (int j = 0; j < R; j += 2) {
        flash_jvp_occ2_kernel<<<grid, 128, SMEM_OCC2, c10::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(K.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(V.data_ptr()),
            dq + j * ls, dk + j * ls, dv + j * ls,
            reinterpret_cast<__nv_bfloat16*>(O.data_ptr()),
            dout + j * ls, ls, L, (float)scale, H);
    }
    return {O, DO};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_jvp_occ2_full", &flash_jvp_occ2_full, "Fused flash-JVP over all r directions");
}
