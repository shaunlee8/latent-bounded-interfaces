// Fused flash-JVP kernel family, lane-pair (R=2), hd64, BM=BN=64, wgmma.
// Structural variants (baseline, opt, wg2, occ2, pipe, rs, pipe2) differ in
// KV staging, occupancy, and warpgroup schedule; the entry point is
// flash_jvp_occ2_full (occupancy-2, full-r loop, flat epilogue).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <tl_templates/cuda/instruction/wgmma.h>
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

// smem: q, dq0, dq1, k, v, dk0, dk1, dv0, dv1, p, pds  (11 x 8 KB)
constexpr int TILE = BM * HD;
constexpr size_t SMEM = (size_t)11 * TILE * 2;

__global__ void __launch_bounds__(128, 1) flash_jvp_kernel(
    const bf16* __restrict__ Q, const bf16* __restrict__ K,
    const bf16* __restrict__ V, const bf16* __restrict__ DQ,
    const bf16* __restrict__ DK, const bf16* __restrict__ DV,
    bf16* __restrict__ O, bf16* __restrict__ DO,
    long lane_str, int Lctx, float scale) {
    extern __shared__ __align__(1024) char smem[];
    bf16* q_s   = reinterpret_cast<bf16*>(smem);
    bf16* dq0_s = q_s + TILE;
    bf16* dq1_s = dq0_s + TILE;
    bf16* k_s   = dq1_s + TILE;
    bf16* v_s   = k_s + TILE;
    bf16* dk0_s = v_s + TILE;
    bf16* dk1_s = dk0_s + TILE;
    bf16* dv0_s = dk1_s + TILE;
    bf16* dv1_s = dv0_s + TILE;
    bf16* p_s   = dv1_s + TILE;
    bf16* pds_s = p_s + TILE;

    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int pid_m = blockIdx.x;
    const long bh = blockIdx.y;
    const long base = bh * Lctx * HD;
    const int row0 = pid_m * BM;

    // Stage the q-side tiles once (vectorized 8-elem rows, swizzled).
    for (int idx = tid; idx < BM * 8; idx += 128) {
        const int r = idx >> 3, u = idx & 7;
        const long g = base + (long)(row0 + r) * HD + u * 8;
        const int so = sw128(r, u * 8);
        *reinterpret_cast<uint4*>(q_s + so) = *reinterpret_cast<const uint4*>(Q + g);
        *reinterpret_cast<uint4*>(dq0_s + so) = *reinterpret_cast<const uint4*>(DQ + g);
        *reinterpret_cast<uint4*>(dq1_s + so) = *reinterpret_cast<const uint4*>(DQ + lane_str + g);
    }
    __syncthreads();

    tl::GmmaDescriptor d_q, d_dq0, d_dq1, d_k, d_v, d_dk0, d_dk1, d_dv0, d_dv1, d_p, d_pds;
    tl::initialize_wgmma_descriptor<1, 1, 64>(d_q, q_s);
    tl::initialize_wgmma_descriptor<1, 1, 64>(d_dq0, dq0_s);
    tl::initialize_wgmma_descriptor<1, 1, 64>(d_dq1, dq1_s);
    tl::initialize_wgmma_descriptor<1, 1, 64>(d_k, k_s);
    tl::initialize_wgmma_descriptor<1, 1024, 64>(d_v, v_s);
    tl::initialize_wgmma_descriptor<1, 1, 64>(d_dk0, dk0_s);
    tl::initialize_wgmma_descriptor<1, 1, 64>(d_dk1, dk1_s);
    tl::initialize_wgmma_descriptor<1, 1024, 64>(d_dv0, dv0_s);
    tl::initialize_wgmma_descriptor<1, 1024, 64>(d_dv1, dv1_s);
    tl::initialize_wgmma_descriptor<1, 1, 64>(d_p, p_s);
    tl::initialize_wgmma_descriptor<1, 1, 64>(d_pds, pds_s);

    float m0 = -INFINITY, m1 = -INFINITY;   // two rows per thread
    float l0 = 0.f, l1 = 0.f;
    float acc_o[32], do0[32], do1[32];
    float as0_0 = 0.f, as0_1 = 0.f, as1_0 = 0.f, as1_1 = 0.f;
#pragma unroll
    for (int i = 0; i < 32; ++i) { acc_o[i] = 0.f; do0[i] = 0.f; do1[i] = 0.f; }

    const int n_tiles = pid_m + 1;
    for (int t = 0; t < n_tiles; ++t) {
        const int col0 = t * BN;
        for (int idx = tid; idx < BN * 8; idx += 128) {
            const int r = idx >> 3, u = idx & 7;
            const long g = base + (long)(col0 + r) * HD + u * 8;
            const int so = sw128(r, u * 8);
            *reinterpret_cast<uint4*>(k_s + so) = *reinterpret_cast<const uint4*>(K + g);
            *reinterpret_cast<uint4*>(v_s + so) = *reinterpret_cast<const uint4*>(V + g);
            *reinterpret_cast<uint4*>(dk0_s + so) = *reinterpret_cast<const uint4*>(DK + g);
            *reinterpret_cast<uint4*>(dk1_s + so) = *reinterpret_cast<const uint4*>(DK + lane_str + g);
            *reinterpret_cast<uint4*>(dv0_s + so) = *reinterpret_cast<const uint4*>(DV + g);
            *reinterpret_cast<uint4*>(dv1_s + so) = *reinterpret_cast<const uint4*>(DV + lane_str + g);
        }
        __syncthreads();

        // S = q k^T; dS_l = dq_l k^T + q dk_l^T.  A [M,K], B [N,K] row-major
        // (transA=false, transB=false).
        float S[32], dS0[32], dS1[32];
        tl::warpgroup_fence_operand(S, 32);
        tl::warpgroup_fence_operand(dS0, 32);
        tl::warpgroup_fence_operand(dS1, 32);
        tl::warpgroup_arrive();
        tl::fence_proxy_async();
#pragma unroll
        for (int ki = 0; ki < 4; ++ki) {
            const int off = ki * 2;   // 16 bf16 cols = 32 B = 2 descriptor units
            tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                         tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(
                uint64_t(d_q + off), uint64_t(d_k + off),
                reinterpret_cast<uint32_t*>(S), ki != 0);
        }
#pragma unroll
        for (int ki = 0; ki < 4; ++ki) {
            const int off = ki * 2;
            tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                         tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(
                uint64_t(d_dq0 + off), uint64_t(d_k + off),
                reinterpret_cast<uint32_t*>(dS0), ki != 0);
        }
#pragma unroll
        for (int ki = 0; ki < 4; ++ki) {
            const int off = ki * 2;
            tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                         tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(
                uint64_t(d_q + off), uint64_t(d_dk0 + off),
                reinterpret_cast<uint32_t*>(dS0), 1);
        }
#pragma unroll
        for (int ki = 0; ki < 4; ++ki) {
            const int off = ki * 2;
            tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                         tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(
                uint64_t(d_dq1 + off), uint64_t(d_k + off),
                reinterpret_cast<uint32_t*>(dS1), ki != 0);
        }
#pragma unroll
        for (int ki = 0; ki < 4; ++ki) {
            const int off = ki * 2;
            tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                         tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(
                uint64_t(d_q + off), uint64_t(d_dk1 + off),
                reinterpret_cast<uint32_t*>(dS1), 1);
        }
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(S, 32);
        tl::warpgroup_fence_operand(dS0, 32);
        tl::warpgroup_fence_operand(dS1, 32);

        // Scale, causal mask, and the two per-thread row maxima.
        float rmax0 = -INFINITY, rmax1 = -INFINITY;
#pragma unroll
        for (int reg = 0; reg < 32; ++reg) {
            int r, c; acc_rc(warp, lane, reg, r, c);
            const bool dead = (col0 + c) > (row0 + r);
            float s = dead ? -INFINITY : S[reg] * scale;
            S[reg] = s;
            dS0[reg] = dead ? 0.f : dS0[reg] * scale;
            dS1[reg] = dead ? 0.f : dS1[reg] * scale;
            if ((reg & 2) == 0) rmax0 = fmaxf(rmax0, s); else rmax1 = fmaxf(rmax1, s);
        }
#pragma unroll
        for (int d = 1; d <= 2; d <<= 1) {
            rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xffffffff, rmax0, d));
            rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xffffffff, rmax1, d));
        }
        const float mn0 = fmaxf(m0, rmax0), mn1 = fmaxf(m1, rmax1);
        const float a0 = __expf(m0 - mn0), a1 = __expf(m1 - mn1);
        m0 = mn0; m1 = mn1;

        // p = exp(s - m); accumulate row sums; rescale carried accumulators.
        float rs0 = 0.f, rs1 = 0.f;
#pragma unroll
        for (int reg = 0; reg < 32; ++reg) {
            const bool hi = (reg & 2) != 0;
            const float mm = hi ? m1 : m0;
            const float p = __expf(S[reg] - mm);
            S[reg] = p;
            if (hi) rs1 += p; else rs0 += p;
        }
#pragma unroll
        for (int d = 1; d <= 2; d <<= 1) {
            rs0 += __shfl_xor_sync(0xffffffff, rs0, d);
            rs1 += __shfl_xor_sync(0xffffffff, rs1, d);
        }
        l0 = l0 * a0 + rs0; l1 = l1 * a1 + rs1;
#pragma unroll
        for (int reg = 0; reg < 32; ++reg) {
            const float a = (reg & 2) ? a1 : a0;
            acc_o[reg] *= a; do0[reg] *= a; do1[reg] *= a;
        }
        as0_0 *= a0; as0_1 *= a1; as1_0 *= a0; as1_1 *= a1;

        // Stage p; acc_o += p v (A [M,K=BN] @ B stored [K,N]: transB=true).
        __syncthreads();
#pragma unroll
        for (int reg = 0; reg < 32; ++reg) {
            int r, c; acc_rc(warp, lane, reg, r, c);
            p_s[sw128(r, c)] = __float2bfloat16(S[reg]);
        }
        __syncthreads();
        {
            float* accp = acc_o;
            tl::warpgroup_fence_operand(accp, 32);
            tl::warpgroup_arrive();
            tl::fence_proxy_async();
#pragma unroll
            for (int ki = 0; ki < 4; ++ki) {
                tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                             tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(
                    uint64_t(d_p + ki * 2), uint64_t(d_v + ki * 128),
                    reinterpret_cast<uint32_t*>(accp), 1);
            }
            tl::warpgroup_commit_batch();
            tl::warpgroup_wait<0>();
            tl::warpgroup_fence_operand(accp, 32);
        }

        // Lane 0: pds = p o dS0; do0 += pds v + p dv0; as0 += rowsum(pds).
        for (int lane_i = 0; lane_i < 2; ++lane_i) {
            float* dS = lane_i ? dS1 : dS0;
            float* acc = lane_i ? do1 : do0;
            float prs0 = 0.f, prs1 = 0.f;
            __syncthreads();
#pragma unroll
            for (int reg = 0; reg < 32; ++reg) {
                int r, c; acc_rc(warp, lane, reg, r, c);
                const float pd = S[reg] * dS[reg];
                pds_s[sw128(r, c)] = __float2bfloat16(pd);
                if ((reg & 2) != 0) prs1 += pd; else prs0 += pd;
            }
            __syncthreads();
#pragma unroll
            for (int d = 1; d <= 2; d <<= 1) {
                prs0 += __shfl_xor_sync(0xffffffff, prs0, d);
                prs1 += __shfl_xor_sync(0xffffffff, prs1, d);
            }
            if (lane_i == 0) { as0_0 += prs0; as0_1 += prs1; }
            else             { as1_0 += prs0; as1_1 += prs1; }
            tl::warpgroup_fence_operand(acc, 32);
            tl::warpgroup_arrive();
            tl::fence_proxy_async();
#pragma unroll
            for (int ki = 0; ki < 4; ++ki) {
                tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                             tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(
                    uint64_t(d_pds + ki * 2), uint64_t(d_v + ki * 128),
                    reinterpret_cast<uint32_t*>(acc), 1);
            }
#pragma unroll
            for (int ki = 0; ki < 4; ++ki) {
                tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                             tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(
                    uint64_t(d_p + ki * 2),
                    uint64_t((lane_i ? d_dv1 : d_dv0) + ki * 128),
                    reinterpret_cast<uint32_t*>(acc), 1);
            }
            tl::warpgroup_commit_batch();
            tl::warpgroup_wait<0>();
            tl::warpgroup_fence_operand(acc, 32);
        }
        __syncthreads();
    }

    // Epilogue: o = acc_o/l; do_l = (do_l - as_l * o)/l.
#pragma unroll
    for (int reg = 0; reg < 32; ++reg) {
        int r, c; acc_rc(warp, lane, reg, r, c);
        const bool hi = (reg & 2) != 0;
        const float li = hi ? l1 : l0;
        const float o = acc_o[reg] / li;
        const long g = base + (long)(row0 + r) * HD + c;
        O[g] = __float2bfloat16(o);
        DO[g] = __float2bfloat16((do0[reg] - (hi ? as0_1 : as0_0) * o) / li);
        DO[lane_str + g] = __float2bfloat16((do1[reg] - (hi ? as1_1 : as1_0) * o) / li);
    }
}


// ---- Optimized variant: cp.async double-buffered KV staging, one merged
// p/pds store phase, single epilogue commit batch. ----

__device__ __forceinline__ void cp16(bf16* dst, const bf16* src) {
    unsigned sdst = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(sdst), "l"(src));
}

// smem: q,dq0,dq1 (3) + KV double buffer (2x6) + p,pds0,pds1 (3) = 18 tiles.
constexpr size_t SMEM_OPT = (size_t)18 * TILE * 2;

__global__ void __launch_bounds__(128, 1) flash_jvp_opt_kernel(
    const bf16* __restrict__ Q, const bf16* __restrict__ K,
    const bf16* __restrict__ V, const bf16* __restrict__ DQ,
    const bf16* __restrict__ DK, const bf16* __restrict__ DV,
    bf16* __restrict__ O, bf16* __restrict__ DO,
    long lane_str, int Lctx, float scale) {
    extern __shared__ __align__(1024) char smem[];
    bf16* q_s   = reinterpret_cast<bf16*>(smem);
    bf16* dq0_s = q_s + TILE;
    bf16* dq1_s = dq0_s + TILE;
    bf16* kvbuf = dq1_s + TILE;               // [2][6][TILE]
    bf16* p_s   = kvbuf + 12 * TILE;
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
        bf16* buf = kvbuf + (t & 1) * 6 * TILE;
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

    prefetch(0);

    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
    float acc_o[32], do0[32], do1[32];
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
        bf16* buf = kvbuf + (t & 1) * 6 * TILE;
        tl::GmmaDescriptor d_k, d_v, d_dk0, d_dk1, d_dv0, d_dv1;
        tl::initialize_wgmma_descriptor<1, 1, 64>(d_k, buf);
        tl::initialize_wgmma_descriptor<1, 1024, 64>(d_v, buf + TILE);
        tl::initialize_wgmma_descriptor<1, 1, 64>(d_dk0, buf + 2 * TILE);
        tl::initialize_wgmma_descriptor<1, 1, 64>(d_dk1, buf + 3 * TILE);
        tl::initialize_wgmma_descriptor<1, 1024, 64>(d_dv0, buf + 4 * TILE);
        tl::initialize_wgmma_descriptor<1, 1024, 64>(d_dv1, buf + 5 * TILE);

        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        __syncthreads();
        prefetch(t + 1);

        float S[32], dS0[32], dS1[32];
        tl::warpgroup_fence_operand(S, 32);
        tl::warpgroup_fence_operand(dS0, 32);
        tl::warpgroup_fence_operand(dS1, 32);
        tl::warpgroup_arrive();
        tl::fence_proxy_async();
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
            tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                         tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(
                uint64_t(d_q + ki * 2), uint64_t(d_k + ki * 2),
                reinterpret_cast<uint32_t*>(S), ki != 0);
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
            tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                         tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(
                uint64_t(d_dq0 + ki * 2), uint64_t(d_k + ki * 2),
                reinterpret_cast<uint32_t*>(dS0), ki != 0);
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
            tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                         tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(
                uint64_t(d_q + ki * 2), uint64_t(d_dk0 + ki * 2),
                reinterpret_cast<uint32_t*>(dS0), 1);
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
            tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                         tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(
                uint64_t(d_dq1 + ki * 2), uint64_t(d_k + ki * 2),
                reinterpret_cast<uint32_t*>(dS1), ki != 0);
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
            tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                         tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(
                uint64_t(d_q + ki * 2), uint64_t(d_dk1 + ki * 2),
                reinterpret_cast<uint32_t*>(dS1), 1);
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(S, 32);
        tl::warpgroup_fence_operand(dS0, 32);
        tl::warpgroup_fence_operand(dS1, 32);

        const int col0 = t * BN;
        float rmax0 = -INFINITY, rmax1 = -INFINITY;
#pragma unroll
        for (int reg = 0; reg < 32; ++reg) {
            int r, c; acc_rc(warp, lane, reg, r, c);
            const bool dead = (col0 + c) > (row0 + r);
            float sv = dead ? -INFINITY : S[reg] * scale;
            S[reg] = sv;
            dS0[reg] = dead ? 0.f : dS0[reg] * scale;
            dS1[reg] = dead ? 0.f : dS1[reg] * scale;
            if ((reg & 2) == 0) rmax0 = fmaxf(rmax0, sv); else rmax1 = fmaxf(rmax1, sv);
        }
#pragma unroll
        for (int d = 1; d <= 2; d <<= 1) {
            rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xffffffff, rmax0, d));
            rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xffffffff, rmax1, d));
        }
        const float mn0 = fmaxf(m0, rmax0), mn1 = fmaxf(m1, rmax1);
        const float a0 = __expf(m0 - mn0), a1 = __expf(m1 - mn1);
        m0 = mn0; m1 = mn1;

        float rs0 = 0.f, rs1 = 0.f, prs00 = 0.f, prs01 = 0.f, prs10 = 0.f, prs11 = 0.f;
        __syncthreads();   // previous epilogue reads of p/pds are done (wait below)
#pragma unroll
        for (int reg = 0; reg < 32; ++reg) {
            int r, c; acc_rc(warp, lane, reg, r, c);
            const bool hi = (reg & 2) != 0;
            const float pv = __expf(S[reg] - (hi ? m1 : m0));
            const float pd0 = pv * dS0[reg], pd1 = pv * dS1[reg];
            const int so = sw128(r, c);
            p_s[so] = __float2bfloat16(pv);
            pds0_s[so] = __float2bfloat16(pd0);
            pds1_s[so] = __float2bfloat16(pd1);
            if (hi) { rs1 += pv; prs01 += pd0; prs11 += pd1; }
            else    { rs0 += pv; prs00 += pd0; prs10 += pd1; }
            const float a = hi ? a1 : a0;
            acc_o[reg] *= a; do0[reg] *= a; do1[reg] *= a;
        }
        __syncthreads();

        tl::warpgroup_fence_operand(acc_o, 32);
        tl::warpgroup_fence_operand(do0, 32);
        tl::warpgroup_fence_operand(do1, 32);
        tl::warpgroup_arrive();
        tl::fence_proxy_async();
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
            tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                         tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(
                uint64_t(d_p + ki * 2), uint64_t(d_v + ki * 128),
                reinterpret_cast<uint32_t*>(acc_o), 1);
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
            tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                         tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(
                uint64_t(d_pds0 + ki * 2), uint64_t(d_v + ki * 128),
                reinterpret_cast<uint32_t*>(do0), 1);
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
            tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                         tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(
                uint64_t(d_p + ki * 2), uint64_t(d_dv0 + ki * 128),
                reinterpret_cast<uint32_t*>(do0), 1);
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
            tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                         tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(
                uint64_t(d_pds1 + ki * 2), uint64_t(d_v + ki * 128),
                reinterpret_cast<uint32_t*>(do1), 1);
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
            tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                         tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(
                uint64_t(d_p + ki * 2), uint64_t(d_dv1 + ki * 128),
                reinterpret_cast<uint32_t*>(do1), 1);
        tl::warpgroup_commit_batch();

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

        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(acc_o, 32);
        tl::warpgroup_fence_operand(do0, 32);
        tl::warpgroup_fence_operand(do1, 32);
    }

#pragma unroll
    for (int reg = 0; reg < 32; ++reg) {
        int r, c; acc_rc(warp, lane, reg, r, c);
        const bool hi = (reg & 2) != 0;
        const float li = hi ? l1 : l0;
        const float o = acc_o[reg] / li;
        const long g = base + (long)(row0 + r) * HD + c;
        O[g] = __float2bfloat16(o);
        DO[g] = __float2bfloat16((do0[reg] - (hi ? as0_1 : as0_0) * o) / li);
        DO[lane_str + g] = __float2bfloat16((do1[reg] - (hi ? as1_1 : as1_0) * o) / li);
    }
}


// ---- Two-warpgroup variant: BM=128 (64 rows per warpgroup), BN=32 KV tiles
// shared across warpgroups and double-buffered; per-warpgroup p/pds staging
// (column-padded to keep the SW128 layout); m64n32 score fragments. ----

constexpr int BM2 = 128, BN2 = 32;
constexpr int KVT = BN2 * HD;                  // 2 KB tile
// smem: q,dq0,dq1 [128][64] (3x16KB) + KV 2x6x[32][64] (48KB) +
// p,pds0,pds1 per wg [64][64] (6x8KB) = 144 KB.
constexpr size_t SMEM_WG2 = (size_t)(3 * BM2 * HD + 12 * KVT + 6 * TILE) * 2;

__global__ void __launch_bounds__(256, 1) flash_jvp_wg2_kernel(
    const bf16* __restrict__ Q, const bf16* __restrict__ K,
    const bf16* __restrict__ V, const bf16* __restrict__ DQ,
    const bf16* __restrict__ DK, const bf16* __restrict__ DV,
    bf16* __restrict__ O, bf16* __restrict__ DO,
    long lane_str, int Lctx, float scale) {
    extern __shared__ __align__(1024) char smem[];
    bf16* q_s   = reinterpret_cast<bf16*>(smem);          // [128][64]
    bf16* dq0_s = q_s + BM2 * HD;
    bf16* dq1_s = dq0_s + BM2 * HD;
    bf16* kvbuf = dq1_s + BM2 * HD;                       // [2][6][KVT]
    bf16* pstg  = kvbuf + 12 * KVT;                       // [2 wg][3][TILE]

    const int tid = threadIdx.x;
    const int wg = tid >> 7;
    const int wtid = tid & 127;
    const int warp = wtid >> 5, lane = wtid & 31;
    const int pid_m = blockIdx.x;
    const long bh = blockIdx.y;
    const long base = bh * Lctx * HD;
    const int row0 = pid_m * BM2;
    const int wrow0 = row0 + wg * 64;

    for (int idx = tid; idx < BM2 * 8; idx += 256) {
        const int r = idx >> 3, u = idx & 7;
        const long g = base + (long)(row0 + r) * HD + u * 8;
        const int so = sw128(r, u * 8);
        *reinterpret_cast<uint4*>(q_s + so) = *reinterpret_cast<const uint4*>(Q + g);
        *reinterpret_cast<uint4*>(dq0_s + so) = *reinterpret_cast<const uint4*>(DQ + g);
        *reinterpret_cast<uint4*>(dq1_s + so) = *reinterpret_cast<const uint4*>(DQ + lane_str + g);
    }

    const int n_tiles = (pid_m + 1) * 4;
    const bf16* gsrc[6] = {K, V, DK, DK, DV, DV};
    const long goff[6] = {0, 0, 0, lane_str, 0, lane_str};

    auto prefetch = [&](int t) {
        if (t >= n_tiles) return;
        bf16* buf = kvbuf + (t & 1) * 6 * KVT;
        const long col_base = base + (long)(t * BN2) * HD;
        for (int idx = tid; idx < BN2 * 8; idx += 256) {
            const int r = idx >> 3, u = idx & 7;
            const int so = sw128(r, u * 8);
            const long g = col_base + (long)r * HD + u * 8;
#pragma unroll
            for (int j = 0; j < 6; ++j)
                cp16(buf + j * KVT + so, gsrc[j] + goff[j] + g);
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    };

    prefetch(0);

    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
    float acc_o[32], do0[32], do1[32];
    float as0_0 = 0.f, as0_1 = 0.f, as1_0 = 0.f, as1_1 = 0.f;
#pragma unroll
    for (int i = 0; i < 32; ++i) { acc_o[i] = 0.f; do0[i] = 0.f; do1[i] = 0.f; }

    bf16* q_wg = q_s + wg * 64 * HD;
    bf16* dq0_wg = dq0_s + wg * 64 * HD;
    bf16* dq1_wg = dq1_s + wg * 64 * HD;
    bf16* p_s = pstg + wg * 3 * TILE;
    bf16* pds0_s = p_s + TILE;
    bf16* pds1_s = pds0_s + TILE;
    tl::GmmaDescriptor d_q, d_dq0, d_dq1, d_p, d_pds0, d_pds1;
    tl::initialize_wgmma_descriptor<1, 1, 64>(d_q, q_wg);
    tl::initialize_wgmma_descriptor<1, 1, 64>(d_dq0, dq0_wg);
    tl::initialize_wgmma_descriptor<1, 1, 64>(d_dq1, dq1_wg);
    tl::initialize_wgmma_descriptor<1, 1, 64>(d_p, p_s);
    tl::initialize_wgmma_descriptor<1, 1, 64>(d_pds0, pds0_s);
    tl::initialize_wgmma_descriptor<1, 1, 64>(d_pds1, pds1_s);
    __syncthreads();

    for (int t = 0; t < n_tiles; ++t) {
        bf16* buf = kvbuf + (t & 1) * 6 * KVT;
        tl::GmmaDescriptor d_k, d_v, d_dk0, d_dk1, d_dv0, d_dv1;
        tl::initialize_wgmma_descriptor<1, 1, 64>(d_k, buf);
        tl::initialize_wgmma_descriptor<1, 1024, 64>(d_v, buf + KVT);
        tl::initialize_wgmma_descriptor<1, 1, 64>(d_dk0, buf + 2 * KVT);
        tl::initialize_wgmma_descriptor<1, 1, 64>(d_dk1, buf + 3 * KVT);
        tl::initialize_wgmma_descriptor<1, 1024, 64>(d_dv0, buf + 4 * KVT);
        tl::initialize_wgmma_descriptor<1, 1024, 64>(d_dv1, buf + 5 * KVT);

        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        __syncthreads();
        prefetch(t + 1);

        float S[16], dS0[16], dS1[16];
        tl::warpgroup_fence_operand(S, 16);
        tl::warpgroup_fence_operand(dS0, 16);
        tl::warpgroup_fence_operand(dS1, 16);
        tl::warpgroup_arrive();
        tl::fence_proxy_async();
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
            tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                         tl::DataType::kFloat32, 64, 32, 16, false, false, 1, 1>(
                uint64_t(d_q + ki * 2), uint64_t(d_k + ki * 2),
                reinterpret_cast<uint32_t*>(S), ki != 0);
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
            tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                         tl::DataType::kFloat32, 64, 32, 16, false, false, 1, 1>(
                uint64_t(d_dq0 + ki * 2), uint64_t(d_k + ki * 2),
                reinterpret_cast<uint32_t*>(dS0), ki != 0);
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
            tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                         tl::DataType::kFloat32, 64, 32, 16, false, false, 1, 1>(
                uint64_t(d_q + ki * 2), uint64_t(d_dk0 + ki * 2),
                reinterpret_cast<uint32_t*>(dS0), 1);
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
            tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                         tl::DataType::kFloat32, 64, 32, 16, false, false, 1, 1>(
                uint64_t(d_dq1 + ki * 2), uint64_t(d_k + ki * 2),
                reinterpret_cast<uint32_t*>(dS1), ki != 0);
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
            tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                         tl::DataType::kFloat32, 64, 32, 16, false, false, 1, 1>(
                uint64_t(d_q + ki * 2), uint64_t(d_dk1 + ki * 2),
                reinterpret_cast<uint32_t*>(dS1), 1);
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(S, 16);
        tl::warpgroup_fence_operand(dS0, 16);
        tl::warpgroup_fence_operand(dS1, 16);

        const int col0 = t * BN2;
        float rmax0 = -INFINITY, rmax1 = -INFINITY;
#pragma unroll
        for (int reg = 0; reg < 16; ++reg) {
            int r, c; acc_rc(warp, lane, reg, r, c);
            const bool dead = (col0 + c) > (wrow0 + r);
            float sv = dead ? -INFINITY : S[reg] * scale;
            S[reg] = sv;
            dS0[reg] = dead ? 0.f : dS0[reg] * scale;
            dS1[reg] = dead ? 0.f : dS1[reg] * scale;
            if ((reg & 2) == 0) rmax0 = fmaxf(rmax0, sv); else rmax1 = fmaxf(rmax1, sv);
        }
#pragma unroll
        for (int d = 1; d <= 2; d <<= 1) {
            rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xffffffff, rmax0, d));
            rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xffffffff, rmax1, d));
        }
        const float mn0 = fmaxf(m0, rmax0), mn1 = fmaxf(m1, rmax1);
        const float a0 = __expf(m0 - mn0), a1 = __expf(m1 - mn1);
        m0 = mn0; m1 = mn1;

        float rs0 = 0.f, rs1 = 0.f, prs00 = 0.f, prs01 = 0.f, prs10 = 0.f, prs11 = 0.f;
        asm volatile("bar.sync %0, %1;" :: "r"(1 + wg), "r"(128));   // wg-local barrier
#pragma unroll
        for (int reg = 0; reg < 16; ++reg) {
            int r, c; acc_rc(warp, lane, reg, r, c);
            const bool hi = (reg & 2) != 0;
            const float pv = __expf(S[reg] - (hi ? m1 : m0));
            const float pd0 = pv * dS0[reg], pd1 = pv * dS1[reg];
            const int so = sw128(r, c);
            p_s[so] = __float2bfloat16(pv);
            pds0_s[so] = __float2bfloat16(pd0);
            pds1_s[so] = __float2bfloat16(pd1);
            if (hi) { rs1 += pv; prs01 += pd0; prs11 += pd1; }
            else    { rs0 += pv; prs00 += pd0; prs10 += pd1; }
            const float a = hi ? a1 : a0;
            acc_o[reg] *= a; acc_o[reg + 16] *= a;
            do0[reg] *= a; do0[reg + 16] *= a;
            do1[reg] *= a; do1[reg + 16] *= a;
        }
        asm volatile("bar.sync %0, %1;" :: "r"(1 + wg), "r"(128));

        tl::warpgroup_fence_operand(acc_o, 32);
        tl::warpgroup_fence_operand(do0, 32);
        tl::warpgroup_fence_operand(do1, 32);
        tl::warpgroup_arrive();
        tl::fence_proxy_async();
#pragma unroll
        for (int ki = 0; ki < 2; ++ki)
            tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                         tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(
                uint64_t(d_p + ki * 2), uint64_t(d_v + ki * 128),
                reinterpret_cast<uint32_t*>(acc_o), 1);
#pragma unroll
        for (int ki = 0; ki < 2; ++ki)
            tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                         tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(
                uint64_t(d_pds0 + ki * 2), uint64_t(d_v + ki * 128),
                reinterpret_cast<uint32_t*>(do0), 1);
#pragma unroll
        for (int ki = 0; ki < 2; ++ki)
            tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                         tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(
                uint64_t(d_p + ki * 2), uint64_t(d_dv0 + ki * 128),
                reinterpret_cast<uint32_t*>(do0), 1);
#pragma unroll
        for (int ki = 0; ki < 2; ++ki)
            tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                         tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(
                uint64_t(d_pds1 + ki * 2), uint64_t(d_v + ki * 128),
                reinterpret_cast<uint32_t*>(do1), 1);
#pragma unroll
        for (int ki = 0; ki < 2; ++ki)
            tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                         tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(
                uint64_t(d_p + ki * 2), uint64_t(d_dv1 + ki * 128),
                reinterpret_cast<uint32_t*>(do1), 1);
        tl::warpgroup_commit_batch();

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

        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(acc_o, 32);
        tl::warpgroup_fence_operand(do0, 32);
        tl::warpgroup_fence_operand(do1, 32);
    }

#pragma unroll
    for (int reg = 0; reg < 32; ++reg) {
        int r, c; acc_rc(warp, lane, reg, r, c);
        const bool hi = (reg & 2) != 0;
        const float li = hi ? l1 : l0;
        const float o = acc_o[reg] / li;
        const long g = base + (long)(wrow0 + r) * HD + c;
        O[g] = __float2bfloat16(o);
        DO[g] = __float2bfloat16((do0[reg] - (hi ? as0_1 : as0_0) * o) / li);
        DO[lane_str + g] = __float2bfloat16((do1[reg] - (hi ? as1_1 : as1_0) * o) / li);
    }
}

// ---- Occupancy-2 variant: the opt kernel with SINGLE-buffered KV tiles
// (96 KB smem => 2 CTAs/SM); the co-resident CTA provides the cross-phase
// overlap that double-buffering provided at occupancy 1. ----


// smem: q,dq0,dq1 (3) + KV double buffer (2x6) + p,pds0,pds1 (3) = 18 tiles.
constexpr size_t SMEM_OCC2 = (size_t)12 * TILE * 2;

// Hout == 0: O/DO in [B,H,L,hd]. Hout > 0: flat [B,L,Hout*hd] layout so the
// out_proj GEMM consumes the epilogue directly (per accumulator row the store
// is still one contiguous 64-channel segment; only the stride changes).
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
    float acc_o[32], do0[32], do1[32];
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

        float S[32], dS0[32], dS1[32];
        tl::warpgroup_fence_operand(S, 32);
        tl::warpgroup_fence_operand(dS0, 32);
        tl::warpgroup_fence_operand(dS1, 32);
        tl::warpgroup_arrive();
        tl::fence_proxy_async();
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
            tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                         tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(
                uint64_t(d_q + ki * 2), uint64_t(d_k + ki * 2),
                reinterpret_cast<uint32_t*>(S), ki != 0);
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
            tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                         tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(
                uint64_t(d_dq0 + ki * 2), uint64_t(d_k + ki * 2),
                reinterpret_cast<uint32_t*>(dS0), ki != 0);
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
            tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                         tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(
                uint64_t(d_q + ki * 2), uint64_t(d_dk0 + ki * 2),
                reinterpret_cast<uint32_t*>(dS0), 1);
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
            tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                         tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(
                uint64_t(d_dq1 + ki * 2), uint64_t(d_k + ki * 2),
                reinterpret_cast<uint32_t*>(dS1), ki != 0);
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
            tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                         tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(
                uint64_t(d_q + ki * 2), uint64_t(d_dk1 + ki * 2),
                reinterpret_cast<uint32_t*>(dS1), 1);
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(S, 32);
        tl::warpgroup_fence_operand(dS0, 32);
        tl::warpgroup_fence_operand(dS1, 32);

        const int col0 = t * BN;
        const bool diag = (t == pid_m);
        float rmax0 = -INFINITY, rmax1 = -INFINITY;
        if (diag) {
#pragma unroll
            for (int reg = 0; reg < 32; ++reg) {
                int r, c; acc_rc(warp, lane, reg, r, c);
                const bool dead = (col0 + c) > (row0 + r);
                float sv = dead ? -INFINITY : S[reg] * scale;
                S[reg] = sv;
                dS0[reg] = dead ? 0.f : dS0[reg];
                dS1[reg] = dead ? 0.f : dS1[reg];
                if ((reg & 2) == 0) rmax0 = fmaxf(rmax0, sv); else rmax1 = fmaxf(rmax1, sv);
            }
        } else {
#pragma unroll
            for (int reg = 0; reg < 32; ++reg) {
                const float sv = S[reg] * scale;
                S[reg] = sv;
                if ((reg & 2) == 0) rmax0 = fmaxf(rmax0, sv); else rmax1 = fmaxf(rmax1, sv);
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
        for (int reg = 0; reg < 32; ++reg) {
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
        if (rescale) {
#pragma unroll
            for (int reg = 0; reg < 32; ++reg) {
                const float a = (reg & 2) ? a1 : a0;
                acc_o[reg] *= a; do0[reg] *= a; do1[reg] *= a;
            }
        }
        __syncthreads();

        tl::warpgroup_fence_operand(acc_o, 32);
        tl::warpgroup_fence_operand(do0, 32);
        tl::warpgroup_fence_operand(do1, 32);
        tl::warpgroup_arrive();
        tl::fence_proxy_async();
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
            tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                         tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(
                uint64_t(d_p + ki * 2), uint64_t(d_v + ki * 128),
                reinterpret_cast<uint32_t*>(acc_o), 1);
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
            tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                         tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(
                uint64_t(d_pds0 + ki * 2), uint64_t(d_v + ki * 128),
                reinterpret_cast<uint32_t*>(do0), 1);
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
            tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                         tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(
                uint64_t(d_p + ki * 2), uint64_t(d_dv0 + ki * 128),
                reinterpret_cast<uint32_t*>(do0), 1);
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
            tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                         tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(
                uint64_t(d_pds1 + ki * 2), uint64_t(d_v + ki * 128),
                reinterpret_cast<uint32_t*>(do1), 1);
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
            tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                         tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(
                uint64_t(d_p + ki * 2), uint64_t(d_dv1 + ki * 128),
                reinterpret_cast<uint32_t*>(do1), 1);
        tl::warpgroup_commit_batch();

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

        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(acc_o, 32);
        tl::warpgroup_fence_operand(do0, 32);
        tl::warpgroup_fence_operand(do1, 32);
        __syncthreads();
    }

    long obase = base, ostr = HD;
    if (Hout > 0) {
        const long b = bh / Hout, h = bh % Hout;
        obase = (b * Lctx * Hout + h) * HD;
        ostr = (long)Hout * HD;
    }
#pragma unroll
    for (int reg = 0; reg < 32; ++reg) {
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


// ---- Pipelined variant: double-buffered KV; tile t+1's S/dS batch is
// issued BEHIND tile t's epilogue batch, and in-order group retirement
// (wait<1>) drains the epilogue while the next S-batch flies. S/dS
// registers are reused across tiles (dead after the p/pds store phase). ----

constexpr size_t SMEM_PIPE = (size_t)18 * TILE * 2;

__global__ void __launch_bounds__(128, 1) flash_jvp_pipe_kernel(
    const bf16* __restrict__ Q, const bf16* __restrict__ K,
    const bf16* __restrict__ V, const bf16* __restrict__ DQ,
    const bf16* __restrict__ DK, const bf16* __restrict__ DV,
    bf16* __restrict__ O, bf16* __restrict__ DO,
    long lane_str, int Lctx, float scale) {
    extern __shared__ __align__(1024) char smem[];
    bf16* q_s   = reinterpret_cast<bf16*>(smem);
    bf16* dq0_s = q_s + TILE;
    bf16* dq1_s = dq0_s + TILE;
    bf16* kvbuf = dq1_s + TILE;               // [2][6][TILE]
    bf16* p_s   = kvbuf + 12 * TILE;
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
        bf16* buf = kvbuf + (t & 1) * 6 * TILE;
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
    float acc_o[32], do0[32], do1[32];
    float S[32], dS0[32], dS1[32];
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

    auto issue_sbatch = [&](int t) {
        bf16* buf = kvbuf + (t & 1) * 6 * TILE;
        tl::GmmaDescriptor d_k, d_dk0, d_dk1;
        tl::initialize_wgmma_descriptor<1, 1, 64>(d_k, buf);
        tl::initialize_wgmma_descriptor<1, 1, 64>(d_dk0, buf + 2 * TILE);
        tl::initialize_wgmma_descriptor<1, 1, 64>(d_dk1, buf + 3 * TILE);
        tl::warpgroup_fence_operand(S, 32);
        tl::warpgroup_fence_operand(dS0, 32);
        tl::warpgroup_fence_operand(dS1, 32);
        tl::warpgroup_arrive();
        tl::fence_proxy_async();
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
            tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                         tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(
                uint64_t(d_q + ki * 2), uint64_t(d_k + ki * 2),
                reinterpret_cast<uint32_t*>(S), ki != 0);
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
            tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                         tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(
                uint64_t(d_dq0 + ki * 2), uint64_t(d_k + ki * 2),
                reinterpret_cast<uint32_t*>(dS0), ki != 0);
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
            tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                         tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(
                uint64_t(d_q + ki * 2), uint64_t(d_dk0 + ki * 2),
                reinterpret_cast<uint32_t*>(dS0), 1);
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
            tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                         tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(
                uint64_t(d_dq1 + ki * 2), uint64_t(d_k + ki * 2),
                reinterpret_cast<uint32_t*>(dS1), ki != 0);
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
            tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                         tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(
                uint64_t(d_q + ki * 2), uint64_t(d_dk1 + ki * 2),
                reinterpret_cast<uint32_t*>(dS1), 1);
        tl::warpgroup_commit_batch();
    };

    // Pipeline fill: KV(0), KV(1) in flight; S-batch(0) issued.
    prefetch(0);
    prefetch(1);
    if (n_tiles > 1) {
        asm volatile("cp.async.wait_group 1;\n" ::: "memory");
    } else {
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    }
    __syncthreads();
    issue_sbatch(0);

    for (int t = 0; t < n_tiles; ++t) {
        bf16* buf = kvbuf + (t & 1) * 6 * TILE;
        tl::GmmaDescriptor d_v, d_dv0, d_dv1;
        tl::initialize_wgmma_descriptor<1, 1024, 64>(d_v, buf + TILE);
        tl::initialize_wgmma_descriptor<1, 1024, 64>(d_dv0, buf + 4 * TILE);
        tl::initialize_wgmma_descriptor<1, 1024, 64>(d_dv1, buf + 5 * TILE);

        tl::warpgroup_wait<0>();               // S/dS(t) ready; epilogue(t-1) drained
        tl::warpgroup_fence_operand(S, 32);
        tl::warpgroup_fence_operand(dS0, 32);
        tl::warpgroup_fence_operand(dS1, 32);

        const int col0 = t * BN;
        float rmax0 = -INFINITY, rmax1 = -INFINITY;
#pragma unroll
        for (int reg = 0; reg < 32; ++reg) {
            int r, c; acc_rc(warp, lane, reg, r, c);
            const bool dead = (col0 + c) > (row0 + r);
            float sv = dead ? -INFINITY : S[reg] * scale;
            S[reg] = sv;
            dS0[reg] = dead ? 0.f : dS0[reg] * scale;
            dS1[reg] = dead ? 0.f : dS1[reg] * scale;
            if ((reg & 2) == 0) rmax0 = fmaxf(rmax0, sv); else rmax1 = fmaxf(rmax1, sv);
        }
#pragma unroll
        for (int d = 1; d <= 2; d <<= 1) {
            rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xffffffff, rmax0, d));
            rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xffffffff, rmax1, d));
        }
        const float mn0 = fmaxf(m0, rmax0), mn1 = fmaxf(m1, rmax1);
        const float a0 = __expf(m0 - mn0), a1 = __expf(m1 - mn1);
        m0 = mn0; m1 = mn1;

        float rs0 = 0.f, rs1 = 0.f, prs00 = 0.f, prs01 = 0.f, prs10 = 0.f, prs11 = 0.f;
#pragma unroll
        for (int reg = 0; reg < 32; ++reg) {
            int r, c; acc_rc(warp, lane, reg, r, c);
            const bool hi = (reg & 2) != 0;
            const float pv = __expf(S[reg] - (hi ? m1 : m0));
            const float pd0 = pv * dS0[reg], pd1 = pv * dS1[reg];
            const int so = sw128(r, c);
            p_s[so] = __float2bfloat16(pv);
            pds0_s[so] = __float2bfloat16(pd0);
            pds1_s[so] = __float2bfloat16(pd1);
            if (hi) { rs1 += pv; prs01 += pd0; prs11 += pd1; }
            else    { rs0 += pv; prs00 += pd0; prs10 += pd1; }
            const float a = hi ? a1 : a0;
            acc_o[reg] *= a; do0[reg] *= a; do1[reg] *= a;
        }
        __syncthreads();

        // Epilogue(t) first (older group), then S-batch(t+1) behind it.
        tl::warpgroup_fence_operand(acc_o, 32);
        tl::warpgroup_fence_operand(do0, 32);
        tl::warpgroup_fence_operand(do1, 32);
        tl::warpgroup_arrive();
        tl::fence_proxy_async();
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
            tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                         tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(
                uint64_t(d_p + ki * 2), uint64_t(d_v + ki * 128),
                reinterpret_cast<uint32_t*>(acc_o), 1);
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
            tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                         tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(
                uint64_t(d_pds0 + ki * 2), uint64_t(d_v + ki * 128),
                reinterpret_cast<uint32_t*>(do0), 1);
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
            tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                         tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(
                uint64_t(d_p + ki * 2), uint64_t(d_dv0 + ki * 128),
                reinterpret_cast<uint32_t*>(do0), 1);
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
            tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                         tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(
                uint64_t(d_pds1 + ki * 2), uint64_t(d_v + ki * 128),
                reinterpret_cast<uint32_t*>(do1), 1);
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
            tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                         tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(
                uint64_t(d_p + ki * 2), uint64_t(d_dv1 + ki * 128),
                reinterpret_cast<uint32_t*>(do1), 1);
        tl::warpgroup_commit_batch();

        if (t + 1 < n_tiles) {
            asm volatile("cp.async.wait_group 0;\n" ::: "memory");
            __syncthreads();
            issue_sbatch(t + 1);
        }

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

        if (t + 1 < n_tiles) {
            tl::warpgroup_wait<1>();           // epilogue(t) drained; S(t+1) flies
        } else {
            tl::warpgroup_wait<0>();
        }
        tl::warpgroup_fence_operand(acc_o, 32);
        tl::warpgroup_fence_operand(do0, 32);
        tl::warpgroup_fence_operand(do1, 32);
        __syncthreads();                       // p/pds free before next stores
        prefetch(t + 2);
    }

#pragma unroll
    for (int reg = 0; reg < 32; ++reg) {
        int r, c; acc_rc(warp, lane, reg, r, c);
        const bool hi = (reg & 2) != 0;
        const float li = hi ? l1 : l0;
        const float o = acc_o[reg] / li;
        const long g = base + (long)(row0 + r) * HD + c;
        O[g] = __float2bfloat16(o);
        DO[g] = __float2bfloat16((do0[reg] - (hi ? as0_1 : as0_0) * o) / li);
        DO[lane_str + g] = __float2bfloat16((do1[reg] - (hi ? as1_1 : as1_0) * o) / li);
    }
}

}  // namespace flashjvp

std::vector<torch::Tensor> flash_jvp_pair(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor DQ, torch::Tensor DK, torch::Tensor DV, double scale) {
    using namespace flashjvp;
    TORCH_CHECK(Q.is_contiguous() && DQ.is_contiguous() && DQ.size(0) == 2);
    const int B = Q.size(0), H = Q.size(1), L = Q.size(2);
    TORCH_CHECK(Q.size(3) == HD && L % BM == 0);
    auto O = torch::empty_like(Q);
    auto DO = torch::empty_like(DQ);
    cudaFuncSetAttribute(flash_jvp_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    dim3 grid(L / BM, B * H);
    flash_jvp_kernel<<<grid, 128, SMEM, c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(K.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(V.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(DQ.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(DK.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(DV.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(O.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(DO.data_ptr()),
        DQ.stride(0), L, (float)scale);
    return {O, DO};
}

std::vector<torch::Tensor> flash_jvp_pair_opt(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor DQ, torch::Tensor DK, torch::Tensor DV, double scale) {
    using namespace flashjvp;
    TORCH_CHECK(Q.is_contiguous() && DQ.is_contiguous() && DQ.size(0) == 2);
    const int B = Q.size(0), H = Q.size(1), L = Q.size(2);
    TORCH_CHECK(Q.size(3) == HD && L % BM == 0);
    auto O = torch::empty_like(Q);
    auto DO = torch::empty_like(DQ);
    cudaFuncSetAttribute(flash_jvp_opt_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_OPT);
    dim3 grid(L / BM, B * H);
    flash_jvp_opt_kernel<<<grid, 128, SMEM_OPT, c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(K.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(V.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(DQ.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(DK.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(DV.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(O.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(DO.data_ptr()),
        DQ.stride(0), L, (float)scale);
    return {O, DO};
}

std::vector<torch::Tensor> flash_jvp_pair_wg2(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor DQ, torch::Tensor DK, torch::Tensor DV, double scale) {
    using namespace flashjvp;
    TORCH_CHECK(Q.is_contiguous() && DQ.is_contiguous() && DQ.size(0) == 2);
    const int B = Q.size(0), H = Q.size(1), L = Q.size(2);
    TORCH_CHECK(Q.size(3) == HD && L % BM2 == 0);
    auto O = torch::empty_like(Q);
    auto DO = torch::empty_like(DQ);
    cudaFuncSetAttribute(flash_jvp_wg2_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_WG2);
    dim3 grid(L / BM2, B * H);
    flash_jvp_wg2_kernel<<<grid, 256, SMEM_WG2, c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(K.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(V.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(DQ.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(DK.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(DV.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(O.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(DO.data_ptr()),
        DQ.stride(0), L, (float)scale);
    return {O, DO};
}

std::vector<torch::Tensor> flash_jvp_pair_occ2(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor DQ, torch::Tensor DK, torch::Tensor DV, double scale) {
    using namespace flashjvp;
    TORCH_CHECK(Q.is_contiguous() && DQ.is_contiguous() && DQ.size(0) == 2);
    const int B = Q.size(0), H = Q.size(1), L = Q.size(2);
    TORCH_CHECK(Q.size(3) == HD && L % BM == 0);
    auto O = torch::empty_like(Q);
    auto DO = torch::empty_like(DQ);
    cudaFuncSetAttribute(flash_jvp_occ2_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_OCC2);
    dim3 grid(L / BM, B * H);
    flash_jvp_occ2_kernel<<<grid, 128, SMEM_OCC2, c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(K.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(V.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(DQ.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(DK.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(DV.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(O.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(DO.data_ptr()),
        DQ.stride(0), L, (float)scale, 0);
    return {O, DO};
}

std::vector<torch::Tensor> flash_jvp_pair_pipe(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor DQ, torch::Tensor DK, torch::Tensor DV, double scale) {
    using namespace flashjvp;
    TORCH_CHECK(Q.is_contiguous() && DQ.is_contiguous() && DQ.size(0) == 2);
    const int B = Q.size(0), H = Q.size(1), L = Q.size(2);
    TORCH_CHECK(Q.size(3) == HD && L % BM == 0);
    auto O = torch::empty_like(Q);
    auto DO = torch::empty_like(DQ);
    cudaFuncSetAttribute(flash_jvp_pipe_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_PIPE);
    dim3 grid(L / BM, B * H);
    flash_jvp_pipe_kernel<<<grid, 128, SMEM_PIPE, c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(K.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(V.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(DQ.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(DK.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(DV.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(O.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(DO.data_ptr()),
        DQ.stride(0), L, (float)scale);
    return {O, DO};
}

// ---- Register-P variant (wgmma_rs): the occ2 kernel with p/pds consumed by
// the epilogue directly from registers. The S-accumulator C-fragment maps
// onto the RS A-fragment exactly (k-step ki = C regs [8ki..8ki+7] packed as
// consecutive bf16 pairs; rows (l>>2)+8*(e>>1), cols (l&3)*2+(e&1)+8*j8), so
// the conversion is fp32->bf16x2 packing with no shuffles. Removes the p/pds
// smem staging, two of the three per-tile CTA syncs, and 24 KB of smem. ----

namespace flashjvp {

constexpr size_t SMEM_RS = (size_t)9 * TILE * 2;

__global__ void __launch_bounds__(128, 2) flash_jvp_rs_kernel(
    const bf16* __restrict__ Q, const bf16* __restrict__ K,
    const bf16* __restrict__ V, const bf16* __restrict__ DQ,
    const bf16* __restrict__ DK, const bf16* __restrict__ DV,
    bf16* __restrict__ O, bf16* __restrict__ DO,
    long lane_str, int Lctx, float scale, int Hout) {
    extern __shared__ __align__(1024) char smem[];
    bf16* q_s   = reinterpret_cast<bf16*>(smem);
    bf16* dq0_s = q_s + TILE;
    bf16* dq1_s = dq0_s + TILE;
    bf16* kvbuf = dq1_s + TILE;               // [6][TILE], single-buffered

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
    float acc_o[32], do0[32], do1[32];
    float as0_0 = 0.f, as0_1 = 0.f, as1_0 = 0.f, as1_1 = 0.f;
#pragma unroll
    for (int i = 0; i < 32; ++i) { acc_o[i] = 0.f; do0[i] = 0.f; do1[i] = 0.f; }

    tl::GmmaDescriptor d_q, d_dq0, d_dq1;
    tl::initialize_wgmma_descriptor<1, 1, 64>(d_q, q_s);
    tl::initialize_wgmma_descriptor<1, 1, 64>(d_dq0, dq0_s);
    tl::initialize_wgmma_descriptor<1, 1, 64>(d_dq1, dq1_s);

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

        float S[32], dS0[32], dS1[32];
        tl::warpgroup_fence_operand(S, 32);
        tl::warpgroup_fence_operand(dS0, 32);
        tl::warpgroup_fence_operand(dS1, 32);
        tl::warpgroup_arrive();
        tl::fence_proxy_async();
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
            tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                         tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(
                uint64_t(d_q + ki * 2), uint64_t(d_k + ki * 2),
                reinterpret_cast<uint32_t*>(S), ki != 0);
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
            tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                         tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(
                uint64_t(d_dq0 + ki * 2), uint64_t(d_k + ki * 2),
                reinterpret_cast<uint32_t*>(dS0), ki != 0);
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
            tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                         tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(
                uint64_t(d_q + ki * 2), uint64_t(d_dk0 + ki * 2),
                reinterpret_cast<uint32_t*>(dS0), 1);
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
            tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                         tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(
                uint64_t(d_dq1 + ki * 2), uint64_t(d_k + ki * 2),
                reinterpret_cast<uint32_t*>(dS1), ki != 0);
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
            tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                         tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(
                uint64_t(d_q + ki * 2), uint64_t(d_dk1 + ki * 2),
                reinterpret_cast<uint32_t*>(dS1), 1);
        tl::warpgroup_commit_batch();
        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(S, 32);
        tl::warpgroup_fence_operand(dS0, 32);
        tl::warpgroup_fence_operand(dS1, 32);

        const int col0 = t * BN;
        const bool diag = (t == pid_m);
        float rmax0 = -INFINITY, rmax1 = -INFINITY;
        if (diag) {
#pragma unroll
            for (int reg = 0; reg < 32; ++reg) {
                int r, c; acc_rc(warp, lane, reg, r, c);
                const bool dead = (col0 + c) > (row0 + r);
                float sv = dead ? -INFINITY : S[reg] * scale;
                S[reg] = sv;
                dS0[reg] = dead ? 0.f : dS0[reg];
                dS1[reg] = dead ? 0.f : dS1[reg];
                if ((reg & 2) == 0) rmax0 = fmaxf(rmax0, sv); else rmax1 = fmaxf(rmax1, sv);
            }
        } else {
#pragma unroll
            for (int reg = 0; reg < 32; ++reg) {
                const float sv = S[reg] * scale;
                S[reg] = sv;
                if ((reg & 2) == 0) rmax0 = fmaxf(rmax0, sv); else rmax1 = fmaxf(rmax1, sv);
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

        // p/pds packed straight into RS A-fragments (bf16x2 per .b32); the
        // C-fragment pair (2m, 2m+1) is exactly the A pair for its k-step.
        uint32_t pA[16], pd0A[16], pd1A[16];
        float rs0 = 0.f, rs1 = 0.f, prs00 = 0.f, prs01 = 0.f, prs10 = 0.f, prs11 = 0.f;
#pragma unroll
        for (int m = 0; m < 16; ++m) {
            const int r0i = 2 * m, r1i = 2 * m + 1;
            const bool hi = (r0i & 2) != 0;   // e&2 selects the row half; it differs per reg in the pair
            float pv0, pv1, pd00, pd01, pd10, pd11;
            {
                const bool h0 = (r0i & 2) != 0;
                pv0 = __expf(S[r0i] - (h0 ? m1 : m0));
                pd00 = pv0 * dS0[r0i] * scale; pd10 = pv0 * dS1[r0i] * scale;
                if (h0) { rs1 += pv0; prs01 += pd00; prs11 += pd10; }
                else    { rs0 += pv0; prs00 += pd00; prs10 += pd10; }
            }
            {
                const bool h1 = (r1i & 2) != 0;
                pv1 = __expf(S[r1i] - (h1 ? m1 : m0));
                pd01 = pv1 * dS0[r1i] * scale; pd11 = pv1 * dS1[r1i] * scale;
                if (h1) { rs1 += pv1; prs01 += pd01; prs11 += pd11; }
                else    { rs0 += pv1; prs00 += pd01; prs10 += pd11; }
            }
            (void)hi;
            __nv_bfloat162 bp = __floats2bfloat162_rn(pv0, pv1);
            __nv_bfloat162 b0 = __floats2bfloat162_rn(pd00, pd01);
            __nv_bfloat162 b1 = __floats2bfloat162_rn(pd10, pd11);
            pA[m] = *reinterpret_cast<uint32_t*>(&bp);
            pd0A[m] = *reinterpret_cast<uint32_t*>(&b0);
            pd1A[m] = *reinterpret_cast<uint32_t*>(&b1);
        }
        if (rescale) {
#pragma unroll
            for (int reg = 0; reg < 32; ++reg) {
                const float a = (reg & 2) ? a1 : a0;
                acc_o[reg] *= a; do0[reg] *= a; do1[reg] *= a;
            }
        }

        tl::warpgroup_fence_operand(acc_o, 32);
        tl::warpgroup_fence_operand(do0, 32);
        tl::warpgroup_fence_operand(do1, 32);
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(pA), 16);
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(pd0A), 16);
        tl::warpgroup_fence_operand(reinterpret_cast<float*>(pd1A), 16);
        tl::warpgroup_arrive();
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
            tl::wgmma_rs<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                         tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(
                pA + 4 * ki, uint64_t(d_v + ki * 128),
                reinterpret_cast<uint32_t*>(acc_o), 1);
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
            tl::wgmma_rs<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                         tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(
                pd0A + 4 * ki, uint64_t(d_v + ki * 128),
                reinterpret_cast<uint32_t*>(do0), 1);
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
            tl::wgmma_rs<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                         tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(
                pA + 4 * ki, uint64_t(d_dv0 + ki * 128),
                reinterpret_cast<uint32_t*>(do0), 1);
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
            tl::wgmma_rs<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                         tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(
                pd1A + 4 * ki, uint64_t(d_v + ki * 128),
                reinterpret_cast<uint32_t*>(do1), 1);
#pragma unroll
        for (int ki = 0; ki < 4; ++ki)
            tl::wgmma_rs<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                         tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(
                pA + 4 * ki, uint64_t(d_dv1 + ki * 128),
                reinterpret_cast<uint32_t*>(do1), 1);
        tl::warpgroup_commit_batch();

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

        tl::warpgroup_wait<0>();
        tl::warpgroup_fence_operand(acc_o, 32);
        tl::warpgroup_fence_operand(do0, 32);
        tl::warpgroup_fence_operand(do1, 32);
        __syncthreads();
    }

    long obase = base, ostr = HD;
    if (Hout > 0) {
        const long b = bh / Hout, h = bh % Hout;
        obase = (b * Lctx * Hout + h) * HD;
        ostr = (long)Hout * HD;
    }
#pragma unroll
    for (int reg = 0; reg < 32; ++reg) {
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

}  // namespace flashjvp

// ---- pipe2: two consumer warpgroups per CTA sharing one KV stream.
// wg w owns q rows [pid_m*128 + w*64, +64); both warpgroups consume the same
// KV tiles from shared smem, so KV DRAM traffic halves and the SM runs 8
// warps at occupancy 1. Step A: single-buffered KV, CTA-wide sync at the
// buffer boundary, wg-local barriers around the p/pds staging. ----

namespace flashjvp {

__device__ inline void bar_wg(int wg) {
    asm volatile("bar.sync %0, %1;" :: "r"(1 + wg), "r"(128));
}

// smem: 2 x (q,dq0,dq1) + 2 x 6 KV + 2 x (p,pds0,pds1) = 24 tiles, plus the
// four mbarriers (full/empty per KV buffer) at the tail. Step B: per-buffer
// mbarrier producer/consumer flow; no CTA-wide barrier in the loop.
constexpr size_t SMEM_PIPE2 = (size_t)24 * TILE * 2 + 64;

__global__ void __launch_bounds__(256, 1) flash_jvp_pipe2_kernel(
    const bf16* __restrict__ Q, const bf16* __restrict__ K,
    const bf16* __restrict__ V, const bf16* __restrict__ DQ,
    const bf16* __restrict__ DK, const bf16* __restrict__ DV,
    bf16* __restrict__ O, bf16* __restrict__ DO,
    long lane_str, int Lctx, float scale) {
    extern __shared__ __align__(1024) char smem[];
    const int tid = threadIdx.x;
    const int wg = tid >> 7, wtid = tid & 127;
    const int warp = (wtid >> 5), lane = tid & 31;
    bf16* q_s    = reinterpret_cast<bf16*>(smem) + wg * 3 * TILE;
    bf16* dq0_s  = q_s + TILE;
    bf16* dq1_s  = dq0_s + TILE;
    bf16* kvbuf  = reinterpret_cast<bf16*>(smem) + 6 * TILE;   // [2][6][TILE]
    bf16* p_s    = kvbuf + 12 * TILE + wg * 3 * TILE;
    bf16* pds0_s = p_s + TILE;
    bf16* pds1_s = pds0_s + TILE;
    uint64_t* full  = reinterpret_cast<uint64_t*>(
        reinterpret_cast<bf16*>(smem) + 24 * TILE);
    uint64_t* empty = full + 2;

    const int pid_m = blockIdx.x;
    const long bh = blockIdx.y;
    const long base = bh * Lctx * HD;
    const int row0 = pid_m * 2 * BM + wg * BM;

    for (int idx = wtid; idx < BM * 8; idx += 128) {
        const int r = idx >> 3, u = idx & 7;
        const long g = base + (long)(row0 + r) * HD + u * 8;
        const int so = sw128(r, u * 8);
        *reinterpret_cast<uint4*>(q_s + so) = *reinterpret_cast<const uint4*>(Q + g);
        *reinterpret_cast<uint4*>(dq0_s + so) = *reinterpret_cast<const uint4*>(DQ + g);
        *reinterpret_cast<uint4*>(dq1_s + so) = *reinterpret_cast<const uint4*>(DQ + lane_str + g);
    }

    const int n_t = 2 * pid_m + wg + 1;   // this wg's causal tile count
    const int NT = 2 * pid_m + 2;         // shared KV stream length
    const bf16* gsrc[6] = {K, V, DK, DK, DV, DV};
    const long goff[6] = {0, 0, 0, lane_str, 0, lane_str};

    // A buffer is filled by ONE warpgroup (wg0 even tiles, wg1 odd), so the
    // empty-wait blocks only the refilling wg while the other runs ahead.
    auto prefetch = [&](int t) {
        bf16* buf = kvbuf + (t & 1) * 6 * TILE;
        const long col_base = base + (long)(t * BN) * HD;
        for (int idx = wtid; idx < BN * 8; idx += 128) {
            const int r = idx >> 3, u = idx & 7;
            const int so = sw128(r, u * 8);
            const long g = col_base + (long)r * HD + u * 8;
#pragma unroll
            for (int j = 0; j < 6; ++j)
                cp16(buf + j * TILE + so, gsrc[j] + goff[j] + g);
        }
        // .noinc: each thread's cp-completion consumes one of the 128
        // pre-declared arrivals (the plain form is net-zero and deadlocks).
        tl::mbarrier_cp_async_arrive_noinc(full[t & 1]);
    };

    if (tid == 0) {
        tl::mbarrier_init(full[0], 128);
        tl::mbarrier_init(full[1], 128);
        tl::mbarrier_init(empty[0], 256);
        tl::mbarrier_init(empty[1], 256);
        tl::fence_barrier_init();
    }
    __syncthreads();
    prefetch(wg);   // wg0 -> tile 0, wg1 -> tile 1, in parallel
    // (the warpgroups run free; a per-tile ordering baton would make each
    // wg inherit the slower wg's latency every tile)

    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
    float acc_o[32], do0[32], do1[32];
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

    int phf0 = 0, phf1 = 0, phe0 = 0, phe1 = 0;
    for (int t = 0; t < NT; ++t) {
        const int bsel = t & 1;
        bf16* buf = kvbuf + bsel * 6 * TILE;
        tl::GmmaDescriptor d_k, d_v, d_dk0, d_dk1, d_dv0, d_dv1;
        tl::initialize_wgmma_descriptor<1, 1, 64>(d_k, buf);
        tl::initialize_wgmma_descriptor<1, 1024, 64>(d_v, buf + TILE);
        tl::initialize_wgmma_descriptor<1, 1, 64>(d_dk0, buf + 2 * TILE);
        tl::initialize_wgmma_descriptor<1, 1, 64>(d_dk1, buf + 3 * TILE);
        tl::initialize_wgmma_descriptor<1, 1024, 64>(d_dv0, buf + 4 * TILE);
        tl::initialize_wgmma_descriptor<1, 1024, 64>(d_dv1, buf + 5 * TILE);

        if (bsel) { tl::mbarrier_wait(full[1], phf1); phf1 ^= 1; }
        else      { tl::mbarrier_wait(full[0], phf0); phf0 ^= 1; }

        const bool active = t < n_t;
        float S[32], dS0[32], dS1[32];
        if (active) {
            tl::warpgroup_fence_operand(S, 32);
            tl::warpgroup_fence_operand(dS0, 32);
            tl::warpgroup_fence_operand(dS1, 32);
            tl::warpgroup_arrive();
            tl::fence_proxy_async();
#pragma unroll
            for (int ki = 0; ki < 4; ++ki)
                tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                             tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(
                    uint64_t(d_q + ki * 2), uint64_t(d_k + ki * 2),
                    reinterpret_cast<uint32_t*>(S), ki != 0);
#pragma unroll
            for (int ki = 0; ki < 4; ++ki)
                tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                             tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(
                    uint64_t(d_dq0 + ki * 2), uint64_t(d_k + ki * 2),
                    reinterpret_cast<uint32_t*>(dS0), ki != 0);
#pragma unroll
            for (int ki = 0; ki < 4; ++ki)
                tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                             tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(
                    uint64_t(d_q + ki * 2), uint64_t(d_dk0 + ki * 2),
                    reinterpret_cast<uint32_t*>(dS0), 1);
#pragma unroll
            for (int ki = 0; ki < 4; ++ki)
                tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                             tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(
                    uint64_t(d_dq1 + ki * 2), uint64_t(d_k + ki * 2),
                    reinterpret_cast<uint32_t*>(dS1), ki != 0);
#pragma unroll
            for (int ki = 0; ki < 4; ++ki)
                tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                             tl::DataType::kFloat32, 64, 64, 16, false, false, 1, 1>(
                    uint64_t(d_q + ki * 2), uint64_t(d_dk1 + ki * 2),
                    reinterpret_cast<uint32_t*>(dS1), 1);
            tl::warpgroup_commit_batch();
            tl::warpgroup_wait<0>();
            tl::warpgroup_fence_operand(S, 32);
            tl::warpgroup_fence_operand(dS0, 32);
            tl::warpgroup_fence_operand(dS1, 32);

            const int col0 = t * BN;
            const bool diag = (t == n_t - 1);
            float rmax0 = -INFINITY, rmax1 = -INFINITY;
            if (diag) {
#pragma unroll
                for (int reg = 0; reg < 32; ++reg) {
                    int r, c; acc_rc(warp, lane, reg, r, c);
                    const bool dead = (col0 + c) > (row0 + r);
                    float sv = dead ? -INFINITY : S[reg] * scale;
                    S[reg] = sv;
                    dS0[reg] = dead ? 0.f : dS0[reg];
                    dS1[reg] = dead ? 0.f : dS1[reg];
                    if ((reg & 2) == 0) rmax0 = fmaxf(rmax0, sv); else rmax1 = fmaxf(rmax1, sv);
                }
            } else {
#pragma unroll
                for (int reg = 0; reg < 32; ++reg) {
                    const float sv = S[reg] * scale;
                    S[reg] = sv;
                    if ((reg & 2) == 0) rmax0 = fmaxf(rmax0, sv); else rmax1 = fmaxf(rmax1, sv);
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
            bar_wg(wg);   // prior epilogue reads of this wg's p/pds retired
#pragma unroll
            for (int reg = 0; reg < 32; ++reg) {
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
            if (rescale) {
#pragma unroll
                for (int reg = 0; reg < 32; ++reg) {
                    const float a = (reg & 2) ? a1 : a0;
                    acc_o[reg] *= a; do0[reg] *= a; do1[reg] *= a;
                }
            }
            bar_wg(wg);   // this wg's p/pds tiles fully staged

            tl::warpgroup_fence_operand(acc_o, 32);
            tl::warpgroup_fence_operand(do0, 32);
            tl::warpgroup_fence_operand(do1, 32);
            tl::warpgroup_arrive();
            tl::fence_proxy_async();
#pragma unroll
            for (int ki = 0; ki < 4; ++ki)
                tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                             tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(
                    uint64_t(d_p + ki * 2), uint64_t(d_v + ki * 128),
                    reinterpret_cast<uint32_t*>(acc_o), 1);
#pragma unroll
            for (int ki = 0; ki < 4; ++ki)
                tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                             tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(
                    uint64_t(d_pds0 + ki * 2), uint64_t(d_v + ki * 128),
                    reinterpret_cast<uint32_t*>(do0), 1);
#pragma unroll
            for (int ki = 0; ki < 4; ++ki)
                tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                             tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(
                    uint64_t(d_p + ki * 2), uint64_t(d_dv0 + ki * 128),
                    reinterpret_cast<uint32_t*>(do0), 1);
#pragma unroll
            for (int ki = 0; ki < 4; ++ki)
                tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                             tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(
                    uint64_t(d_pds1 + ki * 2), uint64_t(d_v + ki * 128),
                    reinterpret_cast<uint32_t*>(do1), 1);
#pragma unroll
            for (int ki = 0; ki < 4; ++ki)
                tl::wgmma_ss<tl::DataType::kBFloat16, tl::DataType::kBFloat16,
                             tl::DataType::kFloat32, 64, 64, 16, false, true, 1, 1>(
                    uint64_t(d_p + ki * 2), uint64_t(d_dv1 + ki * 128),
                    reinterpret_cast<uint32_t*>(do1), 1);
            tl::warpgroup_commit_batch();

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

            tl::warpgroup_wait<0>();
            tl::warpgroup_fence_operand(acc_o, 32);
            tl::warpgroup_fence_operand(do0, 32);
            tl::warpgroup_fence_operand(do1, 32);
        }
        // release the buffer; the owning wg refills it for tile t+2 once
        // both wgs have released it
        if (bsel) tl::mbarrier_arrive(empty[1]);
        else      tl::mbarrier_arrive(empty[0]);
        if (wg == bsel && t + 2 < NT) {
            if (bsel) { tl::mbarrier_wait(empty[1], phe1); phe1 ^= 1; }
            else      { tl::mbarrier_wait(empty[0], phe0); phe0 ^= 1; }
            prefetch(t + 2);
        }
    }

#pragma unroll
    for (int reg = 0; reg < 32; ++reg) {
        int r, c; acc_rc(warp, lane, reg, r, c);
        const bool hi = (reg & 2) != 0;
        const float li = hi ? l1 : l0;
        const float o = acc_o[reg] / li;
        const long g = base + (long)(row0 + r) * HD + c;
        O[g] = __float2bfloat16(o);
        DO[g] = __float2bfloat16((do0[reg] - (hi ? as0_1 : as0_0) * o) / li);
        DO[lane_str + g] = __float2bfloat16((do1[reg] - (hi ? as1_1 : as1_0) * o) / li);
    }
}

}  // namespace flashjvp

// Full-r entry: loops lane pairs inside one op so the compiled graph sees a
// single node with fresh outputs (the primal O is recomputed per pair by
// design -- identical bits, last write wins).
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

std::vector<torch::Tensor> flash_jvp_pair_pipe2(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor DQ, torch::Tensor DK, torch::Tensor DV, double scale) {
    using namespace flashjvp;
    TORCH_CHECK(Q.is_contiguous() && DQ.is_contiguous() && DQ.size(0) == 2);
    const int B = Q.size(0), H = Q.size(1), L = Q.size(2);
    TORCH_CHECK(Q.size(3) == HD && L % (2 * BM) == 0);
    auto O = torch::empty_like(Q);
    auto DO = torch::empty_like(DQ);
    cudaFuncSetAttribute(flash_jvp_pipe2_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_PIPE2);
    dim3 grid(L / (2 * BM), B * H);
    flash_jvp_pipe2_kernel<<<grid, 256, SMEM_PIPE2, c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(K.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(V.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(DQ.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(DK.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(DV.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(O.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(DO.data_ptr()),
        DQ.stride(0), L, (float)scale);
    return {O, DO};
}

std::vector<torch::Tensor> flash_jvp_pair_rs(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor DQ, torch::Tensor DK, torch::Tensor DV, double scale) {
    using namespace flashjvp;
    TORCH_CHECK(Q.is_contiguous() && DQ.is_contiguous() && DQ.size(0) == 2);
    const int B = Q.size(0), H = Q.size(1), L = Q.size(2);
    TORCH_CHECK(Q.size(3) == HD && L % BM == 0);
    auto O = torch::empty_like(Q);
    auto DO = torch::empty_like(DQ);
    cudaFuncSetAttribute(flash_jvp_rs_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_RS);
    dim3 grid(L / BM, B * H);
    flash_jvp_rs_kernel<<<grid, 128, SMEM_RS, c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(K.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(V.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(DQ.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(DK.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(DV.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(O.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(DO.data_ptr()),
        DQ.stride(0), L, (float)scale, 0);
    return {O, DO};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_jvp_pair", &flash_jvp_pair, "Fused flash-JVP, lane pair, hd64");
    m.def("flash_jvp_occ2_full", &flash_jvp_occ2_full, "Occ2 flash-JVP, full r");
    m.def("flash_jvp_pair_rs", &flash_jvp_pair_rs, "Register-P (wgmma_rs) flash-JVP pair");
    m.def("flash_jvp_pair_pipe2", &flash_jvp_pair_pipe2, "Two-warpgroup shared-KV flash-JVP pair");
    m.def("flash_jvp_pair_opt", &flash_jvp_pair_opt, "Optimized fused flash-JVP pair");
    m.def("flash_jvp_pair_wg2", &flash_jvp_pair_wg2, "Two-warpgroup fused flash-JVP pair");
    m.def("flash_jvp_pair_occ2", &flash_jvp_pair_occ2, "Occupancy-2 fused flash-JVP pair");
    m.def("flash_jvp_pair_pipe", &flash_jvp_pair_pipe, "Pipelined fused flash-JVP pair");
}
