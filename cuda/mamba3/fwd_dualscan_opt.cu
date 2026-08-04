// Forward dual-scan passes, wmma rewrite: same semantics as _simple.cu, heavy
// MACs on tensor cores, operands in skewed bf16 smem, fp32 outputs aliased.

// passC arena, three pools each reused once across its lifetime:
//   pool1 qr,dqr bf16 [cs][NS]x2 -> v,dv bf16 [cs][PS]x2 + dfin f32 [cs][P]

//   pool2 ksc,dksc bf16 [cs][NS]x2 -> out,dout f32 [cs][P]x2 (packed);
//   pool3 QK,dQK f32 [cs][cs]x2 -> WQKb,dWQKb bf16 [cs][CS2]x2 (in place)

// exp(L) folds into the readout operands (qre = e*qr, dqre = e*dqr + e*dL*qr)
// so readout gemms accumulate straight into out/dout. 256 threads, 2 CTAs/SM.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include "fwd_dualscan.hpp"
#include "wmma_gemm.cuh"

namespace lbi_mamba3 {

namespace {

using bf16 = __nv_bfloat16;
using wmma_gemm::gemm_AB;
using wmma_gemm::gemm_ABt;
using wmma_gemm::gemm_AtB;

constexpr int SKEW = 8;        // elements of row padding on wmma smem operands
constexpr int MASKC = 16;      // mask elements register-staged per thread/chunk
constexpr int VSTASH_MAX = 32; // v/dv elements prefetched into registers

__device__ inline float ld(const bf16* p) { return __bfloat162float(*p); }

// 16-byte async gmem->smem copy; bulk operands issue back-to-back and drain
// with one wait, so the gen phase pays a single memory latency.
__device__ __forceinline__ void cp16(void* dst, const void* src) {
    const unsigned sdst = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(sdst), "l"(src));
}
__device__ __forceinline__ void cp_wait_all() {
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::: "memory");
}

// Copy `rows` rows of `rowbytes` (multiple of 16) from strided gmem into
// strided smem, spread over the block's threads.
__device__ __forceinline__ void cp_rows(bf16* dst, long dst_stride,
                                        const bf16* src, long src_stride,
                                        int rows, int rowbytes,
                                        int tid, int nthr) {
    const int nch = rowbytes / 16;
    for (int u = tid; u < rows * nch; u += nthr) {
        const int j = u / nch, kk = u % nch;
        cp16(reinterpret_cast<char*>(dst) + j * dst_stride * 2 + kk * 16,
             reinterpret_cast<const char*>(src) + j * src_stride * 2 + kk * 16);
    }
}

// Native fold: copy rows of fp32 gmem (unpadded; tail zero-filled) into a
// skewed bf16 smem slot, converting in registers (RN, matching torch).
__device__ __forceinline__ void ld_rows_f32(bf16* dst, int dst_stride,
                                            const float* src, long src_stride,
                                            int rows, int rvalid, int nfl,
                                            int tid, int nthr) {
    const int nch = nfl / 4;
    for (int u = tid; u < rows * nch; u += nthr) {
        const int j = u / nch, kk = u % nch;
        float4 f = make_float4(0.f, 0.f, 0.f, 0.f);
        if (j < rvalid)
            f = *reinterpret_cast<const float4*>(src + (long)j * src_stride + kk * 4);
        bf16* d = dst + j * dst_stride + kk * 4;
        d[0] = __float2bfloat16(f.x); d[1] = __float2bfloat16(f.y);
        d[2] = __float2bfloat16(f.z); d[3] = __float2bfloat16(f.w);
    }
}

// pass A (tangent): A_w = wl*(dLl-dL_j)*ksc + wl*dksc, B_w = wl*ksc;
// DSC = A_w^T @ v + B_w^T @ dv on tensor cores, operands transformed in place.
namespace wmmaA = nvcuda::wmma;

__global__ void __launch_bounds__(256, 3) fwd_passA_tan_opt_kernel(FwdDualscanParams p) {
    const int R = p.lanes;
    const int c = blockIdx.x / R;
    const int l = blockIdx.x % R;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    const int cs = p.chunk_size, N = p.d_state, P = p.headdim, Da = p.n_rot;
    const int S = p.seqlen, H = p.heads, G = p.groups;
    const int hg = h / (H / G);
    const int nc = S / cs;
    const int s0 = c * cs;
    const int tid = threadIdx.x, nthr = blockDim.x;
    const int warp = tid / 32, nwarps = nthr / 32;
    const int NS = N + SKEW, PS = P + SKEW;

    extern __shared__ char arena[];
    bf16* A_w = reinterpret_cast<bf16*>(arena);            // [cs][NS]
    bf16* B_w = A_w + cs * NS;                             // [cs][NS]
    bf16* v_b = B_w + cs * NS;                             // [cs][PS]
    bf16* dv_b = v_b + cs * PS;                            // [cs][PS]
    float* wl_s = reinterpret_cast<float*>(dv_b + cs * PS);
    float* a_s = wl_s + cs;
    float* sc_s = a_s + cs;
    float* dsc_s = sc_s + cs;
    float* ws_all = dsc_s + cs;                            // [nwarps][256] f32

    const long row0 = ((long)b * H + h) * S + s0;
    const long lrow0 = (((long)l * p.batch + b) * H + h) * S + s0;
    cp_rows(A_w, NS, p.KR + (((long)b * S + s0) * H + h) * (long)N,
            (long)H * N, cs, N * 2, tid, nthr);
    cp_rows(B_w, NS, p.DKRAW + ((((long)l * p.batch + b) * S + s0) * G + hg) * (long)N,
            (long)G * N, cs, N * 2, tid, nthr);
    cp_rows(v_b, PS, p.V + (((long)b * S + s0) * H + h) * (long)P,
            (long)H * P, cs, P * 2, tid, nthr);
    cp_rows(dv_b, PS, p.DV + ((((long)l * p.batch + b) * S + s0) * H + h) * (long)P,
            (long)H * P, cs, P * 2, tid, nthr);
    {
        const float llast = p.L[row0 + cs - 1];
        const float dllast = p.DL[lrow0 + cs - 1];
        for (int i = tid; i < cs; i += nthr) {
            const float wl = expf(llast - p.L[row0 + i]);
            wl_s[i] = wl;
            a_s[i] = wl * (dllast - p.DL[lrow0 + i]);
            sc_s[i] = p.SCALE[row0 + i];
            dsc_s[i] = p.DSCALE[lrow0 + i];
        }
    }
    cp_wait_all();
    __syncthreads();
    for (int idx = tid; idx < cs * (N / 2); idx += nthr) {
        const int j = idx / (N / 2), pp = idx % (N / 2);
        const float k0 = ld(A_w + j * NS + 2 * pp);
        const float k1 = ld(A_w + j * NS + 2 * pp + 1);
        const float d0 = ld(B_w + j * NS + 2 * pp);
        const float d1 = ld(B_w + j * NS + 2 * pp + 1);
        float cw = 1.f, sw = 0.f, dt = 0.f;
        if (pp < Da) {
            const long arow = (row0 + j) * (long)Da + pp;
            cw = ld(p.COS + arow);
            sw = ld(p.SIN + arow);
            dt = ld(p.DTHETA + (lrow0 + j) * (long)Da + pp);
        }
        const float sc = sc_s[j], dsc = dsc_s[j];
        const float ksc0 = k0 * sc, ksc1 = k1 * sc;
        const float dksc0 = (d0 * cw - d1 * sw - dt * k1) * sc + k0 * dsc;
        const float dksc1 = (d0 * sw + d1 * cw + dt * k0) * sc + k1 * dsc;
        const float wl = wl_s[j], aj = a_s[j];
        A_w[j * NS + 2 * pp] = __float2bfloat16(aj * ksc0 + wl * dksc0);
        A_w[j * NS + 2 * pp + 1] = __float2bfloat16(aj * ksc1 + wl * dksc1);
        B_w[j * NS + 2 * pp] = __float2bfloat16(wl * ksc0);
        B_w[j * NS + 2 * pp + 1] = __float2bfloat16(wl * ksc1);
    }
    __syncthreads();
    bf16* DSCb = p.DSCB + ((((long)l * p.batch + b) * H + h) * nc + c) * (long)N * P;
    const int lane = tid % 32;
    float* ws = ws_all + warp * 256;
    const int mt = N / 16, nt = P / 16;
    if (mt * nt == 4 * nwarps && nwarps % nt == 0) {
        // 4 same-column tiles per warp, register-resident, shared v/dv frags
        const int in = (warp % nt) * 16;
        wmmaA::fragment<wmmaA::accumulator, 16, 16, 16, float> acc[4];
#pragma unroll
        for (int tt = 0; tt < 4; ++tt) wmmaA::fill_fragment(acc[tt], 0.f);
        for (int k = 0; k < cs / 16; ++k) {
            wmmaA::fragment<wmmaA::matrix_b, 16, 16, 16, bf16, wmmaA::row_major> vb, dvb;
            wmmaA::load_matrix_sync(vb, v_b + (k * 16) * PS + in, PS);
            wmmaA::load_matrix_sync(dvb, dv_b + (k * 16) * PS + in, PS);
#pragma unroll
            for (int tt = 0; tt < 4; ++tt) {
                const int im = ((warp + tt * nwarps) / nt) * 16;
                wmmaA::fragment<wmmaA::matrix_a, 16, 16, 16, bf16, wmmaA::col_major> aa, ab;
                wmmaA::load_matrix_sync(aa, A_w + (k * 16) * NS + im, NS);
                wmmaA::load_matrix_sync(ab, B_w + (k * 16) * NS + im, NS);
                wmmaA::mma_sync(acc[tt], aa, vb, acc[tt]);
                wmmaA::mma_sync(acc[tt], ab, dvb, acc[tt]);
            }
        }
#pragma unroll
        for (int tt = 0; tt < 4; ++tt) {
            const int im = ((warp + tt * nwarps) / nt) * 16;
            wmmaA::store_matrix_sync(ws, acc[tt], 16, wmmaA::mem_row_major);
            __syncwarp();
            for (int e = lane; e < 256; e += 32)
                DSCb[(im + e / 16) * (long)P + in + e % 16] =
                    __float2bfloat16(ws[e]);
            __syncwarp();
        }
    } else {
        // generic tiling: one tile at a time through the warp scratch
        for (int t = warp; t < mt * nt; t += nwarps) {
            const int im = (t / nt) * 16, in = (t % nt) * 16;
            wmmaA::fragment<wmmaA::accumulator, 16, 16, 16, float> acc;
            wmmaA::fill_fragment(acc, 0.f);
            for (int k = 0; k < cs / 16; ++k) {
                wmmaA::fragment<wmmaA::matrix_a, 16, 16, 16, bf16, wmmaA::col_major> aa, ab;
                wmmaA::fragment<wmmaA::matrix_b, 16, 16, 16, bf16, wmmaA::row_major> vb, dvb;
                wmmaA::load_matrix_sync(aa, A_w + (k * 16) * NS + im, NS);
                wmmaA::load_matrix_sync(ab, B_w + (k * 16) * NS + im, NS);
                wmmaA::load_matrix_sync(vb, v_b + (k * 16) * PS + in, PS);
                wmmaA::load_matrix_sync(dvb, dv_b + (k * 16) * PS + in, PS);
                wmmaA::mma_sync(acc, aa, vb, acc);
                wmmaA::mma_sync(acc, ab, dvb, acc);
            }
            wmmaA::store_matrix_sync(ws, acc, 16, wmmaA::mem_row_major);
            __syncwarp();
            for (int e = lane; e < 256; e += 32)
                DSCb[(im + e / 16) * (long)P + in + e % 16] =
                    __float2bfloat16(ws[e]);
            __syncwarp();
        }
    }
}

// pass A (primal): SC = (wl*sc*kr)^T @ v; KR transformed in place, 4-tile
// register-resident AtB with shared v fragments, bf16 SC out.
__global__ void __launch_bounds__(256, 4) fwd_passA_primal_opt_kernel(FwdDualscanParams p) {
    const int c = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    const int cs = p.chunk_size, N = p.d_state, P = p.headdim;
    const int S = p.seqlen, H = p.heads;
    const int nc = S / cs;
    const int s0 = c * cs;
    const int tid = threadIdx.x, nthr = blockDim.x;
    const int warp = tid / 32, nwarps = nthr / 32, lane = tid % 32;
    const int NS = N + SKEW, PS = P + SKEW;

    extern __shared__ char arena[];
    bf16* B_w = reinterpret_cast<bf16*>(arena);            // [cs][NS]
    bf16* v_b = B_w + cs * NS;                             // [cs][PS]
    float* w_s = reinterpret_cast<float*>(v_b + cs * PS);  // [cs] wl*sc
    float* ws_all = w_s + cs;                              // [nwarps][256]

    const long row0 = ((long)b * H + h) * S + s0;
    cp_rows(B_w, NS, p.KR + (((long)b * S + s0) * H + h) * (long)N,
            (long)H * N, cs, N * 2, tid, nthr);
    cp_rows(v_b, PS, p.V + (((long)b * S + s0) * H + h) * (long)P,
            (long)H * P, cs, P * 2, tid, nthr);
    {
        const float llast = p.L[row0 + cs - 1];
        for (int i = tid; i < cs; i += nthr)
            w_s[i] = expf(llast - p.L[row0 + i]) * p.SCALE[row0 + i];
    }
    cp_wait_all();
    __syncthreads();
    for (int idx = tid; idx < cs * N; idx += nthr) {
        const int j = idx / N, n = idx % N;
        B_w[j * NS + n] = __float2bfloat16(ld(B_w + j * NS + n) * w_s[j]);
    }
    __syncthreads();
    bf16* SCb = p.SCB + (((long)b * H + h) * nc + c) * (long)N * P;
    float* ws = ws_all + warp * 256;
    const int mt = N / 16, nt = P / 16;
    const bool four = (mt * nt == 4 * nwarps && nwarps % nt == 0);
    const int per = four ? 4 : 1;
    for (int t0 = warp; t0 < mt * nt; t0 += nwarps * per) {
        if (four) {
            const int in = (warp % nt) * 16;
            wmmaA::fragment<wmmaA::accumulator, 16, 16, 16, float> acc[4];
#pragma unroll
            for (int tt = 0; tt < 4; ++tt) wmmaA::fill_fragment(acc[tt], 0.f);
            for (int k = 0; k < cs / 16; ++k) {
                wmmaA::fragment<wmmaA::matrix_b, 16, 16, 16, bf16, wmmaA::row_major> vb;
                wmmaA::load_matrix_sync(vb, v_b + (k * 16) * PS + in, PS);
#pragma unroll
                for (int tt = 0; tt < 4; ++tt) {
                    const int im = ((warp + tt * nwarps) / nt) * 16;
                    wmmaA::fragment<wmmaA::matrix_a, 16, 16, 16, bf16, wmmaA::col_major> ab;
                    wmmaA::load_matrix_sync(ab, B_w + (k * 16) * NS + im, NS);
                    wmmaA::mma_sync(acc[tt], ab, vb, acc[tt]);
                }
            }
#pragma unroll
            for (int tt = 0; tt < 4; ++tt) {
                const int im = ((warp + tt * nwarps) / nt) * 16;
                wmmaA::store_matrix_sync(ws, acc[tt], 16, wmmaA::mem_row_major);
                __syncwarp();
                for (int e = lane; e < 256; e += 32)
                    SCb[(im + e / 16) * (long)P + in + e % 16] =
                        __float2bfloat16(ws[e]);
                __syncwarp();
            }
        } else {
            const int im = (t0 / nt) * 16, in = (t0 % nt) * 16;
            wmmaA::fragment<wmmaA::accumulator, 16, 16, 16, float> acc;
            wmmaA::fill_fragment(acc, 0.f);
            for (int k = 0; k < cs / 16; ++k) {
                wmmaA::fragment<wmmaA::matrix_a, 16, 16, 16, bf16, wmmaA::col_major> ab;
                wmmaA::fragment<wmmaA::matrix_b, 16, 16, 16, bf16, wmmaA::row_major> vb;
                wmmaA::load_matrix_sync(ab, B_w + (k * 16) * NS + im, NS);
                wmmaA::load_matrix_sync(vb, v_b + (k * 16) * PS + in, PS);
                wmmaA::mma_sync(acc, ab, vb, acc);
            }
            wmmaA::store_matrix_sync(ws, acc, 16, wmmaA::mem_row_major);
            __syncwarp();
            for (int e = lane; e < 256; e += 32)
                SCb[(im + e / 16) * (long)P + in + e % 16] =
                    __float2bfloat16(ws[e]);
            __syncwarp();
        }
        if (four) break;  // the 4-tile path covers all tiles in one pass
    }
}

// pass B: inter-chunk tri scan, S_IN[c] = sum_{j<c} tri[c,j]*SC[j] and
// DS_IN[l][c] = sum_{j<c} tri[c,j]*(DSC[l][j] + dcd[l][j]*S_IN[j]).

// tri[c,j] = exp(LE[c-1]-LE[j]) strict lower; one CTA per (column, h, b),
// setup amortized over lanes via a double-buffered cp.async pipeline.
constexpr int COLB = 64;

__global__ void fwd_passB_opt_kernel(FwdDualscanParams p) {
    const int cb = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    const int cs = p.chunk_size, R = p.lanes;
    const int S = p.seqlen, H = p.heads;
    const int nc = S / cs;
    const long NP = (long)p.d_state * p.headdim;
    const long col0 = (long)cb * COLB;
    const int tid = threadIdx.x, nthr = blockDim.x;

    extern __shared__ char arena[];
    float* tri_s = reinterpret_cast<float*>(arena);        // [nc][nc]
    float* LE_s = tri_s + nc * nc;                         // [nc]
    float* dcd_s = LE_s + nc;                              // [R][nc]
    float* SIs = dcd_s + R * nc;                           // [nc][COLB] f32
    bf16* SCs = reinterpret_cast<bf16*>(SIs + nc * COLB);  // [nc][COLB]
    bf16* DBUF = SCs + nc * COLB;                          // 2 x [nc][COLB]

    const long lrow = ((long)b * H + h) * S;
    for (int j = tid; j < nc; j += nthr)
        LE_s[j] = p.L[lrow + (long)j * cs + cs - 1];       // llast (pre-cumsum)
    for (int u = tid; u < R * nc; u += nthr) {
        const int l = u / nc, j = u % nc;
        dcd_s[u] = p.DL[(((long)l * p.batch + b) * H + h) * S + (long)j * cs + cs - 1];
    }
    const bf16* SCg = p.SCB + (((long)b * H + h) * nc) * NP + col0;
    cp_rows(SCs, COLB, SCg, NP, nc, COLB * 2, tid, nthr);
    const bf16* D0 = p.DSCB + (((long)b * H + h) * nc) * NP + col0;
    cp_rows(DBUF, COLB, D0, NP, nc, COLB * 2, tid, nthr);
    cp_wait_all();
    __syncthreads();
    if (tid == 0) {  // serial chunk cumsum (nc is small)
        float run = 0.f;
        for (int j = 0; j < nc; ++j) { run += LE_s[j]; LE_s[j] = run; }
    }
    __syncthreads();
    for (int u = tid; u < nc * nc; u += nthr) {
        const int cc = u / nc, j = u % nc;
        const float lec = (cc == 0) ? 0.f : LE_s[cc - 1];
        tri_s[u] = (j < cc) ? expf(lec - LE_s[j]) : 0.f;
    }
    for (int u = tid; u < R * nc; u += nthr) {
        const int j = u % nc;
        const float lecj = (j == 0) ? 0.f : LE_s[j - 1];
        dcd_s[u] *= expf(LE_s[j] - lecj);                  // * exp(llast[j])
    }
    __syncthreads();

    // primal: S_IN = tri @ SC (f32 in smem for the lanes; bf16 out)
    bf16* SOg = p.SB_OUT + (((long)b * H + h) * nc) * NP + col0;
    for (int u = tid; u < nc * COLB; u += nthr) {
        const int cc = u / COLB, col = u % COLB;
        float acc = 0.f;
        for (int j = 0; j < cc; ++j)
            acc += tri_s[cc * nc + j] * ld(SCs + j * COLB + col);
        SIs[u] = acc;
        SOg[(long)cc * NP + col] = __float2bfloat16(acc);
    }
    __syncthreads();

    for (int l = 0; l < R; ++l) {
        if (l + 1 < R) {
            const bf16* Dn = p.DSCB
                + ((((long)(l + 1) * p.batch + b) * H + h) * nc) * NP + col0;
            cp_rows(DBUF + ((l + 1) & 1) * nc * COLB, COLB, Dn, NP,
                    nc, COLB * 2, tid, nthr);
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        asm volatile("cp.async.wait_group 1;\n" ::: "memory");
        __syncthreads();  // buffer l resident (groups drain in order)
        const bf16* Db = DBUF + (l & 1) * nc * COLB;
        bf16* DOg = p.DSB_OUT + ((((long)l * p.batch + b) * H + h) * nc) * NP + col0;
        for (int u = tid; u < nc * COLB; u += nthr) {
            const int cc = u / COLB, col = u % COLB;
            float acc = 0.f;
            for (int j = 0; j < cc; ++j)
                acc += tri_s[cc * nc + j]
                     * (ld(Db + j * COLB + col) + dcd_s[l * nc + j] * SIs[j * COLB + col]);
            DOg[(long)cc * NP + col] = __float2bfloat16(acc);
        }
        __syncthreads();  // done reading buf[l&1] before iter l+1 refills it
    }
}

// pass C
struct PassCPtrs {
    bf16 *qr, *dqr, *ksc, *dksc;               // pools 1,2 (gen-phase names)
    float *QK, *dQK;                           // pool 3
    float *L_s, *dL_s, *sc_s, *dsc_s, *qk_s, *dqk_s, *part_s, *e_s;
    // reuses (each valid only after its pool's first tenant is dead):
    bf16 *WQKb, *dWQKb;                        // over QK/dQK   (post-mask)
    bf16 *v, *dv;                              // over ksc/dksc (post-QK gemms)
    float *ws;                                 // pool-2 tail: per-warp tile
                                               // scratch (mean sink)
};

__device__ PassCPtrs carve_arena(char* arena, int cs, int N, int P) {
    const int NS = N + SKEW, PS = P + SKEW;
    PassCPtrs w;
    char* p0 = arena;
    w.qr = reinterpret_cast<bf16*>(p0); p0 += cs * NS * 2;
    w.dqr = reinterpret_cast<bf16*>(p0); p0 += cs * NS * 2;
    w.ksc = reinterpret_cast<bf16*>(p0); p0 += cs * NS * 2;
    w.dksc = reinterpret_cast<bf16*>(p0); p0 += cs * NS * 2;
    w.QK = reinterpret_cast<float*>(p0); p0 += cs * cs * 4;
    w.dQK = reinterpret_cast<float*>(p0); p0 += cs * cs * 4;
    w.L_s = reinterpret_cast<float*>(p0); p0 += cs * 4;
    w.dL_s = reinterpret_cast<float*>(p0); p0 += cs * 4;
    w.sc_s = reinterpret_cast<float*>(p0); p0 += cs * 4;
    w.dsc_s = reinterpret_cast<float*>(p0); p0 += cs * 4;
    w.qk_s = reinterpret_cast<float*>(p0); p0 += cs * 4;
    w.dqk_s = reinterpret_cast<float*>(p0); p0 += cs * 4;
    w.part_s = reinterpret_cast<float*>(p0); p0 += cs * 4;
    w.e_s = reinterpret_cast<float*>(p0); p0 += cs * 4;
    // reuses (bf16 packed rows trail the f32 rows they replace, so the
    // ascending-chunk in-place mask conversion never clobbers unread f32).
    w.WQKb = reinterpret_cast<bf16*>(w.QK);
    w.dWQKb = reinterpret_cast<bf16*>(w.dQK);
    w.v = w.ksc;
    w.dv = w.ksc + cs * PS;
    w.ws = reinterpret_cast<float*>(w.ksc + 2 * (long)cs * PS);
    return w;
}

// Phases (4 barriers): 1 gen staging; 2 QK pair [ksc dies]; 3 rescale +
// v/dv stash + decay-mask pack [QK dies]. Dependent gemms need no barrier.

// Output contractions run outside the phases as one register-resident tile
// loop (inter readout + intra masked product accumulated before one store).
template <int ABLATE>  // 0 = full; k>0 = stop after phase k (bisection only)
__device__ __forceinline__ void passC_phases(
        const FwdDualscanParams& p, const PassCPtrs& w,
        int b, int h, int l, int c) {
    const int cs = p.chunk_size, N = p.d_state, P = p.headdim, Da = p.n_rot;
    const int S = p.seqlen, H = p.heads, G = p.groups;
    const int hg = h / (H / G);
    const int nc = S / cs;
    const int s0 = c * cs;
    const int tid = threadIdx.x, nthr = blockDim.x;
    const int warp = tid / 32, nwarps = nthr / 32;
    const int NS = N + SKEW, PS = P + SKEW, CS2 = cs + SKEW;

    // phase 1: bulk cp.async staging straight into final skewed slots +
    // in-place elementwise generation; one wait drains everything.
    const long row0 = ((long)b * H + h) * S + s0;
    const long lrow0 = (((long)l * p.batch + b) * H + h) * S + s0;
    const long qrow0 = (((long)b * S + s0) * H + h) * (long)N;
    const long drow0 = ((((long)l * p.batch + b) * S + s0) * G + hg) * (long)N;
    const long qstride = (long)H * N, dstride = (long)G * N;
    cp_rows(w.qr, NS, p.QR + qrow0, qstride, cs, N * 2, tid, nthr);
    cp_rows(w.dqr, NS, p.DQRAW + drow0, dstride, cs, N * 2, tid, nthr);
    cp_rows(w.ksc, NS, p.KR + qrow0, qstride, cs, N * 2, tid, nthr);
    cp_rows(w.dksc, NS, p.DKRAW + drow0, dstride, cs, N * 2, tid, nthr);
    bf16* cos_s = reinterpret_cast<bf16*>(w.QK);
    bf16* sin_s = cos_s + cs * Da;
    bf16* dth_s = sin_s + cs * Da;
    const bool stage_ang =
        (Da % 8 == 0) && (3 * cs * Da * 2 <= 2 * cs * cs * 4);
    if (stage_ang) {
        cp_rows(cos_s, Da, p.COS + row0 * Da, Da, cs, Da * 2, tid, nthr);
        cp_rows(sin_s, Da, p.SIN + row0 * Da, Da, cs, Da * 2, tid, nthr);
        cp_rows(dth_s, Da, p.DTHETA + lrow0 * Da, Da, cs, Da * 2, tid, nthr);
    }
    for (int i = tid; i < cs; i += nthr) {
        const float Lv = p.L[row0 + i];
        w.L_s[i] = Lv;
        w.e_s[i] = expf(Lv);
        w.dL_s[i] = p.DL[lrow0 + i];
        w.sc_s[i] = p.SCALE[row0 + i];
        w.dsc_s[i] = p.DSCALE[lrow0 + i];
        if (p.QKDOT) w.qk_s[i] = p.QKDOT[row0 + i];
        if (p.DQKDOT) w.dqk_s[i] = p.DQKDOT[lrow0 + i];
        if (i < P) w.part_s[i] = 0.f;
    }
    cp_wait_all();
    __syncthreads();
    for (int idx = tid; idx < cs * (N / 2); idx += nthr) {
        const int j = idx / (N / 2), pp = idx % (N / 2);
        float cw = 1.f, sw = 0.f, dt = 0.f;
        if (pp < Da) {
            if (stage_ang) {
                cw = ld(cos_s + j * Da + pp);
                sw = ld(sin_s + j * Da + pp);
                dt = ld(dth_s + j * Da + pp);
            } else {
                cw = ld(p.COS + (row0 + j) * (long)Da + pp);
                sw = ld(p.SIN + (row0 + j) * (long)Da + pp);
                dt = ld(p.DTHETA + (lrow0 + j) * (long)Da + pp);
            }
        }
        const float q0 = ld(w.qr + j * NS + 2 * pp);
        const float q1 = ld(w.qr + j * NS + 2 * pp + 1);
        const float e0 = ld(w.dqr + j * NS + 2 * pp);
        const float e1 = ld(w.dqr + j * NS + 2 * pp + 1);
        w.dqr[j * NS + 2 * pp] = __float2bfloat16(e0 * cw - e1 * sw - dt * q1);
        w.dqr[j * NS + 2 * pp + 1] = __float2bfloat16(e0 * sw + e1 * cw + dt * q0);
        const float k0 = ld(w.ksc + j * NS + 2 * pp);
        const float k1 = ld(w.ksc + j * NS + 2 * pp + 1);
        const float d0 = ld(w.dksc + j * NS + 2 * pp);
        const float d1 = ld(w.dksc + j * NS + 2 * pp + 1);
        const float sc = w.sc_s[j], dsc = w.dsc_s[j];
        w.ksc[j * NS + 2 * pp] = __float2bfloat16(k0 * sc);
        w.ksc[j * NS + 2 * pp + 1] = __float2bfloat16(k1 * sc);
        w.dksc[j * NS + 2 * pp] =
            __float2bfloat16((d0 * cw - d1 * sw - dt * k1) * sc + k0 * dsc);
        w.dksc[j * NS + 2 * pp + 1] =
            __float2bfloat16((d0 * sw + d1 * cw + dt * k0) * sc + k1 * dsc);
    }
    __syncthreads();
    if (ABLATE == 1) return;

    // ---- phase 2: intra QK / dQK, fused product-rule pair ----
    wmma_gemm::gemm_jvp_ABt(w.QK, w.dQK, cs, w.qr, w.dqr, NS, w.ksc, w.dksc,
                            NS, cs, cs, N, warp, nwarps, false);
    __syncthreads();
    if (ABLATE == 2) return;

    // phase 3: v/dv cp.async into the dead ksc pool; exp(L) folded into the
    // readout operands in place; decay-mask pack QK/dQK -> WQKb/dWQKb.
    cp_rows(w.v, PS, p.V + (((long)b * S + s0) * H + h) * (long)P,
            (long)H * P, cs, P * 2, tid, nthr);
    cp_rows(w.dv, PS, p.DV + ((((long)l * p.batch + b) * S + s0) * H + h) * (long)P,
            (long)H * P, cs, P * 2, tid, nthr);
    {
        const char* Sg = reinterpret_cast<const char*>(
            p.S_INB + (((long)b * H + h) * nc + c) * (long)N * P);
        const char* dSg = reinterpret_cast<const char*>(
            p.DS_INB + ((((long)l * p.batch + b) * H + h) * nc + c) * (long)N * P);
        for (int u = tid; u < (N * P * 2) / 128; u += nthr) {
            asm volatile("prefetch.global.L2 [%0];" :: "l"(Sg + u * 128));
            asm volatile("prefetch.global.L2 [%0];" :: "l"(dSg + u * 128));
        }
        if (p.Z) {  // mean sink reads Z per element; warm the lines
            const char* Zg = reinterpret_cast<const char*>(
                p.Z + (((long)b * S + s0) * H + h) * (long)P);
            for (int i = tid; i < cs; i += nthr)
                asm volatile("prefetch.global.L2 [%0];"
                             :: "l"(Zg + (long)i * H * P * 2));
        }
    }
    for (int idx = tid; idx < cs * N; idx += nthr) {
        const int i = idx / N, n = idx % N;
        const float e = w.e_s[i];
        const float qv = ld(w.qr + i * NS + n);
        const float dqv = ld(w.dqr + i * NS + n);
        w.dqr[i * NS + n] = __float2bfloat16(e * dqv + e * w.dL_s[i] * qv);
        w.qr[i * NS + n] = __float2bfloat16(e * qv);
    }
    for (int base = 0; base < cs * cs; base += nthr * MASKC) {
        float wq[MASKC], dwq[MASKC];
#pragma unroll
        for (int k = 0; k < MASKC; ++k) {
            const int idx = base + tid + k * nthr;
            wq[k] = 0.f; dwq[k] = 0.f;
            if (idx < cs * cs) {
                const int i = idx / cs, j = idx % cs;
                if (i >= j) {
                    const float wgt = expf(w.L_s[i] - w.L_s[j]);
                    wq[k] = wgt * w.QK[i * cs + j];
                    dwq[k] = wgt * (w.dL_s[i] - w.dL_s[j]) * w.QK[i * cs + j]
                           + wgt * w.dQK[i * cs + j];
                }
            }
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < MASKC; ++k) {
            const int idx = base + tid + k * nthr;
            if (idx < cs * cs) {
                const int i = idx / cs, j = idx % cs;
                w.WQKb[i * CS2 + j] = __float2bfloat16(wq[k]);
                w.dWQKb[i * CS2 + j] = __float2bfloat16(dwq[k]);
            }
        }
        __syncthreads();
    }
    cp_wait_all();  // v/dv issued at phase-3 start; drain before the output loop
    __syncthreads();
}

// One output tile of the fused inter+intra contraction: acc/dacc = qre/dqre
// @ Sin then += WQKb/dWQKb @ v, product-rule pairs sharing fragments.
namespace wmma = nvcuda::wmma;
using AccFrag = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

__device__ __forceinline__ void fused_out_tile(
        const PassCPtrs& w, const bf16* Sin_g, const bf16* dSin_g,
        int im, int in, int cs, int N, int P,
        AccFrag& acc, AccFrag& dacc) {
    const int NS = N + SKEW, PS = P + SKEW, CS2 = cs + SKEW;
    wmma::fill_fragment(acc, 0.f);
    wmma::fill_fragment(dacc, 0.f);
    for (int k = 0; k < N / 16; ++k) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a, da;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> sb, dsb;
        wmma::load_matrix_sync(a, w.qr + im * NS + k * 16, NS);
        wmma::load_matrix_sync(da, w.dqr + im * NS + k * 16, NS);
        wmma::load_matrix_sync(sb, Sin_g + (k * 16) * P + in, P);
        wmma::load_matrix_sync(dsb, dSin_g + (k * 16) * P + in, P);
        wmma::mma_sync(acc, a, sb, acc);
        wmma::mma_sync(dacc, da, sb, dacc);
        wmma::mma_sync(dacc, a, dsb, dacc);
    }
    for (int k = 0; k < cs / 16; ++k) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a, da;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> vb, dvb;
        wmma::load_matrix_sync(a, w.WQKb + im * CS2 + k * 16, CS2);
        wmma::load_matrix_sync(da, w.dWQKb + im * CS2 + k * 16, CS2);
        wmma::load_matrix_sync(vb, w.v + (k * 16) * PS + in, PS);
        wmma::load_matrix_sync(dvb, w.dv + (k * 16) * PS + in, PS);
        wmma::mma_sync(acc, a, vb, acc);
        wmma::mma_sync(dacc, da, vb, dacc);
        wmma::mma_sync(dacc, a, dvb, dacc);
    }
}

// Two row-tiles sharing one column block: B fragments load once and feed
// both accumulators -- half the fragment traffic, double the mma ILP.
__device__ __forceinline__ void fused_out_tile2(
        const PassCPtrs& w, const bf16* Sin_g, const bf16* dSin_g,
        int im0, int im1, int in, int cs, int N, int P,
        AccFrag& acc0, AccFrag& dacc0, AccFrag& acc1, AccFrag& dacc1) {
    const int NS = N + SKEW, PS = P + SKEW, CS2 = cs + SKEW;
    wmma::fill_fragment(acc0, 0.f);
    wmma::fill_fragment(dacc0, 0.f);
    wmma::fill_fragment(acc1, 0.f);
    wmma::fill_fragment(dacc1, 0.f);
    for (int k = 0; k < N / 16; ++k) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a0, da0, a1, da1;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> sb, dsb;
        wmma::load_matrix_sync(sb, Sin_g + (k * 16) * P + in, P);
        wmma::load_matrix_sync(dsb, dSin_g + (k * 16) * P + in, P);
        wmma::load_matrix_sync(a0, w.qr + im0 * NS + k * 16, NS);
        wmma::load_matrix_sync(da0, w.dqr + im0 * NS + k * 16, NS);
        wmma::load_matrix_sync(a1, w.qr + im1 * NS + k * 16, NS);
        wmma::load_matrix_sync(da1, w.dqr + im1 * NS + k * 16, NS);
        wmma::mma_sync(acc0, a0, sb, acc0);
        wmma::mma_sync(acc1, a1, sb, acc1);
        wmma::mma_sync(dacc0, da0, sb, dacc0);
        wmma::mma_sync(dacc1, da1, sb, dacc1);
        wmma::mma_sync(dacc0, a0, dsb, dacc0);
        wmma::mma_sync(dacc1, a1, dsb, dacc1);
    }
    for (int k = 0; k < cs / 16; ++k) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a0, da0, a1, da1;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> vb, dvb;
        wmma::load_matrix_sync(vb, w.v + (k * 16) * PS + in, PS);
        wmma::load_matrix_sync(dvb, w.dv + (k * 16) * PS + in, PS);
        wmma::load_matrix_sync(a0, w.WQKb + im0 * CS2 + k * 16, CS2);
        wmma::load_matrix_sync(da0, w.dWQKb + im0 * CS2 + k * 16, CS2);
        wmma::load_matrix_sync(a1, w.WQKb + im1 * CS2 + k * 16, CS2);
        wmma::load_matrix_sync(da1, w.dWQKb + im1 * CS2 + k * 16, CS2);
        wmma::mma_sync(acc0, a0, vb, acc0);
        wmma::mma_sync(acc1, a1, vb, acc1);
        wmma::mma_sync(dacc0, da0, vb, dacc0);
        wmma::mma_sync(dacc1, da1, vb, dacc1);
        wmma::mma_sync(dacc0, a0, dvb, dacc0);
        wmma::mma_sync(dacc1, a1, dvb, dacc1);
    }
}

template <int ABLATE>
__global__ void __launch_bounds__(256, 2) fwd_passC_opt_kernel(FwdDualscanParams p) {
    const int R = p.lanes;
    const int c = blockIdx.x / R;
    const int l = blockIdx.x % R;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    const int cs = p.chunk_size, N = p.d_state, P = p.headdim;
    const int S = p.seqlen, H = p.heads;
    const int nc = S / cs;
    const int s0 = c * cs;
    const int tid = threadIdx.x, nthr = blockDim.x;
    const int warp = tid / 32, nwarps = nthr / 32;

    extern __shared__ char arena[];
    PassCPtrs w = carve_arena(arena, cs, N, P);
    passC_phases<ABLATE>(p, w, b, h, l, c);
    if (ABLATE != 0) return;

    // fused output contraction, register accumulators, stores straight to
    // strided gmem (ld = H*P) -- no smem round trip, no epilogue.
    const bf16* Sin_g = p.S_INB + (((long)b * H + h) * nc + c) * (long)N * P;
    const bf16* dSin_g = p.DS_INB + ((((long)l * p.batch + b) * H + h) * nc + c) * (long)N * P;
    float* OUTb = p.OUT + (((long)b * S + s0) * H + h) * P;
    float* DOUTb = p.DOUT + ((((long)l * p.batch + b) * S + s0) * H + h) * P;
    const int ldg = H * P;
    const int nt = P / 16, mt = cs / 16;
    if (mt * nt == 2 * nwarps && nwarps % nt == 0) {
        const int in = (warp % nt) * 16;
        const int im0 = (warp / nt) * 16, im1 = ((warp + nwarps) / nt) * 16;
        AccFrag acc0, dacc0, acc1, dacc1;
        fused_out_tile2(w, Sin_g, dSin_g, im0, im1, in, cs, N, P,
                        acc0, dacc0, acc1, dacc1);
        wmma::store_matrix_sync(DOUTb + im0 * ldg + in, dacc0, ldg, wmma::mem_row_major);
        wmma::store_matrix_sync(DOUTb + im1 * ldg + in, dacc1, ldg, wmma::mem_row_major);
        if (l == 0) {
            wmma::store_matrix_sync(OUTb + im0 * ldg + in, acc0, ldg, wmma::mem_row_major);
            wmma::store_matrix_sync(OUTb + im1 * ldg + in, acc1, ldg, wmma::mem_row_major);
        }
    } else {
        for (int t = warp; t < mt * nt; t += nwarps) {
            const int im = (t / nt) * 16, in = (t % nt) * 16;
            AccFrag acc, dacc;
            fused_out_tile(w, Sin_g, dSin_g, im, in, cs, N, P, acc, dacc);
            wmma::store_matrix_sync(DOUTb + im * ldg + in, dacc, ldg, wmma::mem_row_major);
            if (l == 0)
                wmma::store_matrix_sync(OUTb + im * ldg + in, acc, ldg, wmma::mem_row_major);
        }
    }
}

__global__ void __launch_bounds__(256, 2) fwd_passC_mean_opt_kernel(FwdDualscanParams p) {
    const int R = p.lanes;
    const int c = blockIdx.x / R;
    const int l = blockIdx.x % R;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    const int cs = p.chunk_size, N = p.d_state, P = p.headdim;
    const int S = p.seqlen, H = p.heads;
    const int nc = S / cs;
    const int s0 = c * cs;
    const int tid = threadIdx.x, nthr = blockDim.x;
    const int warp = tid / 32, nwarps = nthr / 32, lane = tid % 32;
    const int PS = P + SKEW;

    extern __shared__ char arena[];
    PassCPtrs w = carve_arena(arena, cs, N, P);
    passC_phases<0>(p, w, b, h, l, c);

    // DZ stages via cp.async into pool-3's free tail chunks, issued here
    // so it drains behind the output gemms.
    const int CS2 = cs + SKEW;
    const int rows1 = min(cs, (cs * cs * 4 - cs * CS2 * 2) / (P * 2));
    bf16* dz1 = reinterpret_cast<bf16*>(
        reinterpret_cast<char*>(w.QK) + (long)cs * CS2 * 2);
    bf16* dz2 = reinterpret_cast<bf16*>(
        reinterpret_cast<char*>(w.dQK) + (long)cs * CS2 * 2);
    const bf16* DZg = p.DZ + ((((long)l * p.batch + b) * S + s0) * H + h) * (long)P;
    cp_rows(dz1, P, DZg, (long)H * P, rows1, P * 2, tid, nthr);
    if (rows1 < cs)
        cp_rows(dz2, P, DZg + (long)rows1 * H * P, (long)H * P,
                cs - rows1, P * 2, tid, nthr);

    // fused output contraction; tiles land in per-warp scratch for the
    // gate/mask epilogue, row-summed via smem atomics into part_s.
    const bf16* Sin_g = p.S_INB + (((long)b * H + h) * nc + c) * (long)N * P;
    const bf16* dSin_g = p.DS_INB + ((((long)l * p.batch + b) * H + h) * nc + c) * (long)N * P;
    const float dskip = p.DSKIP[h];
    float* ws = w.ws + warp * 512;  // [16][16] out + [16][16] dout, f32
    const int nt = P / 16, mt = cs / 16;
    auto dz_at = [&](int i, int q) -> float {
        return (i < rows1) ? ld(dz1 + i * P + q)
                           : ld(dz2 + (i - rows1) * P + q);
    };
    auto sink = [&](int im, int in, AccFrag& acc, AccFrag& dacc) {
        wmma::store_matrix_sync(ws, acc, 16, wmma::mem_row_major);
        wmma::store_matrix_sync(ws + 256, dacc, 16, wmma::mem_row_major);
        __syncwarp();
        for (int e = lane; e < 256; e += 32) {
            const int i = im + e / 16, q = in + e % 16;
            const float out = ws[e], dout = ws[256 + e];
            const float vv = ld(w.v + i * PS + q), dvv = ld(w.dv + i * PS + q);
            const float o = out + dskip * vv - vv * w.qk_s[i];
            const float dO = dout + dskip * dvv - (dvv * w.qk_s[i] + vv * w.dqk_s[i]);
            const float z = ld(p.Z + (((long)b * S + s0 + i) * H + h) * P + q);
            const float dz = dz_at(i, q);
            const float sg = 1.f / (1.f + expf(-z));
            const float gate = z * sg;
            const float dgate = sg * (1.f + z * (1.f - sg)) * dz;
            if (s0 + i < p.s_true)
                atomicAdd(&w.part_s[q], dO * gate + o * dgate);
        }
        __syncwarp();  // ws reads done before the next tile overwrites it
    };
    if (mt * nt == 2 * nwarps && nwarps % nt == 0) {
        const int in = (warp % nt) * 16;
        const int im0 = (warp / nt) * 16, im1 = ((warp + nwarps) / nt) * 16;
        AccFrag acc0, dacc0, acc1, dacc1;
        fused_out_tile2(w, Sin_g, dSin_g, im0, im1, in, cs, N, P,
                        acc0, dacc0, acc1, dacc1);
        cp_wait_all();     // DZ staged during the contraction
        __syncthreads();
        sink(im0, in, acc0, dacc0);
        sink(im1, in, acc1, dacc1);
    } else {
        cp_wait_all();
        __syncthreads();   // before the loop: warps may have unequal trips
        for (int t = warp; t < mt * nt; t += nwarps) {
            const int im = (t / nt) * 16, in = (t % nt) * 16;
            AccFrag acc, dacc;
            fused_out_tile(w, Sin_g, dSin_g, im, in, cs, N, P, acc, dacc);
            sink(im, in, acc, dacc);
        }
    }
    __syncthreads();
    for (int q = tid; q < P; q += nthr)
        p.PART[((((long)l * p.batch + b) * H + h) * nc + c) * P + q] = w.part_s[q];
}

size_t passC_arena_bytes(int cs, int N, int P) {
    const int NS = N + SKEW;
    return (size_t)cs * NS * 2 * 4    // pools 1+2: 4 bf16 [cs][NS] buffers
         + (size_t)cs * cs * 4 * 2    // pool 3: QK/dQK f32
         + (size_t)cs * 4 * 8;        // scalars (+ part_s, e_s)
}

// Fused A+B+C(mean) chunk walk: one CTA per (lane, h, b) holds S and dS in
// smem and walks chunks serially; the state stream never touches gmem.

// State update: S <- wc*S + B_w^T v; dS <- wc*dS + wc*dllast*S + A_w^T v
// + B_w^T dv, with wc = exp(llast_c) -- equivalent to the pass-B tri algebra.

// sm_80+ m16n16k16 f32 accumulator element mapping (two m16n8 halves);
// the parity gates are the runtime proof of this mapping.
__device__ __forceinline__ void frag_rc(int lane, int i, int& r_, int& c_) {
    r_ = (lane >> 2) + ((i & 2) ? 8 : 0);
    c_ = (lane & 3) * 2 + (i & 1) + ((i & 4) ? 8 : 0);
}

struct FusedPtrs {
    bf16 *qr, *dqr, *ksc, *dksc;     // pools 1,2 ([cs][NS]); 2 becomes A_w/B_w
    float *QK, *dQK;                 // pool 3 (angle staging -> QK -> WQKb)
    bf16 *v, *dv, *z, *dz;           // [cs][PS] each, dedicated
    bf16 *S, *dS;                    // [N][PS] running states
    float *L_s, *dL_s, *sc_s, *dsc_s, *qk_s, *dqk_s, *e_s, *wl_s, *a_s, *part_s;
    bf16 *WQKb, *dWQKb;              // packed over QK/dQK post-mask
    float *ws;                       // [nwarps][512] tile scratch
};

__device__ FusedPtrs carve_fused(char* arena, int cs, int N, int P, int nwarps) {
    const int NS = N + SKEW, PS = P + SKEW;
    FusedPtrs w;
    char* p0 = arena;
    w.qr = reinterpret_cast<bf16*>(p0); p0 += cs * NS * 2;
    w.dqr = reinterpret_cast<bf16*>(p0); p0 += cs * NS * 2;
    w.ksc = reinterpret_cast<bf16*>(p0); p0 += cs * NS * 2;
    w.dksc = reinterpret_cast<bf16*>(p0); p0 += cs * NS * 2;
    w.QK = reinterpret_cast<float*>(p0); p0 += cs * cs * 4;
    w.dQK = reinterpret_cast<float*>(p0); p0 += cs * cs * 4;
    w.v = reinterpret_cast<bf16*>(p0); p0 += cs * PS * 2;
    w.dv = reinterpret_cast<bf16*>(p0); p0 += cs * PS * 2;
    w.z = reinterpret_cast<bf16*>(p0); p0 += cs * P * 2;   // elementwise only,
    w.dz = reinterpret_cast<bf16*>(p0); p0 += cs * P * 2;  // no skew needed
    w.S = reinterpret_cast<bf16*>(p0); p0 += N * PS * 2;
    w.dS = reinterpret_cast<bf16*>(p0); p0 += N * PS * 2;
    w.L_s = reinterpret_cast<float*>(p0); p0 += cs * 4;
    w.dL_s = reinterpret_cast<float*>(p0); p0 += cs * 4;
    w.sc_s = reinterpret_cast<float*>(p0); p0 += cs * 4;
    w.dsc_s = reinterpret_cast<float*>(p0); p0 += cs * 4;
    w.qk_s = reinterpret_cast<float*>(p0); p0 += cs * 4;
    w.dqk_s = reinterpret_cast<float*>(p0); p0 += cs * 4;
    w.e_s = reinterpret_cast<float*>(p0); p0 += cs * 4;
    w.wl_s = reinterpret_cast<float*>(p0); p0 += cs * 4;
    w.a_s = reinterpret_cast<float*>(p0); p0 += cs * 4;
    w.part_s = reinterpret_cast<float*>(p0); p0 += (P > cs ? P : cs) * 4;
    w.WQKb = reinterpret_cast<bf16*>(w.QK);
    w.dWQKb = reinterpret_cast<bf16*>(w.dQK);
    w.ws = nullptr;  // dead since the fragment-native epilogues
    return w;
}

__host__ __device__ size_t fused_arena_bytes(int cs, int N, int P, int nwarps) {
    (void)nwarps;
    const int NS = N + SKEW, PS = P + SKEW;
    return (size_t)cs * NS * 2 * 4 + (size_t)cs * cs * 4 * 2
         + (size_t)cs * PS * 2 * 2 + (size_t)cs * P * 2 * 2
         + (size_t)N * PS * 2 * 2
         + (size_t)cs * 4 * 9 + (size_t)(P > cs ? P : cs) * 4;
}

template <int ABLATE, int LEAN = 0>  // 0 = full; k>0 = per chunk, stop after
// phase k. LEAN=1: no e-fold phase -- the e row-scaling commutes out of the
// state mma and is applied in the fragment-native epilogue via frag_rc.
__global__ void __launch_bounds__(256, 2) fwd_fused_mean_kernel(FwdDualscanParams p) {
    const int l = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    const int cs = p.chunk_size, N = p.d_state, P = p.headdim, Da = p.n_rot;
    const int S = p.seqlen, H = p.heads, G = p.groups;
    const int hg = h / (H / G);
    const int nc = S / cs;
    const int tid = threadIdx.x, nthr = blockDim.x;
    const int warp = tid / 32, nwarps = nthr / 32, lane = tid % 32;
    const int NS = N + SKEW, PS = P + SKEW, CS2 = cs + SKEW;

    extern __shared__ char arena[];
    FusedPtrs w = carve_fused(arena, cs, N, P, nwarps);
    for (int u = tid; u < N * PS; u += nthr) {
        w.S[u] = __float2bfloat16(0.f);
        w.dS[u] = __float2bfloat16(0.f);
    }
    const float dskip = p.DSKIP[h];

    for (int c = 0; c < nc; ++c) {
        const int s0 = c * cs;
        const long row0 = ((long)b * H + h) * S + s0;
        const long lrow0 = (((long)l * p.batch + b) * H + h) * S + s0;
        // gen: bulk cp.async into final slots + in-place transforms;
        // native-mode tangents load via registers while async is in flight.
        const int natv = p.native;
        const int St = natv ? p.s_true : S;
        const int rval = natv ? min(cs, max(0, St - s0)) : cs;
        cp_rows(w.qr, NS, p.QR + (((long)b * S + s0) * H + h) * (long)N,
                (long)H * N, cs, N * 2, tid, nthr);
        cp_rows(w.z, P, p.Z + (((long)b * S + s0) * H + h) * (long)P,
                (long)H * P, cs, P * 2, tid, nthr);
        cp_rows(w.ksc, NS, p.KR + (((long)b * S + s0) * H + h) * (long)N,
                (long)H * N, cs, N * 2, tid, nthr);
        cp_rows(w.v, PS, p.V + (((long)b * S + s0) * H + h) * (long)P,
                (long)H * P, cs, P * 2, tid, nthr);
        bf16* cos_s = reinterpret_cast<bf16*>(w.QK);
        bf16* sin_s = cos_s + cs * Da;
        bf16* dth_s = sin_s + cs * Da;
        const bool stage_ang = (Da % 8 == 0) && (3 * cs * Da * 2 <= 2 * cs * cs * 4);
        if (stage_ang) {
            cp_rows(cos_s, Da, p.COS + row0 * Da, Da, cs, Da * 2, tid, nthr);
            cp_rows(sin_s, Da, p.SIN + row0 * Da, Da, cs, Da * 2, tid, nthr);
            cp_rows(dth_s, Da, p.DTHETA + lrow0 * Da, Da, cs, Da * 2, tid, nthr);
        }
        if (natv) {
            // native strided views: bf16 pairs ride cp.async, fp32 pairs
            // convert on chip; tail rows zero-fill.
            const long qoff = (long)l * p.qk_sl + (long)b * p.qk_sb
                            + (long)s0 * p.qk_ss + (long)hg * p.qk_sg;
            const long voff = (long)l * p.vz_sl + (long)b * p.vz_sb
                            + (long)s0 * p.vz_ss + (long)h * p.vz_sh;
            if (p.nat_qk32) {
                ld_rows_f32(w.dqr, NS, p.DQRAW32 + qoff, p.qk_ss, cs, rval, N, tid, nthr);
                ld_rows_f32(w.dksc, NS, p.DKRAW32 + qoff, p.qk_ss, cs, rval, N, tid, nthr);
            } else {
                cp_rows(w.dqr, NS, p.DQRAW + qoff, p.qk_ss, rval, N * 2, tid, nthr);
                cp_rows(w.dksc, NS, p.DKRAW + qoff, p.qk_ss, rval, N * 2, tid, nthr);
                for (int u = tid; u < (cs - rval) * N; u += nthr) {
                    const int j = rval + u / N, n = u % N;
                    w.dqr[j * NS + n] = __float2bfloat16(0.f);
                    w.dksc[j * NS + n] = __float2bfloat16(0.f);
                }
            }
            if (p.nat_vz32) {
                ld_rows_f32(w.dv, PS, p.DV32 + voff, p.vz_ss, cs, rval, P, tid, nthr);
                ld_rows_f32(w.dz, P, p.DZ32 + voff, p.vz_ss, cs, rval, P, tid, nthr);
            } else {
                cp_rows(w.dv, PS, p.DV + voff, p.vz_ss, rval, P * 2, tid, nthr);
                cp_rows(w.dz, P, p.DZ + voff, p.vz_ss, rval, P * 2, tid, nthr);
                for (int u = tid; u < (cs - rval) * P; u += nthr) {
                    const int j = rval + u / P, q = u % P;
                    w.dv[j * PS + q] = __float2bfloat16(0.f);
                    w.dz[j * P + q] = __float2bfloat16(0.f);
                }
            }
        } else {
            cp_rows(w.dqr, NS, p.DQRAW + ((((long)l * p.batch + b) * S + s0) * G + hg) * (long)N,
                    (long)G * N, cs, N * 2, tid, nthr);
            cp_rows(w.dz, P, p.DZ + ((((long)l * p.batch + b) * S + s0) * H + h) * (long)P,
                    (long)H * P, cs, P * 2, tid, nthr);
            cp_rows(w.dksc, NS, p.DKRAW + ((((long)l * p.batch + b) * S + s0) * G + hg) * (long)N,
                    (long)G * N, cs, N * 2, tid, nthr);
            cp_rows(w.dv, PS, p.DV + ((((long)l * p.batch + b) * S + s0) * H + h) * (long)P,
                    (long)H * P, cs, P * 2, tid, nthr);
        }
        {
            const float llast = p.L[row0 + cs - 1];
            const float dllast = p.DL[lrow0 + cs - 1];
            for (int i = tid; i < cs; i += nthr) {
                const float Lv = p.L[row0 + i];
                w.L_s[i] = Lv;
                w.e_s[i] = expf(Lv);
                w.dL_s[i] = p.DL[lrow0 + i];
                w.sc_s[i] = p.SCALE[row0 + i];
                w.dsc_s[i] = p.DSCALE[lrow0 + i];
                w.qk_s[i] = p.QKDOT[row0 + i];
                w.dqk_s[i] = p.DQKDOT[lrow0 + i];
                const float wl = expf(llast - Lv);
                w.wl_s[i] = wl;
                w.a_s[i] = wl * (dllast - w.dL_s[i]);
            }
            for (int i = tid; i < P; i += nthr) w.part_s[i] = 0.f;
        }
        cp_wait_all();
        __syncthreads();
        if (ABLATE == 1) continue;
        if (c + 1 < nc) {  // warm L2 for the next chunk's lane-unique rows
            const long nrow0 = lrow0 + cs;
            const char* dth = reinterpret_cast<const char*>(p.DTHETA + nrow0 * Da);
            const char* dvv; const char* dzz; long vstride;
            int prows = cs;
            if (natv) {
                const long nvoff = (long)l * p.vz_sl + (long)b * p.vz_sb
                                 + (long)(s0 + cs) * p.vz_ss + (long)h * p.vz_sh;
                const int es = p.nat_vz32 ? 4 : 2;
                dvv = p.nat_vz32
                    ? reinterpret_cast<const char*>(p.DV32 + nvoff)
                    : reinterpret_cast<const char*>(p.DV + nvoff);
                dzz = p.nat_vz32
                    ? reinterpret_cast<const char*>(p.DZ32 + nvoff)
                    : reinterpret_cast<const char*>(p.DZ + nvoff);
                vstride = p.vz_ss * es;
                prows = min(cs, max(0, St - (s0 + cs)));
            } else {
                dvv = reinterpret_cast<const char*>(
                    p.DV + ((((long)l * p.batch + b) * S + s0 + cs) * H + h) * (long)P);
                dzz = reinterpret_cast<const char*>(
                    p.DZ + ((((long)l * p.batch + b) * S + s0 + cs) * H + h) * (long)P);
                vstride = (long)H * P * 2;
            }
            for (int i = tid; i < cs; i += nthr) {
                if (Da >= 64)
                    asm volatile("prefetch.global.L2 [%0];" :: "l"(dth + (long)i * Da * 2));
                if (i < prows) {
                    asm volatile("prefetch.global.L2 [%0];" :: "l"(dvv + (long)i * vstride));
                    asm volatile("prefetch.global.L2 [%0];" :: "l"(dzz + (long)i * vstride));
                }
            }
        }
        for (int idx = tid; idx < cs * (N / 2); idx += nthr) {
            const int j = idx / (N / 2), pp = idx % (N / 2);
            float cw = 1.f, sw = 0.f, dt = 0.f;
            if (pp < Da) {
                if (stage_ang) {
                    cw = ld(cos_s + j * Da + pp);
                    sw = ld(sin_s + j * Da + pp);
                    dt = ld(dth_s + j * Da + pp);
                } else {
                    cw = ld(p.COS + (row0 + j) * (long)Da + pp);
                    sw = ld(p.SIN + (row0 + j) * (long)Da + pp);
                    dt = ld(p.DTHETA + (lrow0 + j) * (long)Da + pp);
                }
            }
            const float q0 = ld(w.qr + j * NS + 2 * pp);
            const float q1 = ld(w.qr + j * NS + 2 * pp + 1);
            const float e0 = ld(w.dqr + j * NS + 2 * pp);
            const float e1 = ld(w.dqr + j * NS + 2 * pp + 1);
            w.dqr[j * NS + 2 * pp] = __float2bfloat16(e0 * cw - e1 * sw - dt * q1);
            w.dqr[j * NS + 2 * pp + 1] = __float2bfloat16(e0 * sw + e1 * cw + dt * q0);
            const float k0 = ld(w.ksc + j * NS + 2 * pp);
            const float k1 = ld(w.ksc + j * NS + 2 * pp + 1);
            const float d0 = ld(w.dksc + j * NS + 2 * pp);
            const float d1 = ld(w.dksc + j * NS + 2 * pp + 1);
            const float sc = w.sc_s[j], dsc = w.dsc_s[j];
            w.ksc[j * NS + 2 * pp] = __float2bfloat16(k0 * sc);
            w.ksc[j * NS + 2 * pp + 1] = __float2bfloat16(k1 * sc);
            w.dksc[j * NS + 2 * pp] =
                __float2bfloat16((d0 * cw - d1 * sw - dt * k1) * sc + k0 * dsc);
            w.dksc[j * NS + 2 * pp + 1] =
                __float2bfloat16((d0 * sw + d1 * cw + dt * k0) * sc + k1 * dsc);
        }
        __syncthreads();

        // fused QK+mask: tile pairs stay in registers and write bf16 WQKb
        // directly; above-diagonal tiles skip their gemms entirely.
        {
            const int qt = cs / 16;
            for (int t = warp; t < qt * qt; t += nwarps) {
                const int im = (t / qt) * 16, in = (t % qt) * 16;
                if (im + 15 >= in) {
                    AccFrag acc, dacc;
                    wmma::fill_fragment(acc, 0.f);
                    wmma::fill_fragment(dacc, 0.f);
                    for (int k = 0; k < N / 16; ++k) {
                        wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a, da;
                        wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::col_major> kb, dkb;
                        wmma::load_matrix_sync(a, w.qr + im * NS + k * 16, NS);
                        wmma::load_matrix_sync(da, w.dqr + im * NS + k * 16, NS);
                        wmma::load_matrix_sync(kb, w.ksc + in * NS + k * 16, NS);
                        wmma::load_matrix_sync(dkb, w.dksc + in * NS + k * 16, NS);
                        wmma::mma_sync(acc, a, kb, acc);
                        wmma::mma_sync(dacc, da, kb, dacc);
                        wmma::mma_sync(dacc, a, dkb, dacc);
                    }
#pragma unroll
                    for (int e2 = 0; e2 < 8; ++e2) {  // fragment-native
                        int rr, cc; frag_rc(lane, e2, rr, cc);
                        const int i = im + rr, j = in + cc;
                        float wq = 0.f, dwq = 0.f;
                        if (i >= j) {
                            const float wgt = expf(w.L_s[i] - w.L_s[j]);
                            wq = wgt * acc.x[e2];
                            dwq = wgt * (w.dL_s[i] - w.dL_s[j]) * acc.x[e2]
                                + wgt * dacc.x[e2];
                        }
                        w.WQKb[i * CS2 + j] = __float2bfloat16(wq);
                        w.dWQKb[i * CS2 + j] = __float2bfloat16(dwq);
                    }
                }  // above-diagonal tiles are never read (causal intra k-bound)
            }
        }
        __syncthreads();
        if (ABLATE == 2) continue;

        // e-fold qr/dqr in place; LEAN folds this into the contraction
        // epilogue instead.
        if (!LEAN) {
            for (int idx = tid; idx < cs * N; idx += nthr) {
                const int i = idx / N, n = idx % N;
                const float e = w.e_s[i];
                const float qv = ld(w.qr + i * NS + n);
                const float dqv = ld(w.dqr + i * NS + n);
                w.dqr[i * NS + n] = __float2bfloat16(e * dqv + e * w.dL_s[i] * qv);
                w.qr[i * NS + n] = __float2bfloat16(e * qv);
            }
            __syncthreads();
        }
        if (ABLATE == 3) continue;
        // output contraction + mean sink (entering state from smem); the
        // A_w/B_w in-place build rides this phase.
        for (int idx = tid; idx < cs * N; idx += nthr) {
            const int j = idx / N, n = idx % N;
            const float kv = ld(w.ksc + j * NS + n);
            const float dkv = ld(w.dksc + j * NS + n);
            w.ksc[j * NS + n] = __float2bfloat16(w.wl_s[j] * kv);            // B_w
            w.dksc[j * NS + n] = __float2bfloat16(w.a_s[j] * kv + w.wl_s[j] * dkv);  // A_w
        }
        {
            const int nt = P / 16, mt = cs / 16;
            for (int t = warp; t < mt * nt; t += nwarps) {
                const int im = (t / nt) * 16, in = (t % nt) * 16;
                AccFrag acc, dacc, accV, daccV;
                wmma::fill_fragment(acc, 0.f);
                wmma::fill_fragment(dacc, 0.f);
                if (LEAN) {                     // WQKb part kept separate:
                    wmma::fill_fragment(accV, 0.f);   // only the STATE part
                    wmma::fill_fragment(daccV, 0.f);  // carries the e-fold
                }
                for (int k = 0; k < N / 16; ++k) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a, da;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> sb, dsb;
                    wmma::load_matrix_sync(a, w.qr + im * NS + k * 16, NS);
                    wmma::load_matrix_sync(da, w.dqr + im * NS + k * 16, NS);
                    wmma::load_matrix_sync(sb, w.S + (k * 16) * PS + in, PS);
                    wmma::load_matrix_sync(dsb, w.dS + (k * 16) * PS + in, PS);
                    wmma::mma_sync(acc, a, sb, acc);
                    wmma::mma_sync(dacc, da, sb, dacc);
                    wmma::mma_sync(dacc, a, dsb, dacc);
                }
                for (int k = 0; k <= im / 16; ++k) {  // causal: WQKb is 0 above
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a, da;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> vb, dvb;
                    wmma::load_matrix_sync(a, w.WQKb + im * CS2 + k * 16, CS2);
                    wmma::load_matrix_sync(da, w.dWQKb + im * CS2 + k * 16, CS2);
                    wmma::load_matrix_sync(vb, w.v + (k * 16) * PS + in, PS);
                    wmma::load_matrix_sync(dvb, w.dv + (k * 16) * PS + in, PS);
                    wmma::mma_sync(LEAN ? accV : acc, a, vb, LEAN ? accV : acc);
                    wmma::mma_sync(LEAN ? daccV : dacc, da, vb, LEAN ? daccV : dacc);
                    wmma::mma_sync(LEAN ? daccV : dacc, a, dvb, LEAN ? daccV : dacc);
                }
                // fragment-native sink: per-thread column partials, then 4
                // atomics; LEAN applies the e-fold here.
                {
                    float pq[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                    for (int e2 = 0; e2 < 8; ++e2) {
                        int rr, cc; frag_rc(lane, e2, rr, cc);
                        const int i = im + rr, q = in + cc;
                        float out, dout;
                        if (LEAN) {
                            const float e = w.e_s[i];
                            out = e * acc.x[e2] + accV.x[e2];
                            dout = e * (dacc.x[e2] + w.dL_s[i] * acc.x[e2])
                                 + daccV.x[e2];
                        } else {
                            out = acc.x[e2]; dout = dacc.x[e2];
                        }
                        const float vv = ld(w.v + i * PS + q), dvv = ld(w.dv + i * PS + q);
                        const float o = out + dskip * vv - vv * w.qk_s[i];
                        const float dO = dout + dskip * dvv - (dvv * w.qk_s[i] + vv * w.dqk_s[i]);
                        const float z = ld(w.z + i * P + q);
                        const float dz = ld(w.dz + i * P + q);
                        const float sg = 1.f / (1.f + expf(-z));
                        const float gate = z * sg;
                        const float dgate = sg * (1.f + z * (1.f - sg)) * dz;
                        if (s0 + i < p.s_true)
                            pq[(e2 & 1) | ((e2 & 4) >> 1)] += dO * gate + o * dgate;
                    }
#pragma unroll
                    for (int jj = 0; jj < 4; ++jj)
                        atomicAdd(&w.part_s[in + (lane & 3) * 2 + (jj & 1)
                                            + ((jj & 2) ? 8 : 0)], pq[jj]);
                }
            }
        }
        __syncthreads();

        if (ABLATE == 4) continue;
        // ---- state update:
        // S <- wc*S + B_w^T v, dS <- wc*dS + wc*dllast*S + A_w^T v + B_w^T dv
        if (LEAN && P / 16 <= nwarps && nwarps % (P / 16) == 0
            && N / 16 == 4 * (nwarps / (P / 16))) {
            // column-paired update: each warp owns one in-column x 4 row
            // tiles; v/dv fragments load once per k-step, shared across 4.
            const float wc = w.e_s[cs - 1];
            const float dcdc = wc * w.dL_s[cs - 1];
            const int nt = P / 16;
            const int col = warp % nt, sub = warp / nt;
            const int in = col * 16;
            AccFrag accp[4], acct[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                wmma::fill_fragment(accp[j], 0.f);
                wmma::fill_fragment(acct[j], 0.f);
            }
            for (int k = 0; k < cs / 16; ++k) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> vb, dvb;
                wmma::load_matrix_sync(vb, w.v + (k * 16) * PS + in, PS);
                wmma::load_matrix_sync(dvb, w.dv + (k * 16) * PS + in, PS);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int im = (sub * 4 + j) * 16;
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::col_major> aB, aA;
                    wmma::load_matrix_sync(aB, w.ksc + (k * 16) * NS + im, NS);
                    wmma::load_matrix_sync(aA, w.dksc + (k * 16) * NS + im, NS);
                    wmma::mma_sync(accp[j], aB, vb, accp[j]);
                    wmma::mma_sync(acct[j], aA, vb, acct[j]);
                    wmma::mma_sync(acct[j], aB, dvb, acct[j]);
                }
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int im = (sub * 4 + j) * 16;
#pragma unroll
                for (int e2 = 0; e2 < 8; ++e2) {
                    int rr, cc; frag_rc(lane, e2, rr, cc);
                    const int n_ = im + rr, q = in + cc;
                    const float s_old = ld(w.S + n_ * PS + q);
                    const float ds_old = ld(w.dS + n_ * PS + q);
                    w.S[n_ * PS + q] = __float2bfloat16(wc * s_old + accp[j].x[e2]);
                    w.dS[n_ * PS + q] =
                        __float2bfloat16(wc * ds_old + dcdc * s_old + acct[j].x[e2]);
                }
            }
        } else {
            const float wc = w.e_s[cs - 1];             // exp(llast_c)
            const float dcdc = wc * w.dL_s[cs - 1];     // wc * dllast_c
            const int nt = P / 16, mt = N / 16;
            for (int t = warp; t < mt * nt; t += nwarps) {
                const int im = (t / nt) * 16, in = (t % nt) * 16;
                AccFrag accp, acct;
                wmma::fill_fragment(accp, 0.f);
                wmma::fill_fragment(acct, 0.f);
                for (int k = 0; k < cs / 16; ++k) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::col_major> aB, aA;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> vb, dvb;
                    wmma::load_matrix_sync(aB, w.ksc + (k * 16) * NS + im, NS);
                    wmma::load_matrix_sync(aA, w.dksc + (k * 16) * NS + im, NS);
                    wmma::load_matrix_sync(vb, w.v + (k * 16) * PS + in, PS);
                    wmma::load_matrix_sync(dvb, w.dv + (k * 16) * PS + in, PS);
                    wmma::mma_sync(accp, aB, vb, accp);
                    wmma::mma_sync(acct, aA, vb, acct);
                    wmma::mma_sync(acct, aB, dvb, acct);
                }
#pragma unroll
                for (int e2 = 0; e2 < 8; ++e2) {  // fragment-native commit
                    int rr, cc; frag_rc(lane, e2, rr, cc);
                    const int n_ = im + rr, q = in + cc;
                    const float s_old = ld(w.S + n_ * PS + q);
                    const float ds_old = ld(w.dS + n_ * PS + q);
                    w.S[n_ * PS + q] = __float2bfloat16(wc * s_old + accp.x[e2]);
                    w.dS[n_ * PS + q] =
                        __float2bfloat16(wc * ds_old + dcdc * s_old + acct.x[e2]);
                }
            }
        }
        // (part_s is complete -- its atomics preceded the last barrier)
        for (int q = tid; q < P; q += nthr)
            p.PART[((((long)l * p.batch + b) * H + h) * nc + c) * P + q] = w.part_s[q];
        __syncthreads();
    }
}


// Chunk-parallel ticket-chained fused walk: one CTA per (lane, h, b, chunk),
// state-independent work ahead of the poll, publish-early, states via L2.
template <int NOCHAIN>  // 1 = skip poll/fences/publish-order (timing ablation only)
__global__ void __launch_bounds__(256, 2) fwd_r8a_mean_kernel(FwdDualscanParams p) {
    // chain-major scheduling: all chains at chunk c launch before chunk
    // c+1, so a CTA's predecessor has almost always retired when it polls.
    const int c = blockIdx.z;
    const int h = blockIdx.x;
    const int l = blockIdx.y / p.batch;
    const int b = blockIdx.y % p.batch;
    const int cs = p.chunk_size, N = p.d_state, P = p.headdim, Da = p.n_rot;
    const int S = p.seqlen, H = p.heads, G = p.groups;
    const int hg = h / (H / G);
    const int nc = S / cs;
    const int tid = threadIdx.x, nthr = blockDim.x;
    const int warp = tid / 32, nwarps = nthr / 32, lane = tid % 32;
    const int NS = N + SKEW, PS = P + SKEW, CS2 = cs + SKEW;

    extern __shared__ char arena[];
    FusedPtrs w = carve_fused(arena, cs, N, P, nwarps);  // w.S/w.dS = local SC/DSC
    const float dskip = p.DSKIP[h];
    const int s0 = c * cs;
    const long row0 = ((long)b * H + h) * S + s0;
    const long lrow0 = (((long)l * p.batch + b) * H + h) * S + s0;

    // ---- gen: bulk cp.async into final slots (emitted bf16 fields) ----
    cp_rows(w.qr, NS, p.QR + (((long)b * S + s0) * H + h) * (long)N,
            (long)H * N, cs, N * 2, tid, nthr);
    cp_rows(w.ksc, NS, p.KR + (((long)b * S + s0) * H + h) * (long)N,
            (long)H * N, cs, N * 2, tid, nthr);
    cp_rows(w.v, PS, p.V + (((long)b * S + s0) * H + h) * (long)P,
            (long)H * P, cs, P * 2, tid, nthr);
    cp_rows(w.z, P, p.Z + (((long)b * S + s0) * H + h) * (long)P,
            (long)H * P, cs, P * 2, tid, nthr);
    cp_rows(w.dqr, NS, p.DQRAW + ((((long)l * p.batch + b) * S + s0) * G + hg) * (long)N,
            (long)G * N, cs, N * 2, tid, nthr);
    cp_rows(w.dksc, NS, p.DKRAW + ((((long)l * p.batch + b) * S + s0) * G + hg) * (long)N,
            (long)G * N, cs, N * 2, tid, nthr);
    cp_rows(w.dv, PS, p.DV + ((((long)l * p.batch + b) * S + s0) * H + h) * (long)P,
            (long)H * P, cs, P * 2, tid, nthr);
    cp_rows(w.dz, P, p.DZ + ((((long)l * p.batch + b) * S + s0) * H + h) * (long)P,
            (long)H * P, cs, P * 2, tid, nthr);
    bf16* cos_s = reinterpret_cast<bf16*>(w.QK);
    bf16* sin_s = cos_s + cs * Da;
    bf16* dth_s = sin_s + cs * Da;
    const bool stage_ang = (Da % 8 == 0) && (3 * cs * Da * 2 <= 2 * cs * cs * 4);
    if (stage_ang) {
        cp_rows(cos_s, Da, p.COS + row0 * Da, Da, cs, Da * 2, tid, nthr);
        cp_rows(sin_s, Da, p.SIN + row0 * Da, Da, cs, Da * 2, tid, nthr);
        cp_rows(dth_s, Da, p.DTHETA + lrow0 * Da, Da, cs, Da * 2, tid, nthr);
    }
    {
        const float llast = p.L[row0 + cs - 1];
        const float dllast = p.DL[lrow0 + cs - 1];
        for (int i = tid; i < cs; i += nthr) {
            const float Lv = p.L[row0 + i];
            w.L_s[i] = Lv;
            w.e_s[i] = expf(Lv);
            w.dL_s[i] = p.DL[lrow0 + i];
            w.sc_s[i] = p.SCALE[row0 + i];
            w.dsc_s[i] = p.DSCALE[lrow0 + i];
            w.qk_s[i] = p.QKDOT[row0 + i];
            w.dqk_s[i] = p.DQKDOT[lrow0 + i];
            const float wl = expf(llast - Lv);
            w.wl_s[i] = wl;
            w.a_s[i] = wl * (dllast - w.dL_s[i]);
        }
        for (int i = tid; i < P; i += nthr) w.part_s[i] = 0.f;
    }
    cp_wait_all();
    __syncthreads();

    // ---- in-place transforms (rotary tangent + K scaling) ----
    for (int idx = tid; idx < cs * (N / 2); idx += nthr) {
        const int j = idx / (N / 2), pp = idx % (N / 2);
        float cw = 1.f, sw = 0.f, dt = 0.f;
        if (pp < Da) {
            if (stage_ang) {
                cw = ld(cos_s + j * Da + pp);
                sw = ld(sin_s + j * Da + pp);
                dt = ld(dth_s + j * Da + pp);
            } else {
                cw = ld(p.COS + (row0 + j) * (long)Da + pp);
                sw = ld(p.SIN + (row0 + j) * (long)Da + pp);
                dt = ld(p.DTHETA + (lrow0 + j) * (long)Da + pp);
            }
        }
        const float q0 = ld(w.qr + j * NS + 2 * pp);
        const float q1 = ld(w.qr + j * NS + 2 * pp + 1);
        const float e0 = ld(w.dqr + j * NS + 2 * pp);
        const float e1 = ld(w.dqr + j * NS + 2 * pp + 1);
        w.dqr[j * NS + 2 * pp] = __float2bfloat16(e0 * cw - e1 * sw - dt * q1);
        w.dqr[j * NS + 2 * pp + 1] = __float2bfloat16(e0 * sw + e1 * cw + dt * q0);
        const float k0 = ld(w.ksc + j * NS + 2 * pp);
        const float k1 = ld(w.ksc + j * NS + 2 * pp + 1);
        const float d0 = ld(w.dksc + j * NS + 2 * pp);
        const float d1 = ld(w.dksc + j * NS + 2 * pp + 1);
        const float sc = w.sc_s[j], dsc = w.dsc_s[j];
        w.ksc[j * NS + 2 * pp] = __float2bfloat16(k0 * sc);
        w.ksc[j * NS + 2 * pp + 1] = __float2bfloat16(k1 * sc);
        w.dksc[j * NS + 2 * pp] =
            __float2bfloat16((d0 * cw - d1 * sw - dt * k1) * sc + k0 * dsc);
        w.dksc[j * NS + 2 * pp + 1] =
            __float2bfloat16((d0 * sw + d1 * cw + dt * k0) * sc + k1 * dsc);
    }
    __syncthreads();

    // ---- fused QK+mask -> bf16 WQKb (state-independent) ----
    {
        const int qt = cs / 16;
        for (int t = warp; t < qt * qt; t += nwarps) {
            const int im = (t / qt) * 16, in = (t % qt) * 16;
            if (im + 15 >= in) {
                AccFrag acc, dacc;
                wmma::fill_fragment(acc, 0.f);
                wmma::fill_fragment(dacc, 0.f);
                for (int k = 0; k < N / 16; ++k) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a, da;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::col_major> kb, dkb;
                    wmma::load_matrix_sync(a, w.qr + im * NS + k * 16, NS);
                    wmma::load_matrix_sync(da, w.dqr + im * NS + k * 16, NS);
                    wmma::load_matrix_sync(kb, w.ksc + in * NS + k * 16, NS);
                    wmma::load_matrix_sync(dkb, w.dksc + in * NS + k * 16, NS);
                    wmma::mma_sync(acc, a, kb, acc);
                    wmma::mma_sync(dacc, da, kb, dacc);
                    wmma::mma_sync(dacc, a, dkb, dacc);
                }
#pragma unroll
                for (int e2 = 0; e2 < 8; ++e2) {
                    int rr, cc; frag_rc(lane, e2, rr, cc);
                    const int i = im + rr, j = in + cc;
                    float wq = 0.f, dwq = 0.f;
                    if (i >= j) {
                        const float wgt = expf(w.L_s[i] - w.L_s[j]);
                        wq = wgt * acc.x[e2];
                        dwq = wgt * (w.dL_s[i] - w.dL_s[j]) * acc.x[e2]
                            + wgt * dacc.x[e2];
                    }
                    w.WQKb[i * CS2 + j] = __float2bfloat16(wq);
                    w.dWQKb[i * CS2 + j] = __float2bfloat16(dwq);
                }
            }
        }
    }
    __syncthreads();

    // ---- e-fold qr/dqr in place (state-independent) ----
    for (int idx = tid; idx < cs * N; idx += nthr) {
        const int i = idx / N, n = idx % N;
        const float e = w.e_s[i];
        const float qv = ld(w.qr + i * NS + n);
        const float dqv = ld(w.dqr + i * NS + n);
        w.dqr[i * NS + n] = __float2bfloat16(e * dqv + e * w.dL_s[i] * qv);
        w.qr[i * NS + n] = __float2bfloat16(e * qv);
    }
    // ---- A_w/B_w build (state-independent; ksc/dksc die into A_w/B_w) ----
    for (int idx = tid; idx < cs * N; idx += nthr) {
        const int j = idx / N, n = idx % N;
        const float kv = ld(w.ksc + j * NS + n);
        const float dkv = ld(w.dksc + j * NS + n);
        w.ksc[j * NS + n] = __float2bfloat16(w.wl_s[j] * kv);            // B_w
        w.dksc[j * NS + n] = __float2bfloat16(w.a_s[j] * kv + w.wl_s[j] * dkv);  // A_w
    }
    __syncthreads();

    // ---- local chunk state contributions SC = B_w^T v, DSC = A_w^T v +
    // B_w^T dv -> bf16 into w.S/w.dS (state-independent) ----
    {
        const int nt = P / 16, mt = N / 16;
        for (int t = warp; t < mt * nt; t += nwarps) {
            const int im = (t / nt) * 16, in = (t % nt) * 16;
            AccFrag accp, acct;
            wmma::fill_fragment(accp, 0.f);
            wmma::fill_fragment(acct, 0.f);
            for (int k = 0; k < cs / 16; ++k) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::col_major> aB, aA;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> vb, dvb;
                wmma::load_matrix_sync(aB, w.ksc + (k * 16) * NS + im, NS);
                wmma::load_matrix_sync(aA, w.dksc + (k * 16) * NS + im, NS);
                wmma::load_matrix_sync(vb, w.v + (k * 16) * PS + in, PS);
                wmma::load_matrix_sync(dvb, w.dv + (k * 16) * PS + in, PS);
                wmma::mma_sync(accp, aB, vb, accp);
                wmma::mma_sync(acct, aA, vb, acct);
                wmma::mma_sync(acct, aB, dvb, acct);
            }
#pragma unroll
            for (int e2 = 0; e2 < 8; ++e2) {
                int rr, cc; frag_rc(lane, e2, rr, cc);
                const int n_ = im + rr, q = in + cc;
                w.S[n_ * PS + q] = __float2bfloat16(accp.x[e2]);
                w.dS[n_ * PS + q] = __float2bfloat16(acct.x[e2]);
            }
        }
    }
    __syncthreads();

    // ---- ticket chain: poll(c-1), update, publish(+flag) ----
    const long chain = ((long)l * p.batch + b) * H + h;
    const long slotP = (chain * nc + (c - 1)) * (long)N * P;
    const long slotC = (chain * nc + c) * (long)N * P;
    if (!NOCHAIN && c > 0) {
        if (tid == 0) {
            while (atomicAdd(p.FLAGS + chain * nc + (c - 1), 0) == 0)
                __nanosleep(64);
        }
        __syncthreads();
        __threadfence();
    }
    {
        const float wc = w.e_s[cs - 1];             // exp(llast_c)
        const float dcdc = wc * w.dL_s[cs - 1];     // wc * dllast_c
        for (int idx = tid; idx < N * P; idx += nthr) {
            const int n_ = idx / P, q = idx % P;
            const float sp = (c > 0) ? ld(p.SB_OUT + slotP + idx) : 0.f;
            const float dsp = (c > 0) ? ld(p.DSB_OUT + slotP + idx) : 0.f;
            p.SB_OUT[slotC + idx] =
                __float2bfloat16(wc * sp + ld(w.S + n_ * PS + q));
            p.DSB_OUT[slotC + idx] =
                __float2bfloat16(wc * dsp + dcdc * sp + ld(w.dS + n_ * PS + q));
        }
    }
    if (!NOCHAIN) {
        __threadfence();
        __syncthreads();
        if (tid == 0) atomicExch(p.FLAGS + chain * nc + c, 1);
    } else {
        __syncthreads();
    }

    // ---- readout with the ENTERING state (gmem, L2-hot), off the chain ----
    {
        const int nt = P / 16, mt = cs / 16;
        for (int t = warp; t < mt * nt; t += nwarps) {
            const int im = (t / nt) * 16, in = (t % nt) * 16;
            AccFrag acc, dacc;
            wmma::fill_fragment(acc, 0.f);
            wmma::fill_fragment(dacc, 0.f);
            if (c > 0) {
                for (int k = 0; k < N / 16; ++k) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a, da;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> sb, dsb;
                    wmma::load_matrix_sync(a, w.qr + im * NS + k * 16, NS);
                    wmma::load_matrix_sync(da, w.dqr + im * NS + k * 16, NS);
                    wmma::load_matrix_sync(sb, p.SB_OUT + slotP + (k * 16) * (long)P + in, P);
                    wmma::load_matrix_sync(dsb, p.DSB_OUT + slotP + (k * 16) * (long)P + in, P);
                    wmma::mma_sync(acc, a, sb, acc);
                    wmma::mma_sync(dacc, da, sb, dacc);
                    wmma::mma_sync(dacc, a, dsb, dacc);
                }
            }
            for (int k = 0; k <= im / 16; ++k) {  // causal: WQKb is 0 above
                wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a, da;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> vb, dvb;
                wmma::load_matrix_sync(a, w.WQKb + im * CS2 + k * 16, CS2);
                wmma::load_matrix_sync(da, w.dWQKb + im * CS2 + k * 16, CS2);
                wmma::load_matrix_sync(vb, w.v + (k * 16) * PS + in, PS);
                wmma::load_matrix_sync(dvb, w.dv + (k * 16) * PS + in, PS);
                wmma::mma_sync(acc, a, vb, acc);
                wmma::mma_sync(dacc, da, vb, dacc);
                wmma::mma_sync(dacc, a, dvb, dacc);
            }
            {
                float pq[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                for (int e2 = 0; e2 < 8; ++e2) {
                    int rr, cc; frag_rc(lane, e2, rr, cc);
                    const int i = im + rr, q = in + cc;
                    const float out = acc.x[e2], dout = dacc.x[e2];
                    const float vv = ld(w.v + i * PS + q), dvv = ld(w.dv + i * PS + q);
                    const float o = out + dskip * vv - vv * w.qk_s[i];
                    const float dO = dout + dskip * dvv - (dvv * w.qk_s[i] + vv * w.dqk_s[i]);
                    const float z = ld(w.z + i * P + q);
                    const float dz = ld(w.dz + i * P + q);
                    const float sg = 1.f / (1.f + expf(-z));
                    const float gate = z * sg;
                    const float dgate = sg * (1.f + z * (1.f - sg)) * dz;
                    if (s0 + i < p.s_true)
                        pq[(e2 & 1) | ((e2 & 4) >> 1)] += dO * gate + o * dgate;
                }
#pragma unroll
                for (int jj = 0; jj < 4; ++jj)
                    atomicAdd(&w.part_s[in + (lane & 3) * 2 + (jj & 1)
                                        + ((jj & 2) ? 8 : 0)], pq[jj]);
            }
        }
    }
    __syncthreads();
    for (int q = tid; q < P; q += nthr)
        p.PART[((((long)l * p.batch + b) * H + h) * nc + c) * P + q] = w.part_s[q];
}

// rb=2 lane-pair fused walk: one CTA per lane pair; the primal walk is paid
// once and amortized over two tangent lanes (fits W=2 at ds<=64 bf16).
struct Fused2Extra {
    bf16 *dqr2, *dksc2, *dv2, *dz2, *dS2, *dWQKb2;
    float *dL2, *dsc2, *dqk2, *a2, *part2;
};

__device__ Fused2Extra carve_fused2(char* p0, int cs, int N, int P) {
    const int NS = N + SKEW, PS = P + SKEW, CS2 = cs + SKEW;
    Fused2Extra x;
    x.dqr2 = reinterpret_cast<bf16*>(p0); p0 += cs * NS * 2;
    x.dksc2 = reinterpret_cast<bf16*>(p0); p0 += cs * NS * 2;
    x.dv2 = reinterpret_cast<bf16*>(p0); p0 += cs * PS * 2;
    x.dz2 = reinterpret_cast<bf16*>(p0); p0 += cs * P * 2;
    x.dS2 = reinterpret_cast<bf16*>(p0); p0 += N * PS * 2;
    x.dWQKb2 = reinterpret_cast<bf16*>(p0); p0 += cs * CS2 * 2;
    x.dL2 = reinterpret_cast<float*>(p0); p0 += cs * 4;
    x.dsc2 = reinterpret_cast<float*>(p0); p0 += cs * 4;
    x.dqk2 = reinterpret_cast<float*>(p0); p0 += cs * 4;
    x.a2 = reinterpret_cast<float*>(p0); p0 += cs * 4;
    x.part2 = reinterpret_cast<float*>(p0);
    return x;
}

size_t fused2_extra_bytes(int cs, int N, int P) {
    const int NS = N + SKEW, PS = P + SKEW, CS2 = cs + SKEW;
    return (size_t)cs * NS * 2 * 2 + (size_t)cs * PS * 2 + (size_t)cs * P * 2
         + (size_t)N * PS * 2 + (size_t)cs * CS2 * 2
         + (size_t)cs * 4 * 4 + (size_t)(P > cs ? P : cs) * 4;
}

__global__ void __launch_bounds__(256, 2) fwd_fused_mean2_kernel(FwdDualscanParams p) {
    const int l0 = blockIdx.x * 2;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    const int cs = p.chunk_size, N = p.d_state, P = p.headdim, Da = p.n_rot;
    const int S = p.seqlen, H = p.heads, G = p.groups;
    const int hg = h / (H / G);
    const int nc = S / cs;
    const int tid = threadIdx.x, nthr = blockDim.x;
    const int warp = tid / 32, nwarps = nthr / 32, lane = tid % 32;
    const int NS = N + SKEW, PS = P + SKEW, CS2 = cs + SKEW;

    extern __shared__ char arena[];
    FusedPtrs w = carve_fused(arena, cs, N, P, nwarps);
    Fused2Extra x = carve_fused2(
        arena + fused_arena_bytes(cs, N, P, nwarps), cs, N, P);
    bf16* dqrL[2] = {w.dqr, x.dqr2};
    bf16* dkscL[2] = {w.dksc, x.dksc2};
    bf16* dvL[2] = {w.dv, x.dv2};
    bf16* dzL[2] = {w.dz, x.dz2};
    bf16* dSL[2] = {w.dS, x.dS2};
    bf16* dWL[2] = {w.dWQKb, x.dWQKb2};
    float* dLL[2] = {w.dL_s, x.dL2};
    float* dscL[2] = {w.dsc_s, x.dsc2};
    float* dqkL[2] = {w.dqk_s, x.dqk2};
    float* aL[2] = {w.a_s, x.a2};
    float* partL[2] = {w.part_s, x.part2};
    for (int u = tid; u < N * PS; u += nthr) {
        w.S[u] = __float2bfloat16(0.f);
        w.dS[u] = __float2bfloat16(0.f);
        x.dS2[u] = __float2bfloat16(0.f);
    }
    const float dskip = p.DSKIP[h];

    for (int c = 0; c < nc; ++c) {
        const int s0 = c * cs;
        const long row0 = ((long)b * H + h) * S + s0;
        cp_rows(w.qr, NS, p.QR + (((long)b * S + s0) * H + h) * (long)N,
                (long)H * N, cs, N * 2, tid, nthr);
        cp_rows(w.ksc, NS, p.KR + (((long)b * S + s0) * H + h) * (long)N,
                (long)H * N, cs, N * 2, tid, nthr);
        cp_rows(w.v, PS, p.V + (((long)b * S + s0) * H + h) * (long)P,
                (long)H * P, cs, P * 2, tid, nthr);
        cp_rows(w.z, P, p.Z + (((long)b * S + s0) * H + h) * (long)P,
                (long)H * P, cs, P * 2, tid, nthr);
        for (int ll = 0; ll < 2; ++ll) {
            const long lb = (long)(l0 + ll) * p.batch + b;
            cp_rows(dqrL[ll], NS, p.DQRAW + (((lb) * S + s0) * G + hg) * (long)N,
                    (long)G * N, cs, N * 2, tid, nthr);
            cp_rows(dkscL[ll], NS, p.DKRAW + (((lb) * S + s0) * G + hg) * (long)N,
                    (long)G * N, cs, N * 2, tid, nthr);
            cp_rows(dvL[ll], PS, p.DV + (((lb) * S + s0) * H + h) * (long)P,
                    (long)H * P, cs, P * 2, tid, nthr);
            cp_rows(dzL[ll], P, p.DZ + (((lb) * S + s0) * H + h) * (long)P,
                    (long)H * P, cs, P * 2, tid, nthr);
        }
        {
            const float llast = p.L[row0 + cs - 1];
            for (int i = tid; i < cs; i += nthr) {
                const float Lv = p.L[row0 + i];
                w.L_s[i] = Lv;
                w.e_s[i] = expf(Lv);
                w.sc_s[i] = p.SCALE[row0 + i];
                w.qk_s[i] = p.QKDOT[row0 + i];
                const float wl = expf(llast - Lv);
                w.wl_s[i] = wl;
                for (int ll = 0; ll < 2; ++ll) {
                    const long lr0 = (((long)(l0 + ll) * p.batch + b) * H + h) * S + s0;
                    const float dLv = p.DL[lr0 + i];
                    dLL[ll][i] = dLv;
                    dscL[ll][i] = p.DSCALE[lr0 + i];
                    dqkL[ll][i] = p.DQKDOT[lr0 + i];
                    aL[ll][i] = wl * (p.DL[lr0 + cs - 1] - dLv);
                }
            }
            for (int i = tid; i < P; i += nthr) {
                w.part_s[i] = 0.f;
                x.part2[i] = 0.f;
            }
        }
        cp_wait_all();
        __syncthreads();
        for (int idx = tid; idx < cs * (N / 2); idx += nthr) {
            const int j = idx / (N / 2), pp = idx % (N / 2);
            float cw = 1.f, sw = 0.f;
            if (pp < Da) {
                const long arow = (row0 + j) * (long)Da + pp;
                cw = ld(p.COS + arow);
                sw = ld(p.SIN + arow);
            }
            const float q0 = ld(w.qr + j * NS + 2 * pp);
            const float q1 = ld(w.qr + j * NS + 2 * pp + 1);
            const float k0 = ld(w.ksc + j * NS + 2 * pp);
            const float k1 = ld(w.ksc + j * NS + 2 * pp + 1);
            const float sc = w.sc_s[j];
            for (int ll = 0; ll < 2; ++ll) {
                const long lr0 = (((long)(l0 + ll) * p.batch + b) * H + h) * S + s0;
                const float dt = (pp < Da)
                    ? ld(p.DTHETA + (lr0 + j) * (long)Da + pp) : 0.f;
                const float e0 = ld(dqrL[ll] + j * NS + 2 * pp);
                const float e1 = ld(dqrL[ll] + j * NS + 2 * pp + 1);
                dqrL[ll][j * NS + 2 * pp] = __float2bfloat16(e0 * cw - e1 * sw - dt * q1);
                dqrL[ll][j * NS + 2 * pp + 1] = __float2bfloat16(e0 * sw + e1 * cw + dt * q0);
                const float d0 = ld(dkscL[ll] + j * NS + 2 * pp);
                const float d1 = ld(dkscL[ll] + j * NS + 2 * pp + 1);
                const float dsc = dscL[ll][j];
                dkscL[ll][j * NS + 2 * pp] =
                    __float2bfloat16((d0 * cw - d1 * sw - dt * k1) * sc + k0 * dsc);
                dkscL[ll][j * NS + 2 * pp + 1] =
                    __float2bfloat16((d0 * sw + d1 * cw + dt * k0) * sc + k1 * dsc);
            }
            w.ksc[j * NS + 2 * pp] = __float2bfloat16(k0 * sc);
            w.ksc[j * NS + 2 * pp + 1] = __float2bfloat16(k1 * sc);
        }
        __syncthreads();

        // fused QK+mask over 2*qt*qt virtual tiles (primal QK recomputed
        // per lane tile; lane 0 also writes the shared WQKb)
        {
            const int qt = cs / 16;
            for (int t = warp; t < 2 * qt * qt; t += nwarps) {
                const int ll = t / (qt * qt), tt = t % (qt * qt);
                const int im = (tt / qt) * 16, in = (tt % qt) * 16;
                if (im + 15 < in) continue;
                AccFrag acc, dacc;
                wmma::fill_fragment(acc, 0.f);
                wmma::fill_fragment(dacc, 0.f);
                for (int k = 0; k < N / 16; ++k) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a, da;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::col_major> kb, dkb;
                    wmma::load_matrix_sync(a, w.qr + im * NS + k * 16, NS);
                    wmma::load_matrix_sync(da, dqrL[ll] + im * NS + k * 16, NS);
                    wmma::load_matrix_sync(kb, w.ksc + in * NS + k * 16, NS);
                    wmma::load_matrix_sync(dkb, dkscL[ll] + in * NS + k * 16, NS);
                    wmma::mma_sync(acc, a, kb, acc);
                    wmma::mma_sync(dacc, da, kb, dacc);
                    wmma::mma_sync(dacc, a, dkb, dacc);
                }
#pragma unroll
                for (int e2 = 0; e2 < 8; ++e2) {
                    int rr, cc; frag_rc(lane, e2, rr, cc);
                    const int i = im + rr, j = in + cc;
                    float wq = 0.f, dwq = 0.f;
                    if (i >= j) {
                        const float wgt = expf(w.L_s[i] - w.L_s[j]);
                        wq = wgt * acc.x[e2];
                        dwq = wgt * (dLL[ll][i] - dLL[ll][j]) * acc.x[e2]
                            + wgt * dacc.x[e2];
                    }
                    if (ll == 0) w.WQKb[i * CS2 + j] = __float2bfloat16(wq);
                    dWL[ll][i * CS2 + j] = __float2bfloat16(dwq);
                }
            }
        }
        __syncthreads();

        // e-fold: qr once, both dqr lanes (reads raw qr before its write)
        for (int idx = tid; idx < cs * N; idx += nthr) {
            const int i = idx / N, n = idx % N;
            const float e = w.e_s[i];
            const float qv = ld(w.qr + i * NS + n);
            for (int ll = 0; ll < 2; ++ll) {
                const float dqv = ld(dqrL[ll] + i * NS + n);
                dqrL[ll][i * NS + n] =
                    __float2bfloat16(e * dqv + e * dLL[ll][i] * qv);
            }
            w.qr[i * NS + n] = __float2bfloat16(e * qv);
        }
        __syncthreads();

        // output contraction + sink over 2*mt*nt virtual tiles
        {
            const int nt = P / 16, mt = cs / 16;
            for (int t = warp; t < 2 * mt * nt; t += nwarps) {
                const int ll = t / (mt * nt), tt = t % (mt * nt);
                const int im = (tt / nt) * 16, in = (tt % nt) * 16;
                AccFrag acc, dacc;
                wmma::fill_fragment(acc, 0.f);
                wmma::fill_fragment(dacc, 0.f);
                for (int k = 0; k < N / 16; ++k) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a, da;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> sb, dsb;
                    wmma::load_matrix_sync(a, w.qr + im * NS + k * 16, NS);
                    wmma::load_matrix_sync(da, dqrL[ll] + im * NS + k * 16, NS);
                    wmma::load_matrix_sync(sb, w.S + (k * 16) * PS + in, PS);
                    wmma::load_matrix_sync(dsb, dSL[ll] + (k * 16) * PS + in, PS);
                    wmma::mma_sync(acc, a, sb, acc);
                    wmma::mma_sync(dacc, da, sb, dacc);
                    wmma::mma_sync(dacc, a, dsb, dacc);
                }
                for (int k = 0; k <= im / 16; ++k) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a, da;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> vb, dvb;
                    wmma::load_matrix_sync(a, w.WQKb + im * CS2 + k * 16, CS2);
                    wmma::load_matrix_sync(da, dWL[ll] + im * CS2 + k * 16, CS2);
                    wmma::load_matrix_sync(vb, w.v + (k * 16) * PS + in, PS);
                    wmma::load_matrix_sync(dvb, dvL[ll] + (k * 16) * PS + in, PS);
                    wmma::mma_sync(acc, a, vb, acc);
                    wmma::mma_sync(dacc, da, vb, dacc);
                    wmma::mma_sync(dacc, a, dvb, dacc);
                }
                float pq[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                for (int e2 = 0; e2 < 8; ++e2) {
                    int rr, cc; frag_rc(lane, e2, rr, cc);
                    const int i = im + rr, q = in + cc;
                    const float vv = ld(w.v + i * PS + q), dvv = ld(dvL[ll] + i * PS + q);
                    const float o = acc.x[e2] + dskip * vv - vv * w.qk_s[i];
                    const float dO = dacc.x[e2] + dskip * dvv
                        - (dvv * w.qk_s[i] + vv * dqkL[ll][i]);
                    const float z = ld(w.z + i * P + q);
                    const float dz = ld(dzL[ll] + i * P + q);
                    const float sg = 1.f / (1.f + expf(-z));
                    const float gate = z * sg;
                    const float dgate = sg * (1.f + z * (1.f - sg)) * dz;
                    if (s0 + i < p.s_true)
                        pq[(e2 & 1) | ((e2 & 4) >> 1)] += dO * gate + o * dgate;
                }
#pragma unroll
                for (int jj = 0; jj < 4; ++jj)
                    atomicAdd(&partL[ll][in + (lane & 3) * 2 + (jj & 1)
                                         + ((jj & 2) ? 8 : 0)], pq[jj]);
            }
        }
        __syncthreads();

        // A_w per lane + shared B_w (reads raw ksc before its write)
        for (int idx = tid; idx < cs * N; idx += nthr) {
            const int j = idx / N, n = idx % N;
            const float kv = ld(w.ksc + j * NS + n);
            for (int ll = 0; ll < 2; ++ll) {
                const float dkv = ld(dkscL[ll] + j * NS + n);
                dkscL[ll][j * NS + n] =
                    __float2bfloat16(aL[ll][j] * kv + w.wl_s[j] * dkv);
            }
            w.ksc[j * NS + n] = __float2bfloat16(w.wl_s[j] * kv);
        }
        __syncthreads();
        {
            const float wc = w.e_s[cs - 1];
            const float dcd1 = wc * w.dL_s[cs - 1];
            const float dcd2 = wc * x.dL2[cs - 1];
            const int nt = P / 16, mt = N / 16;
            for (int t = warp; t < mt * nt; t += nwarps) {
                const int im = (t / nt) * 16, in = (t % nt) * 16;
                AccFrag accp, acct1, acct2;
                wmma::fill_fragment(accp, 0.f);
                wmma::fill_fragment(acct1, 0.f);
                wmma::fill_fragment(acct2, 0.f);
                for (int k = 0; k < cs / 16; ++k) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::col_major> aB, aA1, aA2;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> vb, dvb1, dvb2;
                    wmma::load_matrix_sync(aB, w.ksc + (k * 16) * NS + im, NS);
                    wmma::load_matrix_sync(aA1, w.dksc + (k * 16) * NS + im, NS);
                    wmma::load_matrix_sync(aA2, x.dksc2 + (k * 16) * NS + im, NS);
                    wmma::load_matrix_sync(vb, w.v + (k * 16) * PS + in, PS);
                    wmma::load_matrix_sync(dvb1, w.dv + (k * 16) * PS + in, PS);
                    wmma::load_matrix_sync(dvb2, x.dv2 + (k * 16) * PS + in, PS);
                    wmma::mma_sync(accp, aB, vb, accp);
                    wmma::mma_sync(acct1, aA1, vb, acct1);
                    wmma::mma_sync(acct1, aB, dvb1, acct1);
                    wmma::mma_sync(acct2, aA2, vb, acct2);
                    wmma::mma_sync(acct2, aB, dvb2, acct2);
                }
#pragma unroll
                for (int e2 = 0; e2 < 8; ++e2) {
                    int rr, cc; frag_rc(lane, e2, rr, cc);
                    const int n_ = im + rr, q = in + cc;
                    const float s_old = ld(w.S + n_ * PS + q);
                    const float d1 = ld(w.dS + n_ * PS + q);
                    const float d2 = ld(x.dS2 + n_ * PS + q);
                    w.S[n_ * PS + q] = __float2bfloat16(wc * s_old + accp.x[e2]);
                    w.dS[n_ * PS + q] =
                        __float2bfloat16(wc * d1 + dcd1 * s_old + acct1.x[e2]);
                    x.dS2[n_ * PS + q] =
                        __float2bfloat16(wc * d2 + dcd2 * s_old + acct2.x[e2]);
                }
            }
        }
        __syncthreads();
        for (int q = tid; q < P; q += nthr) {
            p.PART[((((long)l0 * p.batch + b) * H + h) * nc + c) * P + q] = w.part_s[q];
            p.PART[((((long)(l0 + 1) * p.batch + b) * H + h) * nc + c) * P + q] = x.part2[q];
        }
        __syncthreads();
    }
}

}  // namespace

void launch_fwd_r8a_mean(const FwdDualscanParams& p, cudaStream_t stream) {
    const size_t smem = fused_arena_bytes(p.chunk_size, p.d_state, p.headdim, 8);
    void (*kern)(FwdDualscanParams) = fwd_r8a_mean_kernel<0>;
    if (getenv("LBI_R8A_NOCHAIN")) kern = fwd_r8a_mean_kernel<1>;
    cudaFuncSetAttribute(kern,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    if (getenv("LBI_CUDA_DEBUG")) {
        int nb = -1;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, kern, 256, smem);
        printf("r8a_mean: %zu B arena, %d CTAs/SM\n", smem, nb);
    }
    const int nc = p.seqlen / p.chunk_size;
    dim3 grid(p.heads, p.lanes * p.batch, nc);
    kern<<<grid, 256, smem, stream>>>(p);
}

void launch_fwd_fused_mean(const FwdDualscanParams& p, cudaStream_t stream) {
    const size_t smem = fused_arena_bytes(p.chunk_size, p.d_state, p.headdim, 8);
    const char* ab = getenv("LBI_CUDA_ABLATE");
    const int abl = ab ? atoi(ab) : 0;
    void (*kern)(FwdDualscanParams) = fwd_fused_mean_kernel<0>;
    switch (abl) {
        case 1: kern = fwd_fused_mean_kernel<1>; break;
        case 2: kern = fwd_fused_mean_kernel<2>; break;
        case 3: kern = fwd_fused_mean_kernel<3>; break;
        case 4: kern = fwd_fused_mean_kernel<4>; break;
    }
    if (getenv("LBI_CUDA_LEAN") && abl == 0)
        kern = fwd_fused_mean_kernel<0, 1>;   // lean walk (e-fold deleted)
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem);
    if (getenv("LBI_CUDA_DEBUG")) {
        int nb = -1;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, kern, 256, smem);
        printf("fused_mean: %zu B arena, %d CTAs/SM, ablate %d\n", smem, nb, abl);
    }
    const size_t smem2 = smem + fused2_extra_bytes(p.chunk_size,
                                                   p.d_state, p.headdim);
    if (getenv("LBI_CUDA_RB2") && p.lanes % 2 == 0 && abl == 0
        && p.native == 0  // rb2 stays bf16-emitted
        && smem2 <= 227 * 1024) {
        cudaFuncSetAttribute(fwd_fused_mean2_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem2);
        if (getenv("LBI_CUDA_DEBUG")) {
            int nb = -1;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &nb, fwd_fused_mean2_kernel, 256, smem2);
            printf("fused_mean rb2: %zu B arena, %d CTAs/SM\n", smem2, nb);
        }
        dim3 grid2(p.lanes / 2, p.heads, p.batch);
        fwd_fused_mean2_kernel<<<grid2, 256, smem2, stream>>>(p);
        return;
    }
    dim3 grid(p.lanes, p.heads, p.batch);
    kern<<<grid, 256, smem, stream>>>(p);
}

void launch_fwd_passA_tan_opt(const FwdDualscanParams& p, cudaStream_t stream) {
    const int cs = p.chunk_size, N = p.d_state, P = p.headdim;
    const int nc = p.seqlen / cs;
    const int NS = N + SKEW, PS = P + SKEW;
    const size_t smem = (size_t)cs * NS * 2 * 2 + (size_t)cs * PS * 2 * 2
                      + (size_t)cs * 4 * 4 + (size_t)8 * 256 * 4;
    cudaFuncSetAttribute(fwd_passA_tan_opt_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    dim3 grid(nc * p.lanes, p.heads, p.batch);
    fwd_passA_tan_opt_kernel<<<grid, 256, smem, stream>>>(p);
}

void launch_fwd_passA_primal_opt(const FwdDualscanParams& p, cudaStream_t stream) {
    const int cs = p.chunk_size, N = p.d_state, P = p.headdim;
    const int nc = p.seqlen / cs;
    const int NS = N + SKEW, PS = P + SKEW;
    const size_t smem = (size_t)cs * NS * 2 + (size_t)cs * PS * 2
                      + (size_t)cs * 4 + (size_t)8 * 256 * 4;
    cudaFuncSetAttribute(fwd_passA_primal_opt_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    dim3 grid(nc, p.heads, p.batch);
    fwd_passA_primal_opt_kernel<<<grid, 256, smem, stream>>>(p);
}

void launch_fwd_passB_opt(const FwdDualscanParams& p, cudaStream_t stream) {
    const int cs = p.chunk_size;
    const int nc = p.seqlen / cs;
    const long NP = (long)p.d_state * p.headdim;
    const size_t smem = (size_t)nc * nc * 4 + (size_t)nc * 4
                      + (size_t)p.lanes * nc * 4
                      + (size_t)nc * COLB * 4
                      + (size_t)3 * nc * COLB * 2;
    cudaFuncSetAttribute(fwd_passB_opt_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    dim3 grid((unsigned)(NP / COLB), p.heads, p.batch);
    fwd_passB_opt_kernel<<<grid, 256, smem, stream>>>(p);
}

void launch_fwd_passC_opt(const FwdDualscanParams& p, cudaStream_t stream) {
    size_t smem = passC_arena_bytes(p.chunk_size, p.d_state, p.headdim);
    if (const char* pad = getenv("LBI_CUDA_SMEM_PAD")) smem += atol(pad);
    const char* ab = getenv("LBI_CUDA_ABLATE");
    const int abl = ab ? atoi(ab) : 0;
    void (*kern)(FwdDualscanParams) = fwd_passC_opt_kernel<0>;
    switch (abl) {
        case 1: kern = fwd_passC_opt_kernel<1>; break;
        case 2: kern = fwd_passC_opt_kernel<2>; break;
        case 3: kern = fwd_passC_opt_kernel<3>; break;
        case 4: kern = fwd_passC_opt_kernel<4>; break;
        case 5: kern = fwd_passC_opt_kernel<5>; break;
        case 6: kern = fwd_passC_opt_kernel<6>; break;
    }
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem);
    if (getenv("LBI_CUDA_DEBUG")) {
        int nb = -1;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, kern, 256, smem);
        printf("passC_opt: %zu B arena, %d CTAs/SM, ablate %d\n", smem, nb, abl);
    }
    dim3 grid((p.seqlen / p.chunk_size) * p.lanes, p.heads, p.batch);
    kern<<<grid, 256, smem, stream>>>(p);
}

void launch_fwd_passC_mean_opt(const FwdDualscanParams& p, cudaStream_t stream) {
    const size_t smem = passC_arena_bytes(p.chunk_size, p.d_state, p.headdim);
    cudaFuncSetAttribute(fwd_passC_mean_opt_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    dim3 grid((p.seqlen / p.chunk_size) * p.lanes, p.heads, p.batch);
    fwd_passC_mean_opt_kernel<<<grid, 256, smem, stream>>>(p);
}

}  // namespace lbi_mamba3
