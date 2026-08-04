// Pass C, wmma rewrite: same semantics as chunkparallel_pass_c_simple.cu but
// the eight GEMMs run on tensor cores (16x16x16 bf16 tiles, fp32 accumulate).

// Operands in bf16 smem, results in fp32 smem, scaled/masked in place between
// GEMMs; 128 threads = 4 warps, output tiles round-robin. cs, N, PD % 16 == 0.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>

#include "chunkparallel_pass_c.hpp"

namespace lbi_mamba3 {
namespace {

using bf16 = __nv_bfloat16;
namespace wmma = nvcuda::wmma;
constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;

__device__ inline float ld(const bf16* p) { return __bfloat162float(*p); }

// C[M,N] fp32 = A @ B^T, A stored [M,K] row-major, B stored [N,K] row-major.
// accumulate: if true, C += ...  (reads existing C as the accumulator init).
__device__ void gemm_ABt(float* C, const bf16* A, const bf16* B,
                         int M, int N, int K, int warp, int nwarps, bool accumulate) {
    const int mt = M / WMMA_M, nt = N / WMMA_N, kt = K / WMMA_K;
    for (int t = warp; t < mt * nt; t += nwarps) {
        const int im = (t / nt) * WMMA_M, in = (t % nt) * WMMA_N;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
        if (accumulate) wmma::load_matrix_sync(acc, C + im * N + in, N, wmma::mem_row_major);
        else wmma::fill_fragment(acc, 0.f);
        for (int k = 0; k < kt; ++k) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, bf16, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, bf16, wmma::col_major> b;
            wmma::load_matrix_sync(a, A + im * K + k * WMMA_K, K);
            wmma::load_matrix_sync(b, B + in * K + k * WMMA_K, K);
            wmma::mma_sync(acc, a, b, acc);
        }
        wmma::store_matrix_sync(C + im * N + in, acc, N, wmma::mem_row_major);
    }
}

// C[M,N] fp32 = A @ B, A stored [M,K] row-major, B stored [K,N] row-major.
__device__ void gemm_AB(float* C, const bf16* A, const bf16* B,
                        int M, int N, int K, int warp, int nwarps, bool accumulate) {
    const int mt = M / WMMA_M, nt = N / WMMA_N, kt = K / WMMA_K;
    for (int t = warp; t < mt * nt; t += nwarps) {
        const int im = (t / nt) * WMMA_M, in = (t % nt) * WMMA_N;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
        if (accumulate) wmma::load_matrix_sync(acc, C + im * N + in, N, wmma::mem_row_major);
        else wmma::fill_fragment(acc, 0.f);
        for (int k = 0; k < kt; ++k) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, bf16, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, bf16, wmma::row_major> b;
            wmma::load_matrix_sync(a, A + im * K + k * WMMA_K, K);
            wmma::load_matrix_sync(b, B + (k * WMMA_K) * N + in, N);
            wmma::mma_sync(acc, a, b, acc);
        }
        wmma::store_matrix_sync(C + im * N + in, acc, N, wmma::mem_row_major);
    }
}

// C[M,N] fp32 = A^T @ B, A stored [K,M] row-major, B stored [K,N] row-major.
__device__ void gemm_AtB(float* C, const bf16* A, const bf16* B,
                         int M, int N, int K, int warp, int nwarps, bool accumulate) {
    const int mt = M / WMMA_M, nt = N / WMMA_N, kt = K / WMMA_K;
    for (int t = warp; t < mt * nt; t += nwarps) {
        const int im = (t / nt) * WMMA_M, in = (t % nt) * WMMA_N;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
        if (accumulate) wmma::load_matrix_sync(acc, C + im * N + in, N, wmma::mem_row_major);
        else wmma::fill_fragment(acc, 0.f);
        for (int k = 0; k < kt; ++k) {
            // A^T[m,kk] = A[kk,m]; col_major read with ldm=M yields the transpose.
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, bf16, wmma::col_major> a;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, bf16, wmma::row_major> b;
            wmma::load_matrix_sync(a, A + (k * WMMA_K) * M + im, M);
            wmma::load_matrix_sync(b, B + (k * WMMA_K) * N + in, N);
            wmma::mma_sync(acc, a, b, acc);
        }
        wmma::store_matrix_sync(C + im * N + in, acc, N, wmma::mem_row_major);
    }
}

__global__ void chunkparallel_pass_c_mma_kernel(ChunkParallelPassCParams p) {
    const int h = blockIdx.x, b = blockIdx.y;
    const int lane = blockIdx.z / p.nchunks, c = blockIdx.z % p.nchunks;
    const int cs = p.chunk_size, N = p.d_state, PD = p.headdim, NR = p.n_rot;
    const int S = p.seqlen, H = p.heads, Gq = p.groups, h_qk = h / (H / Gq);
    const int s0 = c * cs, tid = threadIdx.x, nthr = blockDim.x;
    const int warp = tid / 32, nwarps = nthr / 32;

    extern __shared__ char smem_raw[];
    // bf16 operands
    bf16* q_pre = reinterpret_cast<bf16*>(smem_raw);   // [cs][N]
    bf16* k_pre = q_pre + cs * N;                      // [cs][N]
    bf16* q_rot = k_pre + cs * N;                      // [cs][N]
    bf16* k_sc = q_rot + cs * N;                       // [cs][N]
    bf16* v_s = k_sc + cs * N;                         // [cs][PD]
    bf16* dout_s = v_s + cs * PD;                      // [cs][PD] gated
    // in_s and st_s share one [N][PD] buffer (in_s dead after the dk GEMM,
    // st loaded lazily before the dq GEMM); further smem cuts trade smem
    // for gmem traffic and lose.
    bf16* in_s = dout_s + cs * PD;                     // [N][PD]  (aka st_s)
    bf16* st_s = in_s;
    bf16* lkq_m = in_s + N * PD;                       // [cs][cs]
    bf16* dkq_m = lkq_m + cs * cs;                     // [cs][cs]
    // fp32 working buffers (16-byte align after the bf16 region)
    float* fbuf = reinterpret_cast<float*>(dkq_m + cs * cs + (cs * cs & 1));
    float* c_cc = fbuf;                                // [cs][cs] gemm result
    float* c_cp = c_cc + cs * cs;                      // [cs][PD]
    float* dk_f = c_cp + cs * PD;                      // [cs][N]
    float* dq_f = dk_f + cs * N;                       // [cs][N]
    float* lkq_u = dq_f + cs * N;                      // [cs][cs] unmasked
    float* ang_s = lkq_u + cs * cs;                    // [cs][NR]
    float* sc = ang_s + cs * NR;                       // 6 scalar rows [cs] each
    float* sc_gamma = sc; float* sc_tscale = sc + cs; float* sc_exprev = sc + 2 * cs;
    float* sc_expcs = sc + 3 * cs; float* sc_dqg = sc + 4 * cs;

    // ---- scalars + angles ----
    for (int i = tid; i < cs; i += nthr) {
        const int s = s0 + i;
        const float dt = p.DT[(b * H + h) * S + s];
        const float trap = ld(p.TRAP + (b * H + h) * S + s);
        const float gamma = dt / (1.f + expf(-trap));
        float shg = 0.f;
        if (s + 1 < S && s < S - 1) {
            const float dt1 = p.DT[(b * H + h) * S + s + 1];
            const float tr1 = ld(p.TRAP + (b * H + h) * S + s + 1);
            shg = dt1 / (1.f + expf(tr1));
        }
        sc_gamma[i] = gamma;
        sc_tscale[i] = gamma + shg;
        sc_exprev[i] = expf(p.DA_CS_REV[(b * H + h) * S + s]);
        sc_expcs[i] = expf(p.DA_CS[(b * H + h) * S + s]);
    }
    for (int i = tid; i < cs * NR; i += nthr)
        ang_s[i] = p.ANGLES[((b * S + (s0 + i / NR)) * H + h) * NR + (i % NR)];
    __syncthreads();

    // ---- q/k prep: bias, rotary, trap-scale on k ----
    for (int i = tid; i < cs * N; i += nthr) {
        const int r = i / N, n = i % N;
        const long qk = (((long)(b * S + (s0 + r)) * Gq) + h_qk) * N + n;
        q_pre[i] = __float2bfloat16(ld(p.Q + qk) + p.Q_BIAS[h * N + n]);
        k_pre[i] = __float2bfloat16(ld(p.K + qk) + p.K_BIAS[h * N + n]);
    }
    __syncthreads();
    for (int i = tid; i < cs * N; i += nthr) {
        const int r = i / N, n = i % N;
        float qv = ld(q_pre + i), kv = ld(k_pre + i);
        if (n < NR) {
            const float ca = cosf(ang_s[r * NR + n]), sa = sinf(ang_s[r * NR + n]);
            qv = ca * ld(q_pre + r * N + n) - sa * ld(q_pre + r * N + N / 2 + n);
            kv = ca * ld(k_pre + r * N + n) - sa * ld(k_pre + r * N + N / 2 + n);
        } else if (n >= N / 2 && n < N / 2 + NR) {
            const int nn = n - N / 2;
            const float ca = cosf(ang_s[r * NR + nn]), sa = sinf(ang_s[r * NR + nn]);
            qv = sa * ld(q_pre + r * N + nn) + ca * ld(q_pre + r * N + N / 2 + nn);
            kv = sa * ld(k_pre + r * N + nn) + ca * ld(k_pre + r * N + N / 2 + nn);
        }
        q_rot[i] = __float2bfloat16(qv);
        k_sc[i] = __float2bfloat16(kv * sc_tscale[r]);
    }
    for (int i = tid; i < cs * PD; i += nthr) {
        const int r = i / PD, pd = i % PD;
        const long vo = ((long)(b * S + (s0 + r)) * H + h) * PD + pd;
        v_s[i] = __float2bfloat16(ld(p.V + vo));
        dout_s[i] = __float2bfloat16(
            ld(p.DOUT + (((long)((b * p.lanes + lane) * S + (s0 + r)) * H + h) * PD + pd))
            * ld(p.GATE + vo));
    }
    for (int i = tid; i < N * PD; i += nthr)
        in_s[i] = p.DSTATES_IN[((((long)(b * p.lanes + lane) * H + h) * p.nchunks + c) * N * PD) + i];
    __syncthreads();

    // ---- lkq = k_sc @ q_rot^T; unmasked (fp32) + masked (bf16) ----
    gemm_ABt(c_cc, k_sc, q_rot, cs, cs, N, warp, nwarps, false);
    __syncthreads();
    for (int i = tid; i < cs * cs; i += nthr) {
        const int r = i / cs, j = i % cs;
        lkq_u[i] = c_cc[i];
        const float es = expf(p.SEGSUM[((((long)(b * H + h) * p.nchunks + c) * cs + j) * cs + r)]);
        lkq_m[i] = __float2bfloat16((r < j) ? c_cc[i] * es : 0.f);
    }
    __syncthreads();

    // ---- dPsiV = (k_sc @ in)*exprev + lkq_m @ dout; dv ----
    gemm_AB(c_cp, k_sc, in_s, cs, PD, N, warp, nwarps, false);
    __syncthreads();
    for (int i = tid; i < cs * PD; i += nthr) c_cp[i] *= sc_exprev[i / PD];
    __syncthreads();
    gemm_AB(c_cp, lkq_m, dout_s, cs, PD, cs, warp, nwarps, true);
    __syncthreads();
    for (int i = tid; i < cs * PD; i += nthr) {
        const int r = i / PD;
        const float qkd = ld(p.QK_DOT + ((long)(b * H + h) * S + (s0 + r)));
        const float acc = c_cp[i] + ld(dout_s + i) * p.D[h] + ld(dout_s + i) * qkd * sc_gamma[r];
        p.DV[(((long)((b * p.lanes + lane) * S + (s0 + r)) * H + h) * PD + (i % PD))] =
            __float2bfloat16(acc);
    }
    __syncthreads();

    // ---- diag reductions + dD ----
    if (tid < cs) {
        const int r = tid;
        float dot = 0.f;
        for (int pd = 0; pd < PD; ++pd) dot += ld(dout_s + r * PD + pd) * ld(v_s + r * PD + pd);
        const float qkd = ld(p.QK_DOT + ((long)(b * H + h) * S + (s0 + r)));
        p.DGAMMA_DIAG[(((long)(b * p.lanes + lane) * H + h) * S + (s0 + r))] = qkd * dot;
        sc_dqg[r] = dot * sc_gamma[r];
    }
    if (tid == 0) {
        float acc = 0.f;
        for (int i = 0; i < cs * PD; ++i) acc += ld(dout_s + i) * ld(v_s + i);
        p.DD[(((long)(b * p.lanes + lane) * H + h) * p.nchunks + c)] = acc;
    }
    __syncthreads();

    // ---- dk: v @ in^T (dk_f); ddacsrev; dkq (c_cc) + DSSDA + masked ----
    gemm_ABt(dk_f, v_s, in_s, cs, N, PD, warp, nwarps, false);
    gemm_ABt(c_cc, v_s, dout_s, cs, cs, PD, warp, nwarps, false);
    __syncthreads();
    if (tid < cs) {
        const int r = tid;
        float acc = 0.f;
        for (int n = 0; n < N; ++n) acc += ld(k_sc + r * N + n) * dk_f[r * N + n];
        p.DDA_CS_REV[(((long)(b * p.lanes + lane) * H + h) * S + (s0 + r))] = acc;
    }
    for (int i = tid; i < cs * cs; i += nthr) {
        const int r = i / cs, j = i % cs;
        p.DSSDA[(((((long)(b * p.lanes + lane) * H + h) * p.nchunks + c) * cs + r) * cs + j)] =
            lkq_u[i] * c_cc[i];
        const float es = expf(p.SEGSUM[((((long)(b * H + h) * p.nchunks + c) * cs + j) * cs + r)]);
        dkq_m[i] = __float2bfloat16((r < j) ? c_cc[i] * es : 0.f);
    }
    __syncthreads();
    // dk_f = dk_f*exprev + dkq_m @ q_rot
    for (int i = tid; i < cs * N; i += nthr) dk_f[i] *= sc_exprev[i / N];
    __syncthreads();
    gemm_AB(dk_f, dkq_m, q_rot, cs, N, cs, warp, nwarps, true);
    __syncthreads();
    if (tid < cs) {
        const int r = tid;
        float acc = 0.f;
        for (int n = 0; n < N; ++n) acc += ld(k_sc + r * N + n) * dk_f[r * N + n];
        p.DFACTOR[(((long)(b * p.lanes + lane) * H + h) * S + (s0 + r))] = acc / sc_tscale[r];
    }
    __syncthreads();
    for (int i = tid; i < cs * N; i += nthr) dk_f[i] *= sc_tscale[i / N];
    __syncthreads();

    // ---- dq: dout @ st^T (dq_f); ddacs; scale; += dkq_m^T @ k_sc ----
    // lazy STATES load into the in_s buffer (in_s dead after the dk GEMM).
    for (int i = tid; i < N * PD; i += nthr)
        st_s[i] = p.STATES[((((long)(b * H + h) * p.nchunks + c) * N * PD) + i)];
    __syncthreads();
    gemm_ABt(dq_f, dout_s, st_s, cs, N, PD, warp, nwarps, false);
    __syncthreads();
    if (tid < cs) {
        const int r = tid;
        float acc = 0.f;
        for (int n = 0; n < N; ++n) acc += ld(q_rot + r * N + n) * dq_f[r * N + n];
        p.DDA_CS[(((long)(b * p.lanes + lane) * H + h) * S + (s0 + r))] = acc;
    }
    __syncthreads();
    for (int i = tid; i < cs * N; i += nthr) dq_f[i] *= sc_expcs[i / N];
    __syncthreads();
    gemm_AtB(dq_f, dkq_m, k_sc, cs, N, cs, warp, nwarps, true);
    __syncthreads();

    // ---- inverse rotary + dangles + diag corrections + atomic head-sums ----
    for (int i = tid; i < cs * NR; i += nthr) {
        const int r = i / NR, n = i % NR;
        const float ca = cosf(ang_s[i]), sa = sinf(ang_s[i]);
        const float dk1 = dk_f[r * N + n], dk2 = dk_f[r * N + N / 2 + n];
        const float dq1 = dq_f[r * N + n], dq2 = dq_f[r * N + N / 2 + n];
        const float kp1 = ld(k_pre + r * N + n), kp2 = ld(k_pre + r * N + N / 2 + n);
        const float qp1 = ld(q_pre + r * N + n), qp2 = ld(q_pre + r * N + N / 2 + n);
        float da = dk1 * (-kp1 * sa - kp2 * ca) + dk2 * (kp1 * ca - kp2 * sa);
        da += dq1 * (-qp1 * sa - qp2 * ca) + dq2 * (qp1 * ca - qp2 * sa);
        p.DANGLES[(((long)((b * p.lanes + lane) * S + (s0 + r)) * H + h) * NR + n)] = da;
    }
    __syncthreads();
    for (int i = tid; i < cs * N; i += nthr) {
        const int r = i / N, n = i % N;
        float dkv = dk_f[i], dqv = dq_f[i];
        if (n < NR) {
            const float ca = cosf(ang_s[r * NR + n]), sa = sinf(ang_s[r * NR + n]);
            dkv = ca * dk_f[r * N + n] + sa * dk_f[r * N + N / 2 + n];
            dqv = ca * dq_f[r * N + n] + sa * dq_f[r * N + N / 2 + n];
        } else if (n >= N / 2 && n < N / 2 + NR) {
            const int nn = n - N / 2;
            const float ca = cosf(ang_s[r * NR + nn]), sa = sinf(ang_s[r * NR + nn]);
            dkv = -sa * dk_f[r * N + nn] + ca * dk_f[r * N + N / 2 + nn];
            dqv = -sa * dq_f[r * N + nn] + ca * dq_f[r * N + N / 2 + nn];
        }
        dkv += sc_dqg[r] * ld(q_pre + i);
        dqv += sc_dqg[r] * ld(k_pre + i);
        const long o = ((long)((b * p.lanes + lane) * S + (s0 + r))) * N + n;
        atomicAdd(p.DK + o, dkv);
        atomicAdd(p.DQ + o, dqv);
    }
}

}  // namespace

void launch_chunkparallel_pass_c_mma(const ChunkParallelPassCParams& params,
                                     cudaStream_t stream) {
    const int cs = params.chunk_size, N = params.d_state, PD = params.headdim, NR = params.n_rot;
    // in_s/st_s aliased -> one N*PD block (Step 1).
    const size_t bf16_elems = (size_t)cs * N * 4 + (size_t)cs * PD * 2 + (size_t)N * PD * 1 +
                              (size_t)cs * cs * 2 + (cs * cs & 1);
    const size_t f32_elems = (size_t)cs * cs + (size_t)cs * PD + (size_t)cs * N * 2 +
                             (size_t)cs * cs + (size_t)cs * NR + 6 * (size_t)cs;
    const size_t smem = bf16_elems * sizeof(bf16) + f32_elems * sizeof(float);
    cudaFuncSetAttribute(chunkparallel_pass_c_mma_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    if (getenv("LBI_MMA_OCC")) {
        int blocks = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks, chunkparallel_pass_c_mma_kernel, 128, smem);
        fprintf(stderr, "[mma] dyn_smem=%zu B  CTAs/SM=%d\n", smem, blocks);
    }
    dim3 grid(params.heads, params.batch, params.lanes * params.nchunks);
    chunkparallel_pass_c_mma_kernel<<<grid, 128, smem, stream>>>(params);
}

}  // namespace lbi_mamba3
