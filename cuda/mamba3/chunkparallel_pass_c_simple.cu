// Pass C of the chunk-parallel P-batched SISO backward, correctness scaffold
// (parity oracle: tilelang mamba_chunkparallel_bwd_bwd; _mma is the fast one).

// One CTA per (head, batch, lane, chunk); IN_c from pass B; Z-gate fused on
// the DOUT load; DK/DQ head-summed via fp32 atomics. All dims runtime.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "chunkparallel_pass_c.hpp"

namespace lbi_mamba3 {

namespace {

using bf16 = __nv_bfloat16;

__device__ inline float ld(const bf16* p) { return __bfloat162float(*p); }

__global__ void chunkparallel_pass_c_simple_kernel(ChunkParallelPassCParams p) {
    const int h = blockIdx.x;
    const int b = blockIdx.y;
    const int lane = blockIdx.z / p.nchunks;
    const int c = blockIdx.z % p.nchunks;
    const int cs = p.chunk_size;
    const int N = p.d_state;
    const int PD = p.headdim;
    const int NR = p.n_rot;
    const int S = p.seqlen;
    const int H = p.heads;
    const int Gq = p.groups;
    const int h_qk = h / (H / Gq);
    const int s0 = c * cs;
    const int tid = threadIdx.x;
    const int nthr = blockDim.x;

    extern __shared__ float sm[];
    // fp32 smem layout (correctness first; the _mma rewrite packs bf16)
    float* q_pre = sm;                       // [cs][N]  biased, unrotated
    float* q_rot = q_pre + cs * N;           // [cs][N]
    float* k_pre = q_rot + cs * N;           // [cs][N]
    float* k_sc = k_pre + cs * N;            // [cs][N]  rotated + trap-scaled
    float* v_s = k_sc + cs * N;              // [cs][PD]
    float* dout_s = v_s + cs * PD;           // [cs][PD] gated
    float* in_s = dout_s + cs * PD;          // [N][PD]  IN_c
    float* st_s = in_s + N * PD;             // [N][PD]  forward states
    float* lkq_u = st_s + N * PD;            // [cs][cs]
    float* lkq_m = lkq_u + cs * cs;          // [cs][cs]
    float* dkq_m = lkq_m + cs * cs;          // [cs][cs]
    float* dk_t = dkq_m + cs * cs;           // [cs][N]  dk workspace
    float* dq_t = dk_t + cs * N;             // [cs][N]  dq workspace
    float* ang_s = dq_t + cs * N;            // [cs][NR] cumulative angles
    float* sc_gamma = ang_s + cs * NR;       // [cs] x5 scalar rows
    float* sc_tscale = sc_gamma + cs;
    float* sc_exprev = sc_tscale + cs;
    float* sc_expcs = sc_exprev + cs;
    float* sc_dqg = sc_expcs + cs;

    // ---- scalars ----
    for (int i = tid; i < cs; i += nthr) {
        const int s = s0 + i;
        const float dt = p.DT[(b * H + h) * S + s];
        const float trap = ld(p.TRAP + (b * H + h) * S + s);
        const float gamma = dt / (1.f + expf(-trap));
        float shg = 0.f;
        if (s + 1 < S) {
            const float dt1 = p.DT[(b * H + h) * S + s + 1];
            const float tr1 = ld(p.TRAP + (b * H + h) * S + s + 1);
            if (s < S - 1) shg = dt1 / (1.f + expf(tr1));
        }
        sc_gamma[i] = gamma;
        sc_tscale[i] = gamma + shg;
        sc_exprev[i] = expf(p.DA_CS_REV[(b * H + h) * S + s]);
        sc_expcs[i] = expf(p.DA_CS[(b * H + h) * S + s]);
    }
    for (int i = tid; i < cs * NR; i += nthr) {
        const int r = i / NR, n = i % NR;
        ang_s[r * NR + n] = p.ANGLES[((b * S + (s0 + r)) * H + h) * NR + n];
    }
    __syncthreads();

    // ---- q/k prep: bias + rotary (+ trap scale on k) ----
    for (int i = tid; i < cs * N; i += nthr) {
        const int r = i / N, n = i % N;
        const long qk_off = (((long)(b * S + (s0 + r)) * Gq) + h_qk) * N + n;
        q_pre[i] = ld(p.Q + qk_off) + p.Q_BIAS[h * N + n];
        k_pre[i] = ld(p.K + qk_off) + p.K_BIAS[h * N + n];
    }
    __syncthreads();
    for (int i = tid; i < cs * N; i += nthr) {
        const int r = i / N, n = i % N;
        float qv = q_pre[i], kv = k_pre[i];
        if (n < NR) {
            const float ca = cosf(ang_s[r * NR + n]), sa = sinf(ang_s[r * NR + n]);
            qv = ca * q_pre[r * N + n] - sa * q_pre[r * N + N / 2 + n];
            kv = ca * k_pre[r * N + n] - sa * k_pre[r * N + N / 2 + n];
        } else if (n >= N / 2 && n < N / 2 + NR) {
            const int nn = n - N / 2;
            const float ca = cosf(ang_s[r * NR + nn]), sa = sinf(ang_s[r * NR + nn]);
            qv = sa * q_pre[r * N + nn] + ca * q_pre[r * N + N / 2 + nn];
            kv = sa * k_pre[r * N + nn] + ca * k_pre[r * N + N / 2 + nn];
        }
        q_rot[i] = qv;
        k_sc[i] = kv * sc_tscale[r];
    }
    // v / gated dout / IN / states loads
    for (int i = tid; i < cs * PD; i += nthr) {
        const int r = i / PD, pd = i % PD;
        const long vo = ((long)(b * S + (s0 + r)) * H + h) * PD + pd;
        v_s[i] = ld(p.V + vo);
        dout_s[i] = ld(p.DOUT + (((long)((b * p.lanes + lane) * S + (s0 + r)) * H + h) * PD + pd))
                    * ld(p.GATE + vo);
    }
    for (int i = tid; i < N * PD; i += nthr) {
        const long base = ((((long)(b * p.lanes + lane) * H + h) * p.nchunks + c) * N * PD) + i;
        in_s[i] = ld(p.DSTATES_IN + base);
        st_s[i] = ld(p.STATES + ((((long)(b * H + h) * p.nchunks + c) * N * PD) + i));
    }
    __syncthreads();

    // ---- lkq = k_sc @ q_rot^T; unmasked + strict-upper segsum mask ----
    for (int i = tid; i < cs * cs; i += nthr) {
        const int r = i / cs, j = i % cs;
        float acc = 0.f;
        for (int n = 0; n < N; ++n) acc += k_sc[r * N + n] * q_rot[j * N + n];
        lkq_u[i] = acc;
        const float es = expf(p.SEGSUM[((((long)(b * H + h) * p.nchunks + c) * cs + j) * cs + r)]);
        lkq_m[i] = (r < j) ? acc * es : 0.f;
    }
    __syncthreads();

    // ---- dPsiV / dv ----
    for (int i = tid; i < cs * PD; i += nthr) {
        const int r = i / PD, pd = i % PD;
        float acc = 0.f;
        for (int n = 0; n < N; ++n) acc += k_sc[r * N + n] * in_s[n * PD + pd];
        acc *= sc_exprev[r];
        for (int j = 0; j < cs; ++j) acc += lkq_m[r * cs + j] * dout_s[j * PD + pd];
        const float qkd = ld(p.QK_DOT + ((long)(b * H + h) * S + (s0 + r)));
        acc += dout_s[i] * p.D[h] + dout_s[i] * qkd * sc_gamma[r];
        p.DV[(((long)((b * p.lanes + lane) * S + (s0 + r)) * H + h) * PD + pd)] = __float2bfloat16(acc);
    }

    // ---- diag / row reductions (one thread per row) + dD ----
    if (tid < cs) {
        const int r = tid;
        float dot = 0.f;
        for (int pd = 0; pd < PD; ++pd) dot += dout_s[r * PD + pd] * v_s[r * PD + pd];
        const float qkd = ld(p.QK_DOT + ((long)(b * H + h) * S + (s0 + r)));
        p.DGAMMA_DIAG[(((long)(b * p.lanes + lane) * H + h) * S + (s0 + r))] = qkd * dot;
        sc_dqg[r] = dot * sc_gamma[r];
    }
    if (tid == 0) {
        float acc = 0.f;
        for (int i = 0; i < cs * PD; ++i) acc += dout_s[i] * v_s[i];
        p.DD[(((long)(b * p.lanes + lane) * H + h) * p.nchunks + c)] = acc;
    }
    __syncthreads();

    // ---- dk inter + ddacsrev; dkq + DSSDA + mask ----
    for (int i = tid; i < cs * N; i += nthr) {
        const int r = i / N, n = i % N;
        float acc = 0.f;
        for (int pd = 0; pd < PD; ++pd) acc += v_s[r * PD + pd] * in_s[n * PD + pd];
        dk_t[i] = acc;
    }
    for (int i = tid; i < cs * cs; i += nthr) {
        const int r = i / cs, j = i % cs;
        float acc = 0.f;
        for (int pd = 0; pd < PD; ++pd) acc += v_s[r * PD + pd] * dout_s[j * PD + pd];
        p.DSSDA[(((((long)(b * p.lanes + lane) * H + h) * p.nchunks + c) * cs + r) * cs + j)] =
            lkq_u[i] * acc;
        const float es = expf(p.SEGSUM[((((long)(b * H + h) * p.nchunks + c) * cs + j) * cs + r)]);
        dkq_m[i] = (r < j) ? acc * es : 0.f;
    }
    __syncthreads();
    if (tid < cs) {
        const int r = tid;
        float acc = 0.f;
        for (int n = 0; n < N; ++n) acc += k_sc[r * N + n] * dk_t[r * N + n];
        p.DDA_CS_REV[(((long)(b * p.lanes + lane) * H + h) * S + (s0 + r))] = acc;
    }
    __syncthreads();
    for (int i = tid; i < cs * N; i += nthr) {
        const int r = i / N, n = i % N;
        float acc = dk_t[i] * sc_exprev[r];
        for (int j = 0; j < cs; ++j) acc += dkq_m[r * cs + j] * q_rot[j * N + n];
        dk_t[i] = acc;
    }
    __syncthreads();
    if (tid < cs) {
        const int r = tid;
        float acc = 0.f;
        for (int n = 0; n < N; ++n) acc += k_sc[r * N + n] * dk_t[r * N + n];
        p.DFACTOR[(((long)(b * p.lanes + lane) * H + h) * S + (s0 + r))] = acc / sc_tscale[r];
    }
    __syncthreads();
    for (int i = tid; i < cs * N; i += nthr) dk_t[i] *= sc_tscale[i / N];
    __syncthreads();

    // ---- dq inter + ddacs; scale; += dkq_m^T @ k_sc ----
    for (int i = tid; i < cs * N; i += nthr) {
        const int r = i / N, n = i % N;
        float acc = 0.f;
        for (int pd = 0; pd < PD; ++pd) acc += dout_s[r * PD + pd] * st_s[n * PD + pd];
        dq_t[i] = acc;
    }
    __syncthreads();
    if (tid < cs) {
        const int r = tid;
        float acc = 0.f;
        for (int n = 0; n < N; ++n) acc += q_rot[r * N + n] * dq_t[r * N + n];
        p.DDA_CS[(((long)(b * p.lanes + lane) * H + h) * S + (s0 + r))] = acc;
    }
    __syncthreads();
    for (int i = tid; i < cs * N; i += nthr) {
        const int r = i / N, n = i % N;
        float acc = dq_t[i] * sc_expcs[r];
        for (int j = 0; j < cs; ++j) acc += dkq_m[j * cs + r] * k_sc[j * N + n];
        dq_t[i] = acc;
    }
    __syncthreads();

    // ---- inverse rotary + dangles + diag corrections + atomic head-sums ----
    for (int i = tid; i < cs * NR; i += nthr) {
        const int r = i / NR, n = i % NR;
        const float ca = cosf(ang_s[i]), sa = sinf(ang_s[i]);
        const float dk1 = dk_t[r * N + n], dk2 = dk_t[r * N + N / 2 + n];
        const float dq1 = dq_t[r * N + n], dq2 = dq_t[r * N + N / 2 + n];
        const float kp1 = k_pre[r * N + n], kp2 = k_pre[r * N + N / 2 + n];
        const float qp1 = q_pre[r * N + n], qp2 = q_pre[r * N + N / 2 + n];
        float da = dk1 * (-kp1 * sa - kp2 * ca) + dk2 * (kp1 * ca - kp2 * sa);
        da += dq1 * (-qp1 * sa - qp2 * ca) + dq2 * (qp1 * ca - qp2 * sa);
        p.DANGLES[(((long)((b * p.lanes + lane) * S + (s0 + r)) * H + h) * NR + n)] = da;
    }
    __syncthreads();
    for (int i = tid; i < cs * N; i += nthr) {
        const int r = i / N, n = i % N;
        float dkv = dk_t[i], dqv = dq_t[i];
        if (n < NR) {
            const float ca = cosf(ang_s[r * NR + n]), sa = sinf(ang_s[r * NR + n]);
            dkv = ca * dk_t[r * N + n] + sa * dk_t[r * N + N / 2 + n];
            dqv = ca * dq_t[r * N + n] + sa * dq_t[r * N + N / 2 + n];
        } else if (n >= N / 2 && n < N / 2 + NR) {
            const int nn = n - N / 2;
            const float ca = cosf(ang_s[r * NR + nn]), sa = sinf(ang_s[r * NR + nn]);
            dkv = -sa * dk_t[r * N + nn] + ca * dk_t[r * N + N / 2 + nn];
            dqv = -sa * dq_t[r * N + nn] + ca * dq_t[r * N + N / 2 + nn];
        }
        dkv += sc_dqg[r] * q_pre[i];
        dqv += sc_dqg[r] * k_pre[i];
        const long o = ((long)((b * p.lanes + lane) * S + (s0 + r))) * N + n;
        atomicAdd(p.DK + o, dkv);
        atomicAdd(p.DQ + o, dqv);
    }
}

}  // namespace

void launch_chunkparallel_pass_c_simple(const ChunkParallelPassCParams& params,
                                        cudaStream_t stream) {
    const int cs = params.chunk_size, N = params.d_state, PD = params.headdim;
    const size_t smem = sizeof(float) *
        (6 * (size_t)cs * N + 2 * (size_t)cs * PD + 2 * (size_t)N * PD +
         3 * (size_t)cs * cs + (size_t)cs * params.n_rot + 5 * (size_t)cs);
    cudaFuncSetAttribute(chunkparallel_pass_c_simple_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    dim3 grid(params.heads, params.batch, params.lanes * params.nchunks);
    chunkparallel_pass_c_simple_kernel<<<grid, 128, smem, stream>>>(params);
}

}  // namespace lbi_mamba3
