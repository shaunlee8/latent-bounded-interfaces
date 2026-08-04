// Forward dual-scan passes, correctness scaffold: plain fp32 math, no tensor
// cores; one CTA per (chunk[, lane], head, batch); parity oracle for _opt.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "fwd_dualscan.hpp"

namespace lbi_mamba3 {

namespace {

using bf16 = __nv_bfloat16;

__device__ inline float ld(const bf16* p) { return __bfloat162float(*p); }

// pass A (primal): SC[c] = sum_j exp(L_last - L_j) * scale_j * kr[j,:] (x) v[j,:]
__global__ void fwd_passA_primal_kernel(FwdDualscanParams p) {
    const int c = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    const int cs = p.chunk_size, N = p.d_state, P = p.headdim;
    const int S = p.seqlen, H = p.heads;
    const int nc = S / cs;
    const int s0 = c * cs;
    const int tid = threadIdx.x, nthr = blockDim.x;

    extern __shared__ float sm[];
    float* kscw = sm;             // [cs][N]  wl_j * scale_j * kr
    float* v_s = kscw + cs * N;   // [cs][P]
    float* L_s = v_s + cs * P;    // [cs]

    for (int i = tid; i < cs; i += nthr)
        L_s[i] = p.L[(b * H + h) * S + s0 + i];
    __syncthreads();
    const float llast = L_s[cs - 1];
    for (int idx = tid; idx < cs * N; idx += nthr) {
        const int j = idx / N, n = idx % N;
        const float wl = expf(llast - L_s[j]);
        const float sc = p.SCALE[(b * H + h) * S + s0 + j];
        kscw[idx] = wl * sc * ld(p.KR + (((long)b * S + s0 + j) * H + h) * N + n);
    }
    for (int idx = tid; idx < cs * P; idx += nthr) {
        const int j = idx / P, q = idx % P;
        v_s[idx] = ld(p.V + (((long)b * S + s0 + j) * H + h) * P + q);
    }
    __syncthreads();
    for (int idx = tid; idx < N * P; idx += nthr) {
        const int n = idx / P, q = idx % P;
        float acc = 0.f;
        for (int j = 0; j < cs; ++j) acc += kscw[j * N + n] * v_s[j * P + q];
        p.SC[((((long)b * H + h) * nc + c) * N + n) * P + q] = acc;
    }
}

// shared prologue helper: generate ksc / dksc rows into smem.
__device__ void gen_k_rows(const FwdDualscanParams& p, int b, int h, int l,
                           int s0, float* ksc, float* dksc, const float* sc_s,
                           const float* dsc_s, int tid, int nthr) {
    const int cs = p.chunk_size, N = p.d_state, Da = p.n_rot;
    const int S = p.seqlen, H = p.heads, G = p.groups;
    const int hg = h / (H / G);
    for (int idx = tid; idx < cs * (N / 2); idx += nthr) {
        const int j = idx / (N / 2), pp = idx % (N / 2);
        const long krow = (((long)b * S + s0 + j) * H + h) * N;
        const float k0 = ld(p.KR + krow + 2 * pp);
        const float k1 = ld(p.KR + krow + 2 * pp + 1);
        const long drow = ((((long)l * p.batch + b) * S + s0 + j) * G + hg) * N;
        const float d0 = ld(p.DKRAW + drow + 2 * pp);
        const float d1 = ld(p.DKRAW + drow + 2 * pp + 1);
        float cw = 1.f, sw = 0.f, dt = 0.f;
        if (pp < Da) {
            const long arow = (((long)b * H + h) * S + s0 + j) * Da + pp;
            cw = ld(p.COS + arow);
            sw = ld(p.SIN + arow);
            dt = ld(p.DTHETA + ((((long)l * p.batch + b) * H + h) * S + s0 + j) * Da + pp);
        }
        const float dkr0 = d0 * cw - d1 * sw - dt * k1;
        const float dkr1 = d0 * sw + d1 * cw + dt * k0;
        const float sc = sc_s[j], dsc = dsc_s[j];
        ksc[j * N + 2 * pp] = k0 * sc;
        ksc[j * N + 2 * pp + 1] = k1 * sc;
        dksc[j * N + 2 * pp] = dkr0 * sc + k0 * dsc;
        dksc[j * N + 2 * pp + 1] = dkr1 * sc + k1 * dsc;
    }
}

__device__ void gen_q_rows(const FwdDualscanParams& p, int b, int h, int l,
                           int s0, float* qr, float* dqr, int tid, int nthr) {
    const int cs = p.chunk_size, N = p.d_state, Da = p.n_rot;
    const int S = p.seqlen, H = p.heads, G = p.groups;
    const int hg = h / (H / G);
    for (int idx = tid; idx < cs * (N / 2); idx += nthr) {
        const int j = idx / (N / 2), pp = idx % (N / 2);
        const long qrow = (((long)b * S + s0 + j) * H + h) * N;
        const float q0 = ld(p.QR + qrow + 2 * pp);
        const float q1 = ld(p.QR + qrow + 2 * pp + 1);
        const long drow = ((((long)l * p.batch + b) * S + s0 + j) * G + hg) * N;
        const float d0 = ld(p.DQRAW + drow + 2 * pp);
        const float d1 = ld(p.DQRAW + drow + 2 * pp + 1);
        float cw = 1.f, sw = 0.f, dt = 0.f;
        if (pp < Da) {
            const long arow = (((long)b * H + h) * S + s0 + j) * Da + pp;
            cw = ld(p.COS + arow);
            sw = ld(p.SIN + arow);
            dt = ld(p.DTHETA + ((((long)l * p.batch + b) * H + h) * S + s0 + j) * Da + pp);
        }
        qr[j * N + 2 * pp] = q0;
        qr[j * N + 2 * pp + 1] = q1;
        dqr[j * N + 2 * pp] = d0 * cw - d1 * sw - dt * q1;
        dqr[j * N + 2 * pp + 1] = d0 * sw + d1 * cw + dt * q0;
    }
}

// pass A (tangent, per lane): dSC via generated dksc + dL coupling.
__global__ void fwd_passA_tan_kernel(FwdDualscanParams p) {
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

    extern __shared__ float sm[];
    float* ksc = sm;               // [cs][N]
    float* dksc = ksc + cs * N;    // [cs][N]
    float* v_s = dksc + cs * N;    // [cs][P]
    float* dv_s = v_s + cs * P;    // [cs][P]
    float* L_s = dv_s + cs * P;    // [cs] x4 scalar rows
    float* dL_s = L_s + cs;
    float* sc_s = dL_s + cs;
    float* dsc_s = sc_s + cs;

    for (int i = tid; i < cs; i += nthr) {
        const long row = ((long)b * H + h) * S + s0 + i;
        const long lrow = (((long)l * p.batch + b) * H + h) * S + s0 + i;
        L_s[i] = p.L[row];
        dL_s[i] = p.DL[lrow];
        sc_s[i] = p.SCALE[row];
        dsc_s[i] = p.DSCALE[lrow];
    }
    for (int idx = tid; idx < cs * P; idx += nthr) {
        const int j = idx / P, q = idx % P;
        v_s[idx] = ld(p.V + (((long)b * S + s0 + j) * H + h) * P + q);
        dv_s[idx] = ld(p.DV + ((((long)l * p.batch + b) * S + s0 + j) * H + h) * P + q);
    }
    __syncthreads();
    gen_k_rows(p, b, h, l, s0, ksc, dksc, sc_s, dsc_s, tid, nthr);
    __syncthreads();
    const float llast = L_s[cs - 1], dllast = dL_s[cs - 1];
    for (int idx = tid; idx < N * P; idx += nthr) {
        const int n = idx / P, q = idx % P;
        float acc = 0.f;
        for (int j = 0; j < cs; ++j) {
            const float wl = expf(llast - L_s[j]);
            acc += wl * ((dllast - dL_s[j]) * ksc[j * N + n] + dksc[j * N + n]) * v_s[j * P + q]
                 + wl * ksc[j * N + n] * dv_s[j * P + q];
        }
        p.DSC[(((((long)l * p.batch + b) * H + h) * nc + c) * N + n) * P + q] = acc;
    }
}

// pass C core: intra dual quadratic + inter readout; writes out/dout rows
// into caller-provided smem row buffers (or gmem for the full variant).
__global__ void fwd_passC_kernel(FwdDualscanParams p) {
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

    extern __shared__ float sm[];
    float* qr = sm;                 // [cs][N]
    float* dqr = qr + cs * N;       // [cs][N]
    float* ksc = dqr + cs * N;      // [cs][N]
    float* dksc = ksc + cs * N;     // [cs][N]
    float* v_s = dksc + cs * N;     // [cs][P]
    float* dv_s = v_s + cs * P;     // [cs][P]
    float* WQK = dv_s + cs * P;     // [cs][cs]
    float* dWQK = WQK + cs * cs;    // [cs][cs]
    float* L_s = dWQK + cs * cs;    // [cs] x4
    float* dL_s = L_s + cs;
    float* sc_s = dL_s + cs;
    float* dsc_s = sc_s + cs;

    for (int i = tid; i < cs; i += nthr) {
        const long row = ((long)b * H + h) * S + s0 + i;
        const long lrow = (((long)l * p.batch + b) * H + h) * S + s0 + i;
        L_s[i] = p.L[row];
        dL_s[i] = p.DL[lrow];
        sc_s[i] = p.SCALE[row];
        dsc_s[i] = p.DSCALE[lrow];
    }
    for (int idx = tid; idx < cs * P; idx += nthr) {
        const int j = idx / P, q = idx % P;
        v_s[idx] = ld(p.V + (((long)b * S + s0 + j) * H + h) * P + q);
        dv_s[idx] = ld(p.DV + ((((long)l * p.batch + b) * S + s0 + j) * H + h) * P + q);
    }
    __syncthreads();
    gen_q_rows(p, b, h, l, s0, qr, dqr, tid, nthr);
    gen_k_rows(p, b, h, l, s0, ksc, dksc, sc_s, dsc_s, tid, nthr);
    __syncthreads();

    // intra QK/dQK with the causal decay mask, in place.
    for (int idx = tid; idx < cs * cs; idx += nthr) {
        const int i = idx / cs, j = idx % cs;
        float qk = 0.f, dqk = 0.f;
        if (i >= j) {
            for (int n = 0; n < N; ++n) {
                qk += qr[i * N + n] * ksc[j * N + n];
                dqk += dqr[i * N + n] * ksc[j * N + n] + qr[i * N + n] * dksc[j * N + n];
            }
            const float w = expf(L_s[i] - L_s[j]);
            dqk = w * (dL_s[i] - dL_s[j]) * qk + w * dqk;
            qk = w * qk;
        }
        WQK[idx] = qk;
        dWQK[idx] = dqk;
    }
    __syncthreads();

    const long sin_base = (((long)b * H + h) * nc + c) * N * P;
    const long dsin_base = ((((long)l * p.batch + b) * H + h) * nc + c) * N * P;
    for (int idx = tid; idx < cs * P; idx += nthr) {
        const int i = idx / P, q = idx % P;
        float intra = 0.f, dintra = 0.f;
        for (int j = 0; j <= i; ++j) {
            intra += WQK[i * cs + j] * v_s[j * P + q];
            dintra += dWQK[i * cs + j] * v_s[j * P + q] + WQK[i * cs + j] * dv_s[j * P + q];
        }
        float qs = 0.f, dqs = 0.f, qds = 0.f;
        for (int n = 0; n < N; ++n) {
            const float sin_v = p.S_IN[sin_base + (long)n * P + q];
            const float dsin_v = p.DS_IN[dsin_base + (long)n * P + q];
            qs += qr[i * N + n] * sin_v;
            dqs += dqr[i * N + n] * sin_v;
            qds += qr[i * N + n] * dsin_v;
        }
        const float e = expf(L_s[i]);
        const float out = intra + e * qs;
        const float dout = dintra + e * (dL_s[i] * qs + dqs + qds);
        p.DOUT[((((long)l * p.batch + b) * S + s0 + i) * H + h) * P + q] = dout;
        if (l == 0) p.OUT[(((long)b * S + s0 + i) * H + h) * P + q] = out;
    }
}

// pass C (mean epilogue): pass C + finalize + masked row-sum -> PART.
__global__ void fwd_passC_mean_kernel(FwdDualscanParams p) {
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

    extern __shared__ float sm[];
    float* qr = sm;
    float* dqr = qr + cs * N;
    float* ksc = dqr + cs * N;
    float* dksc = ksc + cs * N;
    float* v_s = dksc + cs * N;
    float* dv_s = v_s + cs * P;
    float* WQK = dv_s + cs * P;
    float* dWQK = WQK + cs * cs;
    float* dfin = dWQK + cs * cs;   // [cs][P] finalized tangent rows
    float* L_s = dfin + cs * P;     // [cs] x6
    float* dL_s = L_s + cs;
    float* sc_s = dL_s + cs;
    float* dsc_s = sc_s + cs;
    float* qk_s = dsc_s + cs;
    float* dqk_s = qk_s + cs;

    for (int i = tid; i < cs; i += nthr) {
        const long row = ((long)b * H + h) * S + s0 + i;
        const long lrow = (((long)l * p.batch + b) * H + h) * S + s0 + i;
        L_s[i] = p.L[row];
        dL_s[i] = p.DL[lrow];
        sc_s[i] = p.SCALE[row];
        dsc_s[i] = p.DSCALE[lrow];
        qk_s[i] = p.QKDOT[row];
        dqk_s[i] = p.DQKDOT[lrow];
    }
    for (int idx = tid; idx < cs * P; idx += nthr) {
        const int j = idx / P, q = idx % P;
        v_s[idx] = ld(p.V + (((long)b * S + s0 + j) * H + h) * P + q);
        dv_s[idx] = ld(p.DV + ((((long)l * p.batch + b) * S + s0 + j) * H + h) * P + q);
    }
    __syncthreads();
    gen_q_rows(p, b, h, l, s0, qr, dqr, tid, nthr);
    gen_k_rows(p, b, h, l, s0, ksc, dksc, sc_s, dsc_s, tid, nthr);
    __syncthreads();
    for (int idx = tid; idx < cs * cs; idx += nthr) {
        const int i = idx / cs, j = idx % cs;
        float qk = 0.f, dqk = 0.f;
        if (i >= j) {
            for (int n = 0; n < N; ++n) {
                qk += qr[i * N + n] * ksc[j * N + n];
                dqk += dqr[i * N + n] * ksc[j * N + n] + qr[i * N + n] * dksc[j * N + n];
            }
            const float w = expf(L_s[i] - L_s[j]);
            dqk = w * (dL_s[i] - dL_s[j]) * qk + w * dqk;
            qk = w * qk;
        }
        WQK[idx] = qk;
        dWQK[idx] = dqk;
    }
    __syncthreads();

    const long sin_base = (((long)b * H + h) * nc + c) * N * P;
    const long dsin_base = ((((long)l * p.batch + b) * H + h) * nc + c) * N * P;
    const float dskip = p.DSKIP[h];
    for (int idx = tid; idx < cs * P; idx += nthr) {
        const int i = idx / P, q = idx % P;
        float intra = 0.f, dintra = 0.f;
        for (int j = 0; j <= i; ++j) {
            intra += WQK[i * cs + j] * v_s[j * P + q];
            dintra += dWQK[i * cs + j] * v_s[j * P + q] + WQK[i * cs + j] * dv_s[j * P + q];
        }
        float qs = 0.f, dqs = 0.f, qds = 0.f;
        for (int n = 0; n < N; ++n) {
            const float sin_v = p.S_IN[sin_base + (long)n * P + q];
            const float dsin_v = p.DS_IN[dsin_base + (long)n * P + q];
            qs += qr[i * N + n] * sin_v;
            dqs += dqr[i * N + n] * sin_v;
            qds += qr[i * N + n] * dsin_v;
        }
        const float e = expf(L_s[i]);
        const float out = intra + e * qs;
        const float dout = dintra + e * (dL_s[i] * qs + dqs + qds);
        // finalize (D-skip, QK-dot skip, Z-gate), masked.
        const float vv = v_s[idx], dvv = dv_s[idx];
        const float o = out + dskip * vv - vv * qk_s[i];
        const float dO = dout + dskip * dvv - (dvv * qk_s[i] + vv * dqk_s[i]);
        const float z = ld(p.Z + (((long)b * S + s0 + i) * H + h) * P + q);
        const float dz = ld(p.DZ + ((((long)l * p.batch + b) * S + s0 + i) * H + h) * P + q);
        const float sg = 1.f / (1.f + expf(-z));
        const float gate = z * sg;
        const float dgate = sg * (1.f + z * (1.f - sg)) * dz;
        dfin[idx] = (s0 + i < p.s_true) ? (dO * gate + o * dgate) : 0.f;
    }
    __syncthreads();
    for (int q = tid; q < P; q += nthr) {
        float acc = 0.f;
        for (int i = 0; i < cs; ++i) acc += dfin[i * P + q];
        p.PART[((((long)l * p.batch + b) * H + h) * nc + c) * P + q] = acc;
    }
}

}  // namespace

void launch_fwd_passA_primal(const FwdDualscanParams& p, cudaStream_t stream) {
    const int cs = p.chunk_size, N = p.d_state, P = p.headdim;
    const int nc = p.seqlen / cs;
    const size_t smem = sizeof(float) * (cs * N + cs * P + cs);
    cudaFuncSetAttribute(fwd_passA_primal_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    dim3 grid(nc, p.heads, p.batch);
    fwd_passA_primal_kernel<<<grid, 128, smem, stream>>>(p);
}

void launch_fwd_passA_tan(const FwdDualscanParams& p, cudaStream_t stream) {
    const int cs = p.chunk_size, N = p.d_state, P = p.headdim;
    const int nc = p.seqlen / cs;
    const size_t smem = sizeof(float) * (2 * cs * N + 2 * cs * P + 4 * cs);
    cudaFuncSetAttribute(fwd_passA_tan_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    dim3 grid(nc * p.lanes, p.heads, p.batch);
    fwd_passA_tan_kernel<<<grid, 128, smem, stream>>>(p);
}

void launch_fwd_passC(const FwdDualscanParams& p, cudaStream_t stream) {
    const int cs = p.chunk_size, N = p.d_state, P = p.headdim;
    const int nc = p.seqlen / cs;
    const size_t smem =
        sizeof(float) * (4 * cs * N + 2 * cs * P + 2 * cs * cs + 4 * cs);
    cudaFuncSetAttribute(fwd_passC_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    dim3 grid(nc * p.lanes, p.heads, p.batch);
    fwd_passC_kernel<<<grid, 128, smem, stream>>>(p);
}

void launch_fwd_passC_mean(const FwdDualscanParams& p, cudaStream_t stream) {
    const int cs = p.chunk_size, N = p.d_state, P = p.headdim;
    const int nc = p.seqlen / cs;
    const size_t smem =
        sizeof(float) * (4 * cs * N + 3 * cs * P + 2 * cs * cs + 6 * cs);
    cudaFuncSetAttribute(fwd_passC_mean_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    dim3 grid(nc * p.lanes, p.heads, p.batch);
    fwd_passC_mean_kernel<<<grid, 128, smem, stream>>>(p);
}

}  // namespace lbi_mamba3
