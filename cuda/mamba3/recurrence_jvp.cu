// Lane-batched wgmma recurrence kernel of the chunked SSM scan (1 warpgroup/CTA,
// cs=32): emits the raw OUT/DOUT fields, or with FIN the finalized fields
// (D-skip, QK-dot skip, Z-gate folded into the epilogue).

// State: S_new = cd*S + ksc^T@vwl; dS_new = cd*dS + dLc*S_new + dksc^T@vwl
// + ksc^T@dvwl, with vwl = wl.v.

// Out: dout_i = dL_i out_i + (wqk@dvc)_i + (dwq@v)_i + e_i[(dqr@S)+(qr@dS)]_i.
// M/B-stacked tiles: qd=[qr;dqr], kd=[ksc;dksc], wd=[wqk;dwq].
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <tl_templates/cuda/instruction/wgmma.h>
#include <tl_templates/cuda/intrin.h>
#include <tl_templates/cuda/barrier.h>
#include "lbi_mma.cuh"

namespace lbi_recurrence {

using bf16 = __nv_bfloat16;

constexpr int GCS = 32, NN = 128, PP = 64;

__device__ inline int sw128(int R, int r, int c) {
    return (c >> 6) * R * 64 + r * 64 + ((((c >> 3) & 7) ^ (r & 7)) << 3) + (c & 7);
}
__device__ inline int sw32(int R, int r, int c) {
    return (c >> 4) * R * 16 + r * 16 + ((((c >> 3) & 1) ^ ((r >> 2) & 1)) << 3) + (c & 7);
}
__device__ inline void acc_rc(int w, int l, int reg, int& r, int& c) {
    const int j8 = reg >> 2, e = reg & 3;
    r = w * 16 + (l >> 2) + ((e >> 1) << 3);
    c = (j8 << 3) + ((l & 3) << 1) + (e & 1);
}

// smem layout (bf16 elems)
constexpr int OFF_S    = 0;                        // [128][64] SW128  16 KB
constexpr int OFF_DS   = OFF_S   + 128 * 64;       // [128][64] SW128  16 KB
constexpr int OFF_QD   = OFF_DS  + 128 * 64;       // [64][128] SW128  16 KB
constexpr int OFF_KD   = OFF_QD  + 64 * 128;       // [64][128] SW128  16 KB
constexpr int OFF_V    = OFF_KD  + 64 * 128;       // [32][64]  SW32    4 KB
constexpr int OFF_DVC  = OFF_V   + 32 * 64;        // [32][64]  SW32    4 KB
constexpr int OFF_VWL  = OFF_DVC + 32 * 64;        // [32][64]  SW32    4 KB
constexpr int OFF_DVWL = OFF_VWL + 32 * 64;        // [32][64]  SW32    4 KB
constexpr int OFF_WD   = OFF_DVWL + 32 * 64;       // [64][64]  SW128   8 KB
constexpr int OFF_X    = OFF_WD  + 64 * 64;
// dob fp32 [32][64] = 8 KB (cross-warp dout row assembly).
constexpr int OFF_F32  = OFF_X + 4096;
// floats: L,dL,e,ie,sc,dsc[32] + qk,dqk[32] + 64 spare
constexpr size_t OFF_STG = (size_t)OFF_F32 * 2 + (8 * GCS + PP) * 4;
// FULL store staging: OUT/DOUT bf16 [32][72] each (row-padded, conflict-free).
constexpr int STG_STRIDE = 72;
constexpr size_t SMEM_MIN = OFF_STG + 2 * GCS * STG_STRIDE * 2;

template <bool FIN>
__global__ void __launch_bounds__(128, 1) recurrence_jvp_kernel(
    const bf16* __restrict__ QR, const bf16* __restrict__ KR,
    const bf16* __restrict__ V, const bf16* __restrict__ DQRAW,
    const bf16* __restrict__ DKRAW, const bf16* __restrict__ DV,
    const bf16* __restrict__ DTHETA, const bf16* __restrict__ COS,
    const bf16* __restrict__ SIN, const float* __restrict__ SCALE,
    const float* __restrict__ DSCALE, const float* __restrict__ L,
    const float* __restrict__ DL,
    const bf16* __restrict__ Z, const bf16* __restrict__ DZ,
    const float* __restrict__ QKD, const float* __restrict__ DQKD,
    const float* __restrict__ DSK,
    bf16* __restrict__ OUT, bf16* __restrict__ DOUT,
    int B_, int H_, int G_, int Sp, int Da, int nc, int s_true) {
    extern __shared__ __align__(1024) char smem[];
    bf16* base = reinterpret_cast<bf16*>(smem);
    bf16* S_s   = base + OFF_S;
    bf16* dS_s  = base + OFF_DS;
    bf16* qd_s  = base + OFF_QD;
    bf16* kd_s  = base + OFF_KD;
    bf16* v_s   = base + OFF_V;
    bf16* dvc_s = base + OFF_DVC;
    bf16* vwl_s = base + OFF_VWL;
    bf16* dvwl_s = base + OFF_DVWL;
    bf16* wd_s  = base + OFF_WD;
    float* dob_s = reinterpret_cast<float*>(base + OFF_X);     // FULL [32][64]
    bf16* out_st  = reinterpret_cast<bf16*>(smem + OFF_STG);   // FULL [32][72]
    bf16* dout_st = out_st + GCS * STG_STRIDE;                 // FULL [32][72]
    float* L_s   = reinterpret_cast<float*>(base + OFF_F32);
    float* dL_s  = L_s + GCS;
    float* e_s   = dL_s + GCS;
    float* ie_s  = e_s + GCS;
    float* sc_s  = ie_s + GCS;
    float* dsc_s = sc_s + GCS;
    float* qk_s  = dsc_s + GCS;
    float* dqk_s = qk_s + GCS;

    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int bh = blockIdx.x % (B_ * H_);
    const int lane_i = blockIdx.x / (B_ * H_);
    const int b = bh / H_, h = bh % H_, g = h / (H_ / G_);
    const float dsk = FIN ? DSK[h] : 0.f;

    const long lQ = (long)lane_i * B_ * Sp * G_ * NN;
    const long lV = (long)lane_i * B_ * Sp * H_ * PP;
    const long lT = (long)lane_i * B_ * H_ * Sp * Da;
    const long lS = (long)lane_i * B_ * H_ * Sp;
    const long lO = (long)lane_i * B_ * Sp * H_ * PP;   // DOUT lane stride

    for (int i = tid; i < 2 * 128 * 64; i += 128) S_s[i] = __float2bfloat16(0.f);
    __syncthreads();

    __shared__ __align__(8) uint64_t mma_bars[lbi::mma::Pipeline::NBARS];
    lbi::mma::Pipeline pipe;
    pipe.init(mma_bars);

    tl::GmmaDescriptor d_qd, d_kd, d_kdT, d_S, d_dS, d_v, d_dvc, d_vwl, d_dvwl, d_wd;
    tl::initialize_wgmma_descriptor<1, 1, 64>(d_qd, qd_s);
    tl::initialize_wgmma_descriptor<1, 1, 64>(d_kd, kd_s);
    tl::initialize_wgmma_descriptor<1, 512, 64>(d_kdT, kd_s);
    tl::initialize_wgmma_descriptor<1, 1024, 64>(d_S, S_s);
    tl::initialize_wgmma_descriptor<1, 1024, 64>(d_dS, dS_s);
    tl::initialize_wgmma_descriptor<3, 64, 16>(d_v, v_s);
    tl::initialize_wgmma_descriptor<3, 64, 16>(d_dvc, dvc_s);
    tl::initialize_wgmma_descriptor<3, 64, 16>(d_vwl, vwl_s);
    tl::initialize_wgmma_descriptor<3, 64, 16>(d_dvwl, dvwl_s);
    tl::initialize_wgmma_descriptor<1, 1, 64>(d_wd, wd_s);

    for (int c = 0; c < nc; ++c) {
        const int s0 = c * GCS;
        const long qrow = (((long)b * Sp + s0) * H_ + h) * NN;
        const long grow = (((long)b * Sp + s0) * G_ + g) * NN;
        const long vrow = (((long)b * Sp + s0) * H_ + h) * PP;
        const long lrow = ((long)b * H_ + h) * Sp + s0;
        const long trow = (((long)b * H_ + h) * Sp + s0) * Da;
        if (tid < GCS) {
            const float Lv = L[lrow + tid];
            const float ev = __expf(Lv);
            L_s[tid] = Lv;
            dL_s[tid] = DL[lS + lrow + tid];
            e_s[tid] = ev;
            // ie_s slot retired: use sites take difference-form exponentials,
            // keeping the ratio form's 1/exp(L) overflow out (admissibility).
            sc_s[tid] = SCALE[lrow + tid];
            dsc_s[tid] = DSCALE[lS + lrow + tid];
            if (FIN) {
                qk_s[tid] = QKD[lrow + tid];
                dqk_s[tid] = DQKD[lS + lrow + tid];
            }
        }
        __syncthreads();                                   // sync 1
        if (FIN) {
            // Stage this chunk's z/dz rows into the (now free) row-staging
            // buffers as coalesced lines; the epilogue reads each pair from
            // smem right before overwriting the slot with its finalized row.
            for (int idx = tid; idx < GCS * 8; idx += 128) {
                const int i = idx >> 3, u = idx & 7;
                *reinterpret_cast<uint4*>(out_st + i * STG_STRIDE + u * 8) =
                    *reinterpret_cast<const uint4*>(Z + vrow + (long)i * H_ * PP + u * 8);
                *reinterpret_cast<uint4*>(dout_st + i * STG_STRIDE + u * 8) =
                    *reinterpret_cast<const uint4*>(DZ + lO + vrow + (long)i * H_ * PP + u * 8);
            }
        }
        const float wc = e_s[GCS - 1];
        const float dLc = dL_s[GCS - 1];

        // ---- staging: q/k + rotary-JVP tangents ----
        for (int idx = tid; idx < GCS * 16; idx += 128) {
            const int i = idx >> 4, u = idx & 15;
            const int blk = u >> 3, uu = u & 7;
            const int soff = blk * 4096 + i * 64 + ((uu ^ (i & 7)) << 3);
            const uint4 qq = *reinterpret_cast<const uint4*>(QR + qrow + (long)i * H_ * NN + u * 8);
            const uint4 dq = *reinterpret_cast<const uint4*>(DQRAW + lQ + grow + (long)i * G_ * NN + u * 8);
            const uint4 kk = *reinterpret_cast<const uint4*>(KR + qrow + (long)i * H_ * NN + u * 8);
            const uint4 dk = *reinterpret_cast<const uint4*>(DKRAW + lQ + grow + (long)i * G_ * NN + u * 8);
            float ct[4], st[4], dt[4];
            if (4 * u < Da) {
                const long tb = trow + (long)i * Da + 4 * u;
#pragma unroll
                for (int t = 0; t < 4; ++t) {
                    ct[t] = __bfloat162float(COS[tb + t]);
                    st[t] = __bfloat162float(SIN[tb + t]);
                    dt[t] = __bfloat162float(DTHETA[lT + tb + t]);
                }
            } else {
#pragma unroll
                for (int t = 0; t < 4; ++t) { ct[t] = 1.f; st[t] = 0.f; dt[t] = 0.f; }
            }
            const float sc = sc_s[i], dsc = dsc_s[i];
            const __nv_bfloat162* qp = reinterpret_cast<const __nv_bfloat162*>(&qq);
            const __nv_bfloat162* dqp = reinterpret_cast<const __nv_bfloat162*>(&dq);
            const __nv_bfloat162* kp = reinterpret_cast<const __nv_bfloat162*>(&kk);
            const __nv_bfloat162* dkp = reinterpret_cast<const __nv_bfloat162*>(&dk);
            uint4 o_dq, o_k, o_dk;
            __nv_bfloat162* odq = reinterpret_cast<__nv_bfloat162*>(&o_dq);
            __nv_bfloat162* ok = reinterpret_cast<__nv_bfloat162*>(&o_k);
            __nv_bfloat162* odk = reinterpret_cast<__nv_bfloat162*>(&o_dk);
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const float q0 = __bfloat162float(qp[t].x), q1 = __bfloat162float(qp[t].y);
                const float d0 = __bfloat162float(dqp[t].x), d1 = __bfloat162float(dqp[t].y);
                odq[t] = __floats2bfloat162_rn(d0 * ct[t] - d1 * st[t] - dt[t] * q1,
                                               d0 * st[t] + d1 * ct[t] + dt[t] * q0);
                const float k0 = __bfloat162float(kp[t].x), k1 = __bfloat162float(kp[t].y);
                const float e0 = __bfloat162float(dkp[t].x), e1 = __bfloat162float(dkp[t].y);
                const float dkr0 = e0 * ct[t] - e1 * st[t] - dt[t] * k1;
                const float dkr1 = e0 * st[t] + e1 * ct[t] + dt[t] * k0;
                ok[t] = __floats2bfloat162_rn(k0 * sc, k1 * sc);
                odk[t] = __floats2bfloat162_rn(dkr0 * sc + k0 * dsc, dkr1 * sc + k1 * dsc);
            }
            *reinterpret_cast<uint4*>(qd_s + soff) = qq;
            *reinterpret_cast<uint4*>(qd_s + soff + 32 * 64) = o_dq;
            *reinterpret_cast<uint4*>(kd_s + soff) = o_k;
            *reinterpret_cast<uint4*>(kd_s + soff + 32 * 64) = o_dk;
        }
        // ---- staging: v combos ----
        for (int idx = tid; idx < GCS * 8; idx += 128) {
            const int i = idx >> 3, u = idx & 7;
            const int blk = u >> 1, uu = u & 1;
            const int soff = blk * 512 + i * 16 + ((uu ^ ((i >> 2) & 1)) << 3);
            const uint4 vv = *reinterpret_cast<const uint4*>(V + vrow + (long)i * H_ * PP + u * 8);
            const uint4 dv = *reinterpret_cast<const uint4*>(DV + lV + vrow + (long)i * H_ * PP + u * 8);
            const float wl = __expf(L_s[GCS - 1] - L_s[i]); const float dLi = dL_s[i];
            const __nv_bfloat162* vp = reinterpret_cast<const __nv_bfloat162*>(&vv);
            const __nv_bfloat162* dvp = reinterpret_cast<const __nv_bfloat162*>(&dv);
            uint4 o_dvc, o_vwl, o_dvwl;
            __nv_bfloat162* oc = reinterpret_cast<__nv_bfloat162*>(&o_dvc);
            __nv_bfloat162* ow = reinterpret_cast<__nv_bfloat162*>(&o_vwl);
            __nv_bfloat162* od = reinterpret_cast<__nv_bfloat162*>(&o_dvwl);
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const float v0 = __bfloat162float(vp[t].x), v1 = __bfloat162float(vp[t].y);
                const float w0 = __bfloat162float(dvp[t].x), w1 = __bfloat162float(dvp[t].y);
                const float c0 = w0 - dLi * v0, c1 = w1 - dLi * v1;
                oc[t] = __floats2bfloat162_rn(c0, c1);
                ow[t] = __floats2bfloat162_rn(wl * v0, wl * v1);
                od[t] = __floats2bfloat162_rn(wl * c0, wl * c1);
            }
            *reinterpret_cast<uint4*>(v_s + soff) = vv;
            *reinterpret_cast<uint4*>(dvc_s + soff) = o_dvc;
            *reinterpret_cast<uint4*>(vwl_s + soff) = o_vwl;
            *reinterpret_cast<uint4*>(dvwl_s + soff) = o_dvwl;
        }
        __syncthreads();                                   // sync 2

        // ---- G1: stacked QK ; G2: state updates ----
        // Every accumulator and issue below goes through the architecture
        // seam; the state tiles are two 64x64 halves indexed by mi.
        lbi::mma::Accum<32> accQ, accU[2], accDU[2];
        lbi::mma::fence_operand(accQ);
        lbi::mma::fence_operand(accU[0]);
        lbi::mma::fence_operand(accU[1]);
        lbi::mma::fence_operand(accDU[0]);
        lbi::mma::fence_operand(accDU[1]);
        lbi::mma::arrive();
#pragma unroll
        for (int ki = 0; ki < 8; ++ki) {
            const int off = (((ki >> 2) * 8192 + (ki & 3) * 32) >> 4);
            lbi::mma::mma_ss<64, 64, 16, false, false>(
                uint64_t(d_qd + off), uint64_t(d_kd + off), accQ, ki != 0);
        }
        pipe.commit();                                // G1
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
            for (int ki = 0; ki < 2; ++ki)
                lbi::mma::mma_ss<64, 64, 16, true, true>(
                    uint64_t(d_kdT + ((mi * 8192 + ki * 2048) >> 4)),
                    uint64_t(d_vwl + ((ki * 512) >> 4)), accU[mi], ki != 0);
#pragma unroll
            for (int ki = 0; ki < 2; ++ki)
                lbi::mma::mma_ss<64, 64, 16, true, true>(
                    uint64_t(d_kdT + ((mi * 8192 + 4096 + ki * 2048) >> 4)),
                    uint64_t(d_vwl + ((ki * 512) >> 4)), accDU[mi], ki != 0);
#pragma unroll
            for (int ki = 0; ki < 2; ++ki)
                lbi::mma::mma_ss<64, 64, 16, true, true>(
                    uint64_t(d_kdT + ((mi * 8192 + ki * 2048) >> 4)),
                    uint64_t(d_dvwl + ((ki * 512) >> 4)), accDU[mi], true);
        }
        pipe.commit();                                // G2
        pipe.wait<1>();                               // G1 done
        lbi::mma::fence_operand(accQ);

        // ---- wd assembly ----
        if (FIN) {
            // bf16x2 stores over the column pairs (cc, cc+1); the triangular
            // mask and the decay weight are per column.
#pragma unroll
            for (int wb = 0; wb < 32; wb += lbi::mma::ACC_WINDOW) {
                accQ.window(wb);
            #pragma unroll
                for (int rj = 0; rj < lbi::mma::ACC_WINDOW; rj += 2) {
                    const int reg = wb + rj;
                    int r, cc; acc_rc(warp, lane, reg, r, cc);
                    if (cc < 32) {
                        const int i = (r < 32) ? r : r - 32;
                        const float w0 = (i >= cc) ? __expf(L_s[i] - L_s[cc]) : 0.f;
                        const float w1 = (i >= cc + 1) ? __expf(L_s[i] - L_s[cc + 1]) : 0.f;
                        *reinterpret_cast<__nv_bfloat162*>(wd_s + sw128(64, r, cc)) =
                            __floats2bfloat162_rn(w0 * accQ[reg], w1 * accQ[reg + 1]);
                    }
                }
            }
            __syncthreads();                               // sync 3
            if (warp < 2) {
#pragma unroll
                for (int wb = 0; wb < 32; wb += lbi::mma::ACC_WINDOW) {
                    accQ.window(wb);
                #pragma unroll
                    for (int rj = 0; rj < lbi::mma::ACC_WINDOW; rj += 2) {
                        const int reg = wb + rj;
                        int r, cc; acc_rc(warp, lane, reg, r, cc);
                        if (cc >= 32) {
                            const int j = cc - 32;
                            if (r >= j) {
                                // (r, j) and (r, j+1): the second lane is masked
                                // when r == j (strictly the same values as scalar).
                                const int idx = sw128(64, 32 + r, j);
                                const __nv_bfloat162 cur =
                                    *reinterpret_cast<const __nv_bfloat162*>(wd_s + idx);
                                const float w0 = __expf(L_s[r] - L_s[j]);
                                const float a0 = __low2float(cur) + w0 * accQ[reg];
                                float a1 = __high2float(cur);
                                if (r >= j + 1) a1 += __expf(L_s[r] - L_s[j + 1]) * accQ[reg + 1];
                                *reinterpret_cast<__nv_bfloat162*>(wd_s + idx) =
                                    __floats2bfloat162_rn(a0, a1);
                            }
                        }
                    }
                }
            }
            __syncthreads();                               // sync 4
        } else {
        {
#pragma unroll
            for (int wb = 0; wb < 32; wb += lbi::mma::ACC_WINDOW) {
                accQ.window(wb);
            #pragma unroll
                for (int rj = 0; rj < lbi::mma::ACC_WINDOW; ++rj) {
                    const int reg = wb + rj;
                    int r, cc; acc_rc(warp, lane, reg, r, cc);
                    if (r < 32 && cc < 32) {
                        const float w = (r >= cc) ? __expf(L_s[r] - L_s[cc]) : 0.f;
                        wd_s[sw128(64, r, cc)] = __float2bfloat16(w * accQ[reg]);
                    } else if (r >= 32 && cc < 32) {
                        const int i = r - 32;
                        const float w = (i >= cc) ? __expf(L_s[i] - L_s[cc]) : 0.f;
                        wd_s[sw128(64, r, cc)] = __float2bfloat16(w * accQ[reg]);
                    }
                }
            }
        }
        __syncthreads();                                   // sync 3
        if (warp < 2) {
#pragma unroll
            for (int wb = 0; wb < 32; wb += lbi::mma::ACC_WINDOW) {
                accQ.window(wb);
            #pragma unroll
                for (int rj = 0; rj < lbi::mma::ACC_WINDOW; ++rj) {
                    const int reg = wb + rj;
                    int r, cc; acc_rc(warp, lane, reg, r, cc);
                    if (cc >= 32) {
                        const int j = cc - 32;
                        if (r >= j) {
                            const int idx = sw128(64, 32 + r, j);
                            const float w = __expf(L_s[r] - L_s[j]);
                            wd_s[idx] = __float2bfloat16(
                                __bfloat162float(wd_s[idx]) + w * accQ[reg]);
                        }
                    }
                }
            }
        }
        __syncthreads();                                   // sync 4
        }

        // ---- G3: qd@S + qd@dS (reads OLD S/dS) ----
        lbi::mma::Accum<32> accF, accG;
        lbi::mma::fence_operand(accF);
        lbi::mma::fence_operand(accG);
        lbi::mma::arrive();
#pragma unroll
        for (int ki = 0; ki < 8; ++ki)
            lbi::mma::mma_ss<64, 64, 16, false, true>(
                uint64_t(d_qd + (((ki >> 2) * 8192 + (ki & 3) * 32) >> 4)),
                uint64_t(d_S + ((ki * 2048) >> 4)), accF, ki != 0);
#pragma unroll
        for (int ki = 0; ki < 8; ++ki)
            lbi::mma::mma_ss<64, 64, 16, false, true>(
                uint64_t(d_qd + (((ki >> 2) * 8192 + (ki & 3) * 32) >> 4)),
                uint64_t(d_dS + ((ki * 2048) >> 4)), accG, ki != 0);
        pipe.commit();                                // G3
        pipe.wait<0>();                               // G2 + G3 done
        lbi::mma::fence_operand(accU[0]);
        lbi::mma::fence_operand(accU[1]);
        lbi::mma::fence_operand(accDU[0]);
        lbi::mma::fence_operand(accDU[1]);
        lbi::mma::fence_operand(accF);
        lbi::mma::fence_operand(accG);

        // ---- state RMW ----
        if (FIN) {
            // Register pairs (reg, reg+1) are adjacent columns of one row, so
            // the S/dS read-modify-write runs on bf16x2 lanes: half the
            // shared-memory instructions, identical values and rounding.
#pragma unroll
            for (int mi = 0; mi < 2; ++mi)
#pragma unroll
                for (int wb = 0; wb < 32; wb += lbi::mma::ACC_WINDOW) {
                    accU[mi].window(wb);
                    accDU[mi].window(wb);
#pragma unroll
                    for (int rj = 0; rj < lbi::mma::ACC_WINDOW; rj += 2) {
                        const int reg = wb + rj;
                        int r, q; acc_rc(warp, lane, reg, r, q);
                        const int idx = sw128(128, mi * 64 + r, q);
                        const __nv_bfloat162 s2 =
                            *reinterpret_cast<const __nv_bfloat162*>(S_s + idx);
                        const __nv_bfloat162 ds2 =
                            *reinterpret_cast<const __nv_bfloat162*>(dS_s + idx);
                        const float s_new0 = wc * __low2float(s2) + accU[mi][reg];
                        const float s_new1 = wc * __high2float(s2) + accU[mi][reg + 1];
                        *reinterpret_cast<__nv_bfloat162*>(S_s + idx) =
                            __floats2bfloat162_rn(s_new0, s_new1);
                        *reinterpret_cast<__nv_bfloat162*>(dS_s + idx) =
                            __floats2bfloat162_rn(
                                wc * __low2float(ds2) + dLc * s_new0 + accDU[mi][reg],
                                wc * __high2float(ds2) + dLc * s_new1 + accDU[mi][reg + 1]);
                    }
                }
        } else {
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
#pragma unroll
            for (int wb = 0; wb < 32; wb += lbi::mma::ACC_WINDOW) {
                accU[mi].window(wb);
                accDU[mi].window(wb);
#pragma unroll
                for (int rj = 0; rj < lbi::mma::ACC_WINDOW; ++rj) {
                    const int reg = wb + rj;
                    int r, q; acc_rc(warp, lane, reg, r, q);
                    const int idx = sw128(128, mi * 64 + r, q);
                    const float s_new = wc * __bfloat162float(S_s[idx]) + accU[mi][reg];
                    S_s[idx] = __float2bfloat16(s_new);
                    dS_s[idx] = __float2bfloat16(
                        wc * __bfloat162float(dS_s[idx]) + dLc * s_new + accDU[mi][reg]);
                }
            }
        }

        // ---- G4: wd@v + wd@dvc ----
        lbi::mma::Accum<32> accH, accI;
        lbi::mma::fence_operand(accH);
        lbi::mma::fence_operand(accI);
        lbi::mma::arrive();
#pragma unroll
        for (int ki = 0; ki < 2; ++ki)
            lbi::mma::mma_ss<64, 64, 16, false, true>(
                uint64_t(d_wd + ((ki * 32) >> 4)),
                uint64_t(d_v + ((ki * 512) >> 4)), accH, ki != 0);
#pragma unroll
        for (int ki = 0; ki < 2; ++ki)
            lbi::mma::mma_ss<64, 64, 16, false, true>(
                uint64_t(d_wd + ((ki * 32) >> 4)),
                uint64_t(d_dvc + ((ki * 512) >> 4)), accI, ki != 0);
        pipe.commit();                                // G4
        pipe.wait<0>();
        lbi::mma::fence_operand(accH);
        lbi::mma::fence_operand(accI);

        // ---- epilogue ----
        // block-1: cross-warp dout row assembly through fp32 smem
        if (warp >= 2) {
#pragma unroll
            for (int wb = 0; wb < 32; wb += lbi::mma::ACC_WINDOW) {
                accF.window(wb);
                accH.window(wb);
            #pragma unroll
                for (int rj = 0; rj < lbi::mma::ACC_WINDOW; ++rj) {
                    const int reg = wb + rj;
                    int r, q; acc_rc(warp, lane, reg, r, q);
                    dob_s[(r - 32) * 64 + q] = accH[reg] + e_s[r - 32] * accF[reg];
                }
            }
        }
        __syncthreads();                               // sync 5
        // Stage rows in smem so the global stores below are coalesced
        // uint4 lines; values are bitwise identical to direct stores.
        if (warp < 2) {
#pragma unroll
            for (int reg = 0; reg < 32; reg += 2) {
                int r, q; acc_rc(warp, lane, reg, r, q);
                const float o0 = accH[reg] + e_s[r] * accF[reg];
                const float o1 = accH[reg + 1] + e_s[r] * accF[reg + 1];
                float do0 = dL_s[r] * o0 + accI[reg] + e_s[r] * accG[reg]
                            + dob_s[r * 64 + q];
                float do1 = dL_s[r] * o1 + accI[reg + 1] + e_s[r] * accG[reg + 1]
                            + dob_s[r * 64 + q + 1];
                float of0 = o0, of1 = o1;
                if (FIN) {
                    // D-skip, QK-dot skip, Z-gate on the primal and tangent
                    // rows; z/dz read from global (no smem footprint change).
                    const float vv0 = __bfloat162float(v_s[sw32(32, r, q)]);
                    const float vv1 = __bfloat162float(v_s[sw32(32, r, q + 1)]);
                    const float dvv0 = __bfloat162float(dvc_s[sw32(32, r, q)]) + dL_s[r] * vv0;
                    const float dvv1 = __bfloat162float(dvc_s[sw32(32, r, q + 1)]) + dL_s[r] * vv1;
                    const float p0 = o0 + dsk * vv0 - vv0 * qk_s[r];
                    const float p1 = o1 + dsk * vv1 - vv1 * qk_s[r];
                    const float dp0 = do0 + dsk * dvv0 - dvv0 * qk_s[r] - vv0 * dqk_s[r];
                    const float dp1 = do1 + dsk * dvv1 - dvv1 * qk_s[r] - vv1 * dqk_s[r];
                    // z/dz staged after sync 1; this thread owns (r, q..q+1)
                    // in both buffers and overwrites them below.
                    const __nv_bfloat162 z2 =
                        *reinterpret_cast<const __nv_bfloat162*>(out_st + r * STG_STRIDE + q);
                    const __nv_bfloat162 dz2 =
                        *reinterpret_cast<const __nv_bfloat162*>(dout_st + r * STG_STRIDE + q);
                    const float z0 = __low2float(z2), z1 = __high2float(z2);
                    const float dz0 = __low2float(dz2), dz1 = __high2float(dz2);
                    const float sg0 = 1.f / (1.f + __expf(-z0));
                    const float sg1 = 1.f / (1.f + __expf(-z1));
                    const float g0 = z0 * sg0, g1 = z1 * sg1;
                    const float dg0 = sg0 * (1.f + z0 * (1.f - sg0)) * dz0;
                    const float dg1 = sg1 * (1.f + z1 * (1.f - sg1)) * dz1;
                    of0 = p0 * g0; of1 = p1 * g1;
                    do0 = dp0 * g0 + p0 * dg0; do1 = dp1 * g1 + p1 * dg1;
                }
                if (lane_i == 0)
                    *reinterpret_cast<__nv_bfloat162*>(out_st + r * STG_STRIDE + q) =
                        __floats2bfloat162_rn(of0, of1);
                *reinterpret_cast<__nv_bfloat162*>(dout_st + r * STG_STRIDE + q) =
                    __floats2bfloat162_rn(do0, do1);
            }
        }
        __syncthreads();                               // sync 6
        for (int idx = tid; idx < GCS * 8; idx += 128) {
            const int i = idx >> 3, u = idx & 7;
            *reinterpret_cast<uint4*>(DOUT + lO + vrow + (long)i * H_ * PP + u * 8) =
                *reinterpret_cast<const uint4*>(dout_st + i * STG_STRIDE + u * 8);
            if (lane_i == 0)
                *reinterpret_cast<uint4*>(OUT + vrow + (long)i * H_ * PP + u * 8) =
                    *reinterpret_cast<const uint4*>(out_st + i * STG_STRIDE + u * 8);
        }
        // No chunk-end barrier: the copy reads only the staging rows, and
        // their next writers sit behind the next chunk's earlier barriers.
    
    }

}

}  // namespace lbi_recurrence

#define GW_CHECK(t, d)                                                        \
    TORCH_CHECK((t).is_cuda() && (t).is_contiguous() &&                       \
                (t).scalar_type() == (d), #t " must be CUDA contiguous " #d)

struct GwDims { int B, Sp, H, lanes, G, Da; };

// Checks the 13 field tensors both entry points share and returns the dims.
static GwDims gw_common_checks(
    const torch::Tensor& QR, const torch::Tensor& KR, const torch::Tensor& V,
    const torch::Tensor& DQRAW, const torch::Tensor& DKRAW,
    const torch::Tensor& DV, const torch::Tensor& DTHETA,
    const torch::Tensor& COS, const torch::Tensor& SIN,
    const torch::Tensor& SCALE, const torch::Tensor& DSCALE,
    const torch::Tensor& L, const torch::Tensor& DL) {
    using namespace lbi_recurrence;
    const auto bf = torch::kBFloat16;
    const auto f32 = torch::kFloat32;
    GW_CHECK(QR, bf); GW_CHECK(KR, bf); GW_CHECK(V, bf);
    GW_CHECK(DQRAW, bf); GW_CHECK(DKRAW, bf); GW_CHECK(DV, bf);
    GW_CHECK(DTHETA, bf); GW_CHECK(COS, bf); GW_CHECK(SIN, bf);
    GW_CHECK(SCALE, f32); GW_CHECK(DSCALE, f32);
    GW_CHECK(L, f32); GW_CHECK(DL, f32);
    GwDims d{(int)QR.size(0), (int)QR.size(1), (int)QR.size(2),
             (int)DQRAW.size(0), (int)DQRAW.size(3), (int)COS.size(3)};
    TORCH_CHECK(QR.size(3) == NN && V.size(3) == PP && d.Sp % GCS == 0);
    TORCH_CHECK(d.Da <= NN / 2 && d.Da % 4 == 0 && d.H % d.G == 0);
    return d;
}

std::vector<torch::Tensor> recurrence_full(
    torch::Tensor QR, torch::Tensor KR, torch::Tensor V,
    torch::Tensor DQRAW, torch::Tensor DKRAW, torch::Tensor DV,
    torch::Tensor DTHETA, torch::Tensor COS, torch::Tensor SIN,
    torch::Tensor SCALE, torch::Tensor DSCALE,
    torch::Tensor L, torch::Tensor DL) {
    using namespace lbi_recurrence;
    const GwDims d = gw_common_checks(QR, KR, V, DQRAW, DKRAW, DV, DTHETA,
                                      COS, SIN, SCALE, DSCALE, L, DL);
    const int B = d.B, Sp = d.Sp, H = d.H;
    const int lanes = d.lanes, G = d.G, Da = d.Da;
    // bf16 outputs: the finalize casts immediately; fp32 emission was
    // ~576 MB/step of traffic with no consumer.
    auto OUT = torch::empty({B, Sp, H, PP}, V.options());
    auto DOUT = torch::empty({lanes, B, Sp, H, PP}, V.options());
    cudaFuncSetAttribute(recurrence_jvp_kernel<false>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MIN);
    recurrence_jvp_kernel<false><<<dim3(B * H * lanes), 128, SMEM_MIN,
                               c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const bf16*>(QR.data_ptr()),
        reinterpret_cast<const bf16*>(KR.data_ptr()),
        reinterpret_cast<const bf16*>(V.data_ptr()),
        reinterpret_cast<const bf16*>(DQRAW.data_ptr()),
        reinterpret_cast<const bf16*>(DKRAW.data_ptr()),
        reinterpret_cast<const bf16*>(DV.data_ptr()),
        reinterpret_cast<const bf16*>(DTHETA.data_ptr()),
        reinterpret_cast<const bf16*>(COS.data_ptr()),
        reinterpret_cast<const bf16*>(SIN.data_ptr()),
        SCALE.data_ptr<float>(), DSCALE.data_ptr<float>(),
        L.data_ptr<float>(), DL.data_ptr<float>(),
        nullptr, nullptr, nullptr, nullptr, nullptr,
        reinterpret_cast<bf16*>(OUT.data_ptr()),
        reinterpret_cast<bf16*>(DOUT.data_ptr()),
        B, H, G, Sp, Da, Sp / GCS, Sp);
    return {OUT, DOUT};
}

// The finalize (D-skip, QK-dot skip, Z-gate) folded into the epilogue: the
// kernel emits finalized OUT/DOUT from its fp32 accumulators.
std::vector<torch::Tensor> recurrence_full_fin(
    torch::Tensor QR, torch::Tensor KR, torch::Tensor V,
    torch::Tensor DQRAW, torch::Tensor DKRAW, torch::Tensor DV,
    torch::Tensor DTHETA, torch::Tensor COS, torch::Tensor SIN,
    torch::Tensor SCALE, torch::Tensor DSCALE,
    torch::Tensor L, torch::Tensor DL,
    torch::Tensor Z, torch::Tensor DZ,
    torch::Tensor QKD, torch::Tensor DQKD, torch::Tensor DSK) {
    using namespace lbi_recurrence;
    const auto bf = torch::kBFloat16;
    const auto f32 = torch::kFloat32;
    const GwDims d = gw_common_checks(QR, KR, V, DQRAW, DKRAW, DV, DTHETA,
                                      COS, SIN, SCALE, DSCALE, L, DL);
    GW_CHECK(Z, bf); GW_CHECK(DZ, bf);
    GW_CHECK(QKD, f32); GW_CHECK(DQKD, f32); GW_CHECK(DSK, f32);
    const int B = d.B, Sp = d.Sp, H = d.H;
    const int lanes = d.lanes, G = d.G, Da = d.Da;
    TORCH_CHECK(Z.size(1) == Sp && DZ.size(0) == lanes && DZ.size(2) == Sp,
                "Z/DZ must be [B,Sp,H,P] / [lanes,B,Sp,H,P]");
    auto OUT = torch::empty({B, Sp, H, PP}, V.options());
    auto DOUT = torch::empty({lanes, B, Sp, H, PP}, V.options());
    cudaFuncSetAttribute(recurrence_jvp_kernel<true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MIN);
    recurrence_jvp_kernel<true><<<dim3(B * H * lanes), 128, SMEM_MIN,
                                     c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const bf16*>(QR.data_ptr()),
        reinterpret_cast<const bf16*>(KR.data_ptr()),
        reinterpret_cast<const bf16*>(V.data_ptr()),
        reinterpret_cast<const bf16*>(DQRAW.data_ptr()),
        reinterpret_cast<const bf16*>(DKRAW.data_ptr()),
        reinterpret_cast<const bf16*>(DV.data_ptr()),
        reinterpret_cast<const bf16*>(DTHETA.data_ptr()),
        reinterpret_cast<const bf16*>(COS.data_ptr()),
        reinterpret_cast<const bf16*>(SIN.data_ptr()),
        SCALE.data_ptr<float>(), DSCALE.data_ptr<float>(),
        L.data_ptr<float>(), DL.data_ptr<float>(),
        reinterpret_cast<const bf16*>(Z.data_ptr()),
        reinterpret_cast<const bf16*>(DZ.data_ptr()),
        QKD.data_ptr<float>(), DQKD.data_ptr<float>(), DSK.data_ptr<float>(),
        reinterpret_cast<bf16*>(OUT.data_ptr()),
        reinterpret_cast<bf16*>(DOUT.data_ptr()),
        B, H, G, Sp, Da, Sp / GCS, Sp);
    return {OUT, DOUT};
}
