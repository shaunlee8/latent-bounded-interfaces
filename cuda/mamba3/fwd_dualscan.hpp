// Hand-CUDA forward dual-scan passes, chunked lane-grid decomposition;
// semantics frozen by the chunked lane-grid tilelang kernels in
// mamba3_siso_dualscan.py.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace lbi_mamba3 {

struct FwdDualscanParams {
    int batch;       // B
    int seqlen;      // S (padded, multiple of chunk_size)
    int heads;       // H
    int groups;      // G (qk groups; head h reads group h / (H/G))
    int d_state;     // N
    int n_rot;       // Da (rotary angle count; pairs beyond Da are identity)
    int headdim;     // P
    int chunk_size;  // cs
    int lanes;       // R tangent lanes
    int s_true;      // unpadded sequence length (mean mask)

    // primal operands (bf16)
    const __nv_bfloat16* QR;      // [B,S,H,N] rotated Q
    const __nv_bfloat16* KR;      // [B,S,H,N] rotated UNSCALED K
    const __nv_bfloat16* V;       // [B,S,H,P]
    const __nv_bfloat16* COS;     // [B,H,S,Da]
    const __nv_bfloat16* SIN;     // [B,H,S,Da]
    const __nv_bfloat16* Z;       // [B,S,H,P] gate primal (mean variant)
    // per-lane raw tangents (bf16)
    const __nv_bfloat16* DQRAW;   // [R,B,S,G,N]
    const __nv_bfloat16* DKRAW;   // [R,B,S,G,N]
    const __nv_bfloat16* DV;      // [R,B,S,H,P]
    const __nv_bfloat16* DTHETA;  // [R,B,H,S,Da] cumsum'd tangent phase
    const __nv_bfloat16* DZ;      // [R,B,S,H,P] (mean variant)
    // native fold: bulk tangents read unpadded as strided views (strides in
    // elements; DQRAW/DKRAW share one layout, DV/DZ the other).
    int native;            // 0 emitted/padded, 1 native strided views
    int nat_qk32;          // dq/dk pair: 0 = bf16 (DQRAW/DKRAW), 1 = fp32 (*32)
    int nat_vz32;          // dv/dz pair: 0 = bf16 (DV/DZ), 1 = fp32 (*32)
    const float* DQRAW32;  // [R,B,s_true,G,N] (fp32 pair only)
    const float* DKRAW32;
    const float* DV32;     // [R,B,s_true,H,P] (fp32 pair only)
    const float* DZ32;
    long qk_sl, qk_sb, qk_ss, qk_sg;  // dq/dk strides (lane, batch, seq, group)
    long vz_sl, vz_sb, vz_ss, vz_sh;  // dv/dz strides (lane, batch, seq, head)
    // scalar fields (f32)
    const float* SCALE;    // [B,H,S]
    const float* DSCALE;   // [R,B,H,S]
    const float* L;        // [B,H,S] per-chunk log-decay cumsum
    const float* DL;       // [R,B,H,S]
    const float* QKDOT;    // [B,H,S]   (mean variant)
    const float* DQKDOT;   // [R,B,H,S] (mean variant)
    const float* DSKIP;    // [H]       (mean variant)
    // pass-B states (f32 for the simple kernels; bf16 for the opt kernels,
    // read directly from gmem as wmma operands)
    const float* S_IN;     // [B,H,nc,N,P]
    const float* DS_IN;    // [R,B,H,nc,N,P]
    const __nv_bfloat16* S_INB;
    const __nv_bfloat16* DS_INB;
    // outputs (f32, simple kernels)
    float* SC;    // [B,H,nc,N,P]    (pass A primal)
    float* DSC;   // [R,B,H,nc,N,P]  (pass A tangent)
    float* OUT;   // [B,S,H,P]       (pass C, lane 0 writes)
    float* DOUT;  // [R,B,S,H,P]     (pass C)
    float* PART;  // [R,B,H,nc,P]    (pass C mean)
    // bf16 state stream: pass A opt writes SCB/DSCB; pass B opt reads
    // them and writes SB_OUT/DSB_OUT (which pass C opt reads as S_INB/DS_INB)
    __nv_bfloat16* SCB;     // [B,H,nc,N,P]
    __nv_bfloat16* DSCB;    // [R,B,H,nc,N,P]
    __nv_bfloat16* SB_OUT;  // [B,H,nc,N,P]
    __nv_bfloat16* DSB_OUT; // [R,B,H,nc,N,P]
    // chunk-parallel ticket chain: per-chunk OUTGOING states + ready
    // flags; SB_OUT/DSB_OUT double as the chain payload ([R,B,H,nc,N,P]).
    int* FLAGS;             // [R,B,H,nc], zero-initialized per call
};

void launch_fwd_passA_primal(const FwdDualscanParams& p, cudaStream_t stream);
void launch_fwd_passA_tan(const FwdDualscanParams& p, cudaStream_t stream);
void launch_fwd_passC(const FwdDualscanParams& p, cudaStream_t stream);
void launch_fwd_passC_mean(const FwdDualscanParams& p, cudaStream_t stream);

// Tensor-core rewrites (require cs, N, P % 16 == 0, N >= 2*P, P <= cs;
// the opt pass A kernels emit bf16 SCB/DSCB).
void launch_fwd_passA_primal_opt(const FwdDualscanParams& p, cudaStream_t stream);
void launch_fwd_passA_tan_opt(const FwdDualscanParams& p, cudaStream_t stream);
void launch_fwd_passB_opt(const FwdDualscanParams& p, cudaStream_t stream);
void launch_fwd_passC_opt(const FwdDualscanParams& p, cudaStream_t stream);
void launch_fwd_passC_mean_opt(const FwdDualscanParams& p, cudaStream_t stream);

// Fused A+B+C(mean) chunk walk: one CTA per (lane,h,b), running
// states resident in smem, no gmem state stream, single operand read.
void launch_fwd_fused_mean(const FwdDualscanParams& p, cudaStream_t stream);

// Chunk-parallel fused walk: one CTA per (lane,h,b,chunk), inter-chunk
// state chained through L2 via atomic tickets, publish-early.
void launch_fwd_r8a_mean(const FwdDualscanParams& p, cudaStream_t stream);

}  // namespace lbi_mamba3
