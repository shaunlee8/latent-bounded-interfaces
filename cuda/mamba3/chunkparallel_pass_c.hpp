#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace lbi_mamba3 {

// Pass C of the chunk-parallel P-batched SISO backward; all tensors
// contiguous, layouts per the tilelang oracle mamba_chunkparallel_bwd_bwd.
struct ChunkParallelPassCParams {
    // sizes
    int batch, seqlen, heads, groups, d_state, headdim, lanes, nchunks, chunk_size, n_rot;
    // inputs (bf16 unless noted)
    const __nv_bfloat16* DOUT;     // [B, LG, S, H, P]
    const __nv_bfloat16* GATE;     // [B, S, H, P] silu(z), lane-free
    const __nv_bfloat16* Q;        // [B, S, 1, G, N]
    const __nv_bfloat16* K;        // [B, S, 1, G, N]
    const __nv_bfloat16* V;        // [B, S, H, P]
    const float* Q_BIAS;           // [H, 1, N]
    const float* K_BIAS;           // [H, 1, N]
    const __nv_bfloat16* STATES;   // [B, H, nch, N, P]
    const __nv_bfloat16* DSTATES_IN;  // [B, LG, H, nch, N, P]
    const float* ANGLES;           // [B, S, H, n_rot] cumulative
    const float* DA_CS;            // [B, H, S]
    const float* DA_CS_REV;        // [B, H, S]
    const float* DT;               // [B, H, S]
    const __nv_bfloat16* TRAP;     // [B, H, S]
    const float* D;                // [H]
    const __nv_bfloat16* QK_DOT;   // [B, H, S, 1, 1]
    const float* SEGSUM;           // [B, H, nch, cs, cs]
    // outputs
    float* DK;                     // [B, LG, S, N] fp32, ZEROED, atomic head-sum
    float* DQ;                     // [B, LG, S, N] fp32, ZEROED, atomic head-sum
    __nv_bfloat16* DV;             // [B, LG, S, H, P]
    float* DANGLES;                // [B, LG, S, H, n_rot]
    float* DFACTOR;                // [B, LG, H, S]
    float* DGAMMA_DIAG;            // [B, LG, H, S]
    float* DD;                     // [B, LG, H, nch]
    float* DSSDA;                  // [B, LG, H, nch, cs, cs]
    float* DDA_CS_REV;             // [B, LG, H, S]
    float* DDA_CS;                 // [B, LG, H, S]
};

void launch_chunkparallel_pass_c_simple(const ChunkParallelPassCParams& params,
                                        cudaStream_t stream);

void launch_chunkparallel_pass_c_mma(const ChunkParallelPassCParams& params,
                                     cudaStream_t stream);

}  // namespace lbi_mamba3
