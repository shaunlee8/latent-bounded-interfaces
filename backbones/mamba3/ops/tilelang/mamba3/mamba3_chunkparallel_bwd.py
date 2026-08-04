"""Chunk-PARALLEL P-batched SISO backward (pass C of the three-pass form).

The serial lane-grid kernel is latency-bound (~200 KB smem/CTA -> 1 CTA/SM,
with serial chunks of dependent gemm chains per CTA). This kernel removes the
serial chain by the standard SSD decomposition:

  pass A (harness, torch): chunk-local state-cotangent contributions
      G_c = q_rot_c^T @ (dout_c * exp(dA_cs_c))            [N, P] per chunk
  pass B (harness, torch): tiny inter-chunk reverse scan
      IN_{nch-1} = 0;  IN_c = IN_{c+1} * E_{c+1} + G_{c+1},
      E_c = exp(DA_CS[chunk c total])  -- IN_c is the state cotangent ENTERING
      chunk c from the future (what the serial kernel carried).
  pass C (this kernel): ALL gradient outputs, one CTA per
      (head, batch, lane, chunk), IN_c read from gmem -- no dependencies.

The gradient body is the validated `mamba_lanetile_bwd_bwd` body (R = 1,
PsiV == v, diagonal-only dqk) minus the dstates carry/update. Grid is
(H, B, LG * nchunks); intended chunk_size is 32 so shared memory (~100 KB)
allows 2 CTAs/SM -- at 1 CTA/SM chunk-parallelism would buy nothing, since
the same number of chunk-units would still serialize per SM.

dD is emitted per chunk ([B, LG, H, nch]); the harness discards it (hasD only
contributes the in-kernel D-skip term of dv).
"""

import tilelang
import tilelang.language as T


@tilelang.jit(
    out_idx=[],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    })
def mamba_chunkparallel_bwd_bwd(
    B,
    S,
    H,
    G,
    N,
    P,
    lane_grid,
    hasD,
    chunk_size: int = 32,
    rotary_dim_divisor: int = 4,
    dtype: str = 'bfloat16',
    threads: int = 256,
):
    accum_dtype = 'float32'
    LG = lane_grid
    nchunks = tilelang.cdiv(S, chunk_size)
    n_rot = N // rotary_dim_divisor

    @T.prim_func
    def mamba_chunkparallel_bwd_bwd_kernel(
            DOUT: T.Tensor((B, LG, S, H, P), dtype),  # type: ignore
            GATE: T.Tensor([B, S, H, P], dtype),  # type: ignore
            Q: T.Tensor([B, S, 1, G, N], dtype),  # type: ignore
            K: T.Tensor([B, S, 1, G, N], dtype),  # type: ignore
            V: T.Tensor([B, S, H, P], dtype),  # type: ignore
            Q_BIAS: T.Tensor([H, 1, N], T.float32),  # type: ignore
            K_BIAS: T.Tensor([H, 1, N], T.float32),  # type: ignore
            DK: T.Tensor((B, LG, S, N), T.float32),  # type: ignore
            DV: T.Tensor((B, LG, S, H, P), dtype),  # type: ignore
            STATES: T.Tensor([B, H, nchunks, N, P], dtype),  # type: ignore
            DQ: T.Tensor((B, LG, S, N), T.float32),  # type: ignore
            DSTATES_IN: T.Tensor((B, LG, H, nchunks, N, P), dtype),  # type: ignore
            ANGLES: T.Tensor([B, S, H, n_rot], T.float32),  # type: ignore
            DA_CS: T.Tensor([B, H, S], T.float32),  # type: ignore
            DA_CS_REV: T.Tensor([B, H, S], T.float32),  # type: ignore
            DT: T.Tensor([B, H, S], T.float32),  # type: ignore
            TRAP: T.Tensor([B, H, S], dtype),  # type: ignore
            DFACTOR: T.Tensor((B, LG, H, S), T.float32),  # type: ignore
            DGAMMA_DIAG: T.Tensor((B, LG, H, S), T.float32),  # type: ignore
            DANGLES: T.Tensor((B, LG, S, H, n_rot), T.float32),  # type: ignore
            D: T.Tensor([H], T.float32),  # type: ignore
            DD: T.Tensor((B, LG, H, nchunks), T.float32),  # type: ignore
            QK_DOT: T.Tensor([B, H, S, 1, 1], dtype),  # type: ignore
            DSSDA: T.Tensor((B, LG, H, nchunks, chunk_size, chunk_size), T.float32),  # type: ignore
            DDA_CS_REV: T.Tensor((B, LG, H, S), T.float32),  # type: ignore
            DDA_CS: T.Tensor((B, LG, H, S), T.float32),  # type: ignore
            SEGSUM: T.Tensor([B, H, nchunks, chunk_size, chunk_size], T.float32),  # type: ignore
            ):
        with T.Kernel(H, B, LG * nchunks, threads=threads) as (i_h, i_b, i_lc):
            i_h_qk = i_h // (H // G)
            lane = i_lc // nchunks
            chunk_idx = i_lc % nchunks
            chunk_start = chunk_idx * chunk_size

            # --- Shared buffers (diet round 2: ~70 KB at cs=32/N=128 -> 3
            # CTAs/SM). q/k rotate IN PLACE; pre-rotary values are re-derived
            # from gmem + bias where needed (exact); dfactor divides trap_scale
            # out of the reduced sum instead of keeping an unscaled k copy
            # (trap_scale = gamma + shifted_gamma > 0 strictly); ONE stage
            # buffer serves dk then dq (dk stores before the dq section). ---
            q_shared = T.alloc_shared([chunk_size, N], dtype)      # biased -> rotated
            k_shared = T.alloc_shared([chunk_size, N], dtype)      # biased -> rotated -> trap-scaled
            v_shared = T.alloc_shared([chunk_size, P], dtype)
            states_shared = T.alloc_shared([N, P], dtype)
            lkq_unmasked_shared = T.alloc_shared([chunk_size, chunk_size], dtype)
            lkq_masked_shared = T.alloc_shared([chunk_size, chunk_size], dtype)
            dout_shared = T.alloc_shared([chunk_size, P], dtype)
            dstates_op_shared = T.alloc_shared([N, P], dtype)
            dkq_masked_shared = T.alloc_shared([chunk_size, chunk_size], dtype)
            stage_shared = T.alloc_shared([chunk_size, N], dtype)

            q_bias_shared = T.alloc_shared([1, N], dtype)
            k_bias_shared = T.alloc_shared([1, N], dtype)

            # Swizzled layouts on the GEMM-operand tiles (bank-conflict / L1
            # pressure is the bound here, not compute) + L2 grid rasterization.
            # Skip the tiny [cs,cs] tiles (swizzle can hang small tiles).
            T.annotate_layout({
                q_shared: tilelang.layout.make_swizzled_layout(q_shared),
                k_shared: tilelang.layout.make_swizzled_layout(k_shared),
                v_shared: tilelang.layout.make_swizzled_layout(v_shared),
                states_shared: tilelang.layout.make_swizzled_layout(states_shared),
                dstates_op_shared: tilelang.layout.make_swizzled_layout(dstates_op_shared),
                dout_shared: tilelang.layout.make_swizzled_layout(dout_shared),
                stage_shared: tilelang.layout.make_swizzled_layout(stage_shared),
            })
            T.use_swizzle(10, "row")

            T.copy(Q_BIAS[i_h, :, :], q_bias_shared)
            T.copy(K_BIAS[i_h, :, :], k_bias_shared)

            # ============ PROLOGUE ============
            trap_shifted_frag = T.alloc_fragment([chunk_size], T.float32)
            dt_shifted_frag = T.alloc_fragment([chunk_size], dtype)
            for cs in T.Parallel(chunk_size):
                trap_shifted_frag[cs] = T.if_then_else(
                    chunk_start + cs + 1 < S, TRAP[i_b, i_h, chunk_start + cs + 1], 0.0)
                dt_shifted_frag[cs] = T.if_then_else(
                    chunk_start + cs + 1 < S, DT[i_b, i_h, chunk_start + cs + 1], 0.0)
            shifted_gamma_frag = T.alloc_fragment([chunk_size], dtype)
            for cs in T.Parallel(chunk_size):
                shifted_gamma_frag[cs] = T.if_then_else(
                    chunk_start + cs < (S - 1),
                    dt_shifted_frag[cs] * T.sigmoid(-trap_shifted_frag[cs]), 0.0)
            trap_frag = T.alloc_fragment([chunk_size], T.float32)
            T.copy(TRAP[i_b, i_h, chunk_start: chunk_start + chunk_size], trap_frag)
            dt_frag = T.alloc_fragment([chunk_size], dtype)
            T.copy(DT[i_b, i_h, chunk_start: chunk_start + chunk_size], dt_frag)
            gamma_frag = T.alloc_fragment([chunk_size], T.float32)
            for cs in T.Parallel(chunk_size):
                gamma_frag[cs] = dt_frag[cs] * T.sigmoid(trap_frag[cs])
            trap_scale_frag = T.alloc_fragment([chunk_size], dtype)
            for cs in T.Parallel(chunk_size):
                trap_scale_frag[cs] = gamma_frag[cs] + shifted_gamma_frag[cs]
            trap_scale_shared = T.alloc_shared([chunk_size], dtype)
            T.copy(trap_scale_frag, trap_scale_shared)

            dA_cs_rev_shared = T.alloc_shared([chunk_size], T.float32)
            T.copy(DA_CS_REV[i_b, i_h, chunk_start:chunk_start + chunk_size], dA_cs_rev_shared)
            exp_dA_cs_rev_frag = T.alloc_fragment([chunk_size], T.float32)
            T.copy(dA_cs_rev_shared, exp_dA_cs_rev_frag)
            for cs in T.Parallel(chunk_size):
                exp_dA_cs_rev_frag[cs] = T.exp(exp_dA_cs_rev_frag[cs])
            dA_cs_shared = T.alloc_shared([chunk_size], T.float32)
            T.copy(DA_CS[i_b, i_h, chunk_start:chunk_start + chunk_size], dA_cs_shared)

            angles_frag = T.alloc_fragment([chunk_size, n_rot], T.float32)
            T.copy(ANGLES[i_b, chunk_start:chunk_start + chunk_size, i_h, :], angles_frag)

            # q: load + bias -> q_shared; rotate IN PLACE (halves via shared).
            q_frag = T.alloc_fragment([chunk_size, N], dtype)
            for cs, n in T.Parallel(chunk_size, N):
                q_frag[cs, n] = Q[i_b, chunk_start + cs, 0, i_h_qk, n] + q_bias_shared[0, n]
            T.copy(q_frag, q_shared)
            q_h1_frag = T.alloc_fragment([chunk_size, n_rot], dtype)
            q_h2_frag = T.alloc_fragment([chunk_size, n_rot], dtype)
            for cs, n in T.Parallel(chunk_size, n_rot):
                q_h1_frag[cs, n] = q_shared[cs, n]
                q_h2_frag[cs, n] = q_shared[cs, N // 2 + n]
            for cs, n in T.Parallel(chunk_size, n_rot):
                q_shared[cs, n] = (T.cos(angles_frag[cs, n]) * q_h1_frag[cs, n]
                                   - T.sin(angles_frag[cs, n]) * q_h2_frag[cs, n])
                q_shared[cs, N // 2 + n] = (T.sin(angles_frag[cs, n]) * q_h1_frag[cs, n]
                                            + T.cos(angles_frag[cs, n]) * q_h2_frag[cs, n])

            # k: load + bias -> k_shared; rotate IN PLACE; trap-scale IN PLACE.
            k_frag = T.alloc_fragment([chunk_size, N], dtype)
            for cs, n in T.Parallel(chunk_size, N):
                k_frag[cs, n] = K[i_b, chunk_start + cs, 0, i_h_qk, n] + k_bias_shared[0, n]
            T.copy(k_frag, k_shared)
            k_h1_frag = T.alloc_fragment([chunk_size, n_rot], dtype)
            k_h2_frag = T.alloc_fragment([chunk_size, n_rot], dtype)
            for cs, n in T.Parallel(chunk_size, n_rot):
                k_h1_frag[cs, n] = k_shared[cs, n]
                k_h2_frag[cs, n] = k_shared[cs, N // 2 + n]
            for cs, n in T.Parallel(chunk_size, n_rot):
                k_shared[cs, n] = (T.cos(angles_frag[cs, n]) * k_h1_frag[cs, n]
                                   - T.sin(angles_frag[cs, n]) * k_h2_frag[cs, n])
                k_shared[cs, N // 2 + n] = (T.sin(angles_frag[cs, n]) * k_h1_frag[cs, n]
                                            + T.cos(angles_frag[cs, n]) * k_h2_frag[cs, n])
            for cs, n in T.Parallel(chunk_size, N):
                k_shared[cs, n] = k_shared[cs, n] * trap_scale_shared[cs]

            T.copy(V[i_b, chunk_start:chunk_start + chunk_size, i_h, :], v_shared)

            # lkq = k_scaled @ q_rot^T; unmasked saved; masked with segsum.
            lkq_frag = T.alloc_fragment([chunk_size, chunk_size], accum_dtype)
            T.gemm(k_shared, q_shared, lkq_frag, transpose_B=True, clear_accum=True)
            T.copy(lkq_frag, lkq_unmasked_shared)
            lkq_masked_frag = T.alloc_fragment([chunk_size, chunk_size], dtype)
            T.copy(lkq_frag, lkq_masked_frag)
            for cs_i, cs_j in T.Parallel(chunk_size, chunk_size):
                lkq_masked_frag[cs_i, cs_j] = T.if_then_else(
                    cs_i < cs_j,
                    lkq_masked_frag[cs_i, cs_j] * T.exp(SEGSUM[i_b, i_h, chunk_idx, cs_j, cs_i]),
                    0.0)
            T.copy(lkq_masked_frag, lkq_masked_shared)

            qk_dot_frag = T.alloc_fragment([chunk_size], dtype)
            for cs in T.Parallel(chunk_size):
                qk_dot_frag[cs] = QK_DOT[i_b, i_h, chunk_start + cs, 0, 0]

            T.copy(STATES[i_b, i_h, chunk_idx, :, :], states_shared)
            T.copy(DSTATES_IN[i_b, lane, i_h, chunk_idx, :, :], dstates_op_shared)

            # Z-gate VJP fused into the load: dout *= silu(z) (GATE is
            # lane-free; replaces the separate compute_dzdo pass).
            for cs, p in T.Parallel(chunk_size, P):
                dout_shared[cs, p] = (DOUT[i_b, lane, chunk_start + cs, i_h, p]
                                      * GATE[i_b, chunk_start + cs, i_h, p])
            dout_frag = T.alloc_fragment([chunk_size, P], dtype)
            T.copy(dout_shared, dout_frag)
            if hasD:
                D_var = T.alloc_var(T.float32)
                T.copy(D[i_h], D_var)
                dD_tmp_frag = T.alloc_fragment([1], accum_dtype)
                tmp_csP0 = T.alloc_fragment([chunk_size, P], accum_dtype)
                for cs, p in T.Parallel(chunk_size, P):
                    tmp_csP0[cs, p] = dout_frag[cs, p] * v_shared[cs, p]
                T.reduce_sum(T.view(tmp_csP0, shape=[chunk_size * P]), dD_tmp_frag, clear=True)
                DD[i_b, lane, i_h, chunk_idx] = dD_tmp_frag[0]

            # dPsiV / dv.
            dPsiV_frag = T.alloc_fragment([chunk_size, P], accum_dtype)
            T.gemm(k_shared, dstates_op_shared, dPsiV_frag, clear_accum=True)
            for cs, p in T.Parallel(chunk_size, P):
                dPsiV_frag[cs, p] *= exp_dA_cs_rev_frag[cs]
            T.gemm(lkq_masked_shared, dout_shared, dPsiV_frag, clear_accum=False)
            if hasD:
                for cs, p in T.Parallel(chunk_size, P):
                    dPsiV_frag[cs, p] += dout_frag[cs, p] * D_var
            for cs, p in T.Parallel(chunk_size, P):
                dPsiV_frag[cs, p] += dout_frag[cs, p] * qk_dot_frag[cs] * gamma_frag[cs]
            dv_out_frag = T.alloc_fragment([chunk_size, P], dtype)
            T.copy(dPsiV_frag, dv_out_frag)
            T.copy(dv_out_frag, DV[i_b, lane, chunk_start:chunk_start + chunk_size, i_h, :])

            # Diagonal qk correction (R=1).
            dqk_diag_pre_frag = T.alloc_fragment([chunk_size, P], accum_dtype)
            for cs, p in T.Parallel(chunk_size, P):
                dqk_diag_pre_frag[cs, p] = dout_frag[cs, p] * v_shared[cs, p]
            dqk_diag_frag = T.alloc_fragment([chunk_size], accum_dtype)
            T.reduce_sum(dqk_diag_pre_frag, dqk_diag_frag, dim=-1, clear=True)
            dgamma_diag_frag = T.alloc_fragment([chunk_size], accum_dtype)
            for cs in T.Parallel(chunk_size):
                dgamma_diag_frag[cs] = qk_dot_frag[cs] * dqk_diag_frag[cs]
            T.copy(dgamma_diag_frag, DGAMMA_DIAG[i_b, lane, i_h, chunk_start:chunk_start + chunk_size])
            dqk_diag_gamma_frag = T.alloc_fragment([chunk_size], accum_dtype)
            for cs in T.Parallel(chunk_size):
                dqk_diag_gamma_frag[cs] = dqk_diag_frag[cs] * gamma_frag[cs]

            # dk interchunk (PsiV == v); ddA_cs_rev before scaling.
            acc_csN = T.alloc_fragment([chunk_size, N], accum_dtype)
            tmp_csN = T.alloc_fragment([chunk_size, N], accum_dtype)
            red_cs = T.alloc_fragment([chunk_size], accum_dtype)
            T.gemm(v_shared, dstates_op_shared, acc_csN, transpose_B=True, clear_accum=True)
            for cs, n in T.Parallel(chunk_size, N):
                tmp_csN[cs, n] = k_shared[cs, n] * acc_csN[cs, n]
            T.reduce_sum(tmp_csN, red_cs, dim=-1, clear=True)
            T.copy(red_cs, DDA_CS_REV[i_b, lane, i_h, chunk_start:chunk_start + chunk_size])
            for cs, n in T.Parallel(chunk_size, N):
                acc_csN[cs, n] *= exp_dA_cs_rev_frag[cs]

            # dkq intrachunk; DSSDA; mask.
            dkq_frag = T.alloc_fragment([chunk_size, chunk_size], accum_dtype)
            T.gemm(v_shared, dout_shared, dkq_frag, transpose_B=True, clear_accum=True)
            dssda_frag = T.alloc_fragment([chunk_size, chunk_size], accum_dtype)
            for cs_i, cs_j in T.Parallel(chunk_size, chunk_size):
                dssda_frag[cs_i, cs_j] = lkq_unmasked_shared[cs_i, cs_j] * dkq_frag[cs_i, cs_j]
            T.copy(dssda_frag, DSSDA[i_b, lane, i_h, chunk_idx, :, :])
            dkq_masked_frag = T.alloc_fragment([chunk_size, chunk_size], dtype)
            T.copy(dkq_frag, dkq_masked_frag)
            for cs_i, cs_j in T.Parallel(chunk_size, chunk_size):
                dkq_masked_frag[cs_i, cs_j] = T.if_then_else(
                    cs_i < cs_j,
                    dkq_masked_frag[cs_i, cs_j] * T.exp(SEGSUM[i_b, i_h, chunk_idx, cs_j, cs_i]),
                    0.0)
            T.copy(dkq_masked_frag, dkq_masked_shared)

            # dk_nodiag = dk + dkq_masked @ q_rot; dfactor (trap scale divided
            # out of the reduced sum); trap scale applied to dk.
            # L1: dropped the acc_csN<->stage layout round-trip (test).
            T.gemm(dkq_masked_shared, q_shared, acc_csN, clear_accum=False)
            for cs, n in T.Parallel(chunk_size, N):
                tmp_csN[cs, n] = k_shared[cs, n] * acc_csN[cs, n]
            T.reduce_sum(tmp_csN, red_cs, dim=-1, clear=True)
            for cs in T.Parallel(chunk_size):
                red_cs[cs] = red_cs[cs] / trap_scale_shared[cs]
            T.copy(red_cs, DFACTOR[i_b, lane, i_h, chunk_start:chunk_start + chunk_size])
            for cs, n in T.Parallel(chunk_size, N):
                acc_csN[cs, n] *= trap_scale_shared[cs]

            # dk inverse rotary + dangles + diag correction + DK store (BEFORE
            # the dq section so acc_csN and stage_shared are reclaimed).
            dh1_frag = T.alloc_fragment([chunk_size, n_rot], dtype)
            dh2_frag = T.alloc_fragment([chunk_size, n_rot], dtype)
            pre_h1_frag = T.alloc_fragment([chunk_size, n_rot], dtype)
            pre_h2_frag = T.alloc_fragment([chunk_size, n_rot], dtype)
            dangle_frag = T.alloc_fragment([chunk_size, n_rot], T.float32)
            T.copy(acc_csN, stage_shared)
            for cs, n in T.Parallel(chunk_size, n_rot):
                dh1_frag[cs, n] = stage_shared[cs, n]
                dh2_frag[cs, n] = stage_shared[cs, N // 2 + n]
                pre_h1_frag[cs, n] = K[i_b, chunk_start + cs, 0, i_h_qk, n] + k_bias_shared[0, n]
                pre_h2_frag[cs, n] = K[i_b, chunk_start + cs, 0, i_h_qk, N // 2 + n] + k_bias_shared[0, N // 2 + n]
            for cs, n in T.Parallel(chunk_size, n_rot):
                dangle_frag[cs, n] = (
                    dh1_frag[cs, n] * (-pre_h1_frag[cs, n] * T.sin(angles_frag[cs, n])
                                       - pre_h2_frag[cs, n] * T.cos(angles_frag[cs, n]))
                    + dh2_frag[cs, n] * (pre_h1_frag[cs, n] * T.cos(angles_frag[cs, n])
                                         - pre_h2_frag[cs, n] * T.sin(angles_frag[cs, n])))
            for cs, n in T.Parallel(chunk_size, n_rot):
                stage_shared[cs, n] = (T.cos(angles_frag[cs, n]) * dh1_frag[cs, n]
                                       + T.sin(angles_frag[cs, n]) * dh2_frag[cs, n])
                stage_shared[cs, N // 2 + n] = (-T.sin(angles_frag[cs, n]) * dh1_frag[cs, n]
                                                + T.cos(angles_frag[cs, n]) * dh2_frag[cs, n])
            # head-sum in-kernel: fp32 atomicAdd across the i_h grid axis
            # (G=1 -> dK_total = sum_h dk_h); also MORE precise than summing
            # bf16 per-head stores, and kills the [B,LG,S,H,N] materialization.
            for cs, n in T.Parallel(chunk_size, N):
                tmp_csN[cs, n] = stage_shared[cs, n] + dqk_diag_gamma_frag[cs] * (
                    Q[i_b, chunk_start + cs, 0, i_h_qk, n] + q_bias_shared[0, n])
            T.atomic_add(DK[i_b, lane, chunk_start:chunk_start + chunk_size, :], tmp_csN)

            # dq interchunk; ddA_cs; scale; + dkq_masked^T @ k_scaled.
            T.gemm(dout_shared, states_shared, acc_csN, transpose_B=True, clear_accum=True)
            for cs, n in T.Parallel(chunk_size, N):
                tmp_csN[cs, n] = q_shared[cs, n] * acc_csN[cs, n]
            T.reduce_sum(tmp_csN, red_cs, dim=-1, clear=True)
            T.copy(red_cs, DDA_CS[i_b, lane, i_h, chunk_start:chunk_start + chunk_size])
            for cs, n in T.Parallel(chunk_size, N):
                acc_csN[cs, n] *= T.exp(dA_cs_shared[cs])
            # L1: dropped the acc_csN<->stage layout round-trip (test).
            T.gemm(dkq_masked_shared, k_shared, acc_csN, transpose_A=True, clear_accum=False)

            # dq inverse rotary + dangles + diag correction + DQ store.
            T.copy(acc_csN, stage_shared)
            for cs, n in T.Parallel(chunk_size, n_rot):
                dh1_frag[cs, n] = stage_shared[cs, n]
                dh2_frag[cs, n] = stage_shared[cs, N // 2 + n]
                pre_h1_frag[cs, n] = Q[i_b, chunk_start + cs, 0, i_h_qk, n] + q_bias_shared[0, n]
                pre_h2_frag[cs, n] = Q[i_b, chunk_start + cs, 0, i_h_qk, N // 2 + n] + q_bias_shared[0, N // 2 + n]
            for cs, n in T.Parallel(chunk_size, n_rot):
                dangle_frag[cs, n] += (
                    dh1_frag[cs, n] * (-pre_h1_frag[cs, n] * T.sin(angles_frag[cs, n])
                                       - pre_h2_frag[cs, n] * T.cos(angles_frag[cs, n]))
                    + dh2_frag[cs, n] * (pre_h1_frag[cs, n] * T.cos(angles_frag[cs, n])
                                         - pre_h2_frag[cs, n] * T.sin(angles_frag[cs, n])))
            T.copy(dangle_frag, DANGLES[i_b, lane, chunk_start:chunk_start + chunk_size, i_h, :])
            for cs, n in T.Parallel(chunk_size, n_rot):
                stage_shared[cs, n] = (T.cos(angles_frag[cs, n]) * dh1_frag[cs, n]
                                       + T.sin(angles_frag[cs, n]) * dh2_frag[cs, n])
                stage_shared[cs, N // 2 + n] = (-T.sin(angles_frag[cs, n]) * dh1_frag[cs, n]
                                                + T.cos(angles_frag[cs, n]) * dh2_frag[cs, n])
            for cs, n in T.Parallel(chunk_size, N):
                tmp_csN[cs, n] = stage_shared[cs, n] + dqk_diag_gamma_frag[cs] * (
                    K[i_b, chunk_start + cs, 0, i_h_qk, n] + k_bias_shared[0, n])
            T.atomic_add(DQ[i_b, lane, chunk_start:chunk_start + chunk_size, :], tmp_csN)

    return mamba_chunkparallel_bwd_bwd_kernel



@tilelang.jit(
    out_idx=[],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    })
def mamba_chunkparallel_bwd_dstates_local(
    B,
    S,
    H,
    G,
    N,
    P,
    lane_grid,
    chunk_size: int = 32,
    rotary_dim_divisor: int = 4,
    dtype: str = 'bfloat16',
    threads: int = 128,
):
    """Pass A: chunk-local state-cotangent contributions
    G_c = q_rot_c^T @ (dout_c * exp(dA_cs_c)), one CTA per
    (head, batch, lane, chunk). Replaces the ~5 ms torch einsum (which also
    materialized a scaled fp32 copy of dout). q-prep replicates the pass-C
    prologue (bias in bf16, half-split partial rotary)."""
    accum_dtype = 'float32'
    LG = lane_grid
    nchunks = tilelang.cdiv(S, chunk_size)
    n_rot = N // rotary_dim_divisor

    @T.prim_func
    def mamba_chunkparallel_bwd_dstates_local_kernel(
            DOUT: T.Tensor((B, LG, S, H, P), dtype),  # type: ignore
            GATE: T.Tensor([B, S, H, P], dtype),  # type: ignore
            Q: T.Tensor([B, S, 1, G, N], dtype),  # type: ignore
            Q_BIAS: T.Tensor([H, 1, N], T.float32),  # type: ignore
            ANGLES: T.Tensor([B, S, H, n_rot], T.float32),  # type: ignore
            DA_CS: T.Tensor([B, H, S], T.float32),  # type: ignore
            G_OUT: T.Tensor((B, LG, H, nchunks, N, P), dtype),  # type: ignore
            ):
        with T.Kernel(H, B, LG * nchunks, threads=threads) as (i_h, i_b, i_lc):
            i_h_qk = i_h // (H // G)
            lane = i_lc // nchunks
            chunk_idx = i_lc % nchunks
            chunk_start = chunk_idx * chunk_size

            q_shared = T.alloc_shared([chunk_size, N], dtype)
            q_stage_shared = T.alloc_shared([chunk_size, N], dtype)
            dout_shared = T.alloc_shared([chunk_size, P], dtype)

            q_bias_frag = T.alloc_fragment([1, N], dtype)
            T.copy(Q_BIAS[i_h, :, :], q_bias_frag)
            q_frag = T.alloc_fragment([chunk_size, N], dtype)
            for cs, n in T.Parallel(chunk_size, N):
                q_frag[cs, n] = Q[i_b, chunk_start + cs, 0, i_h_qk, n] + q_bias_frag[0, n]
            T.copy(q_frag, q_stage_shared)
            T.copy(q_frag, q_shared)
            angles_frag = T.alloc_fragment([chunk_size, n_rot], T.float32)
            T.copy(ANGLES[i_b, chunk_start:chunk_start + chunk_size, i_h, :], angles_frag)
            q_h1_frag = T.alloc_fragment([chunk_size, n_rot], dtype)
            q_h2_frag = T.alloc_fragment([chunk_size, n_rot], dtype)
            for cs, n in T.Parallel(chunk_size, n_rot):
                q_h1_frag[cs, n] = q_stage_shared[cs, n]
                q_h2_frag[cs, n] = q_stage_shared[cs, N // 2 + n]
            for cs, n in T.Parallel(chunk_size, n_rot):
                q_shared[cs, n] = (T.cos(angles_frag[cs, n]) * q_h1_frag[cs, n]
                                   - T.sin(angles_frag[cs, n]) * q_h2_frag[cs, n])
                q_shared[cs, N // 2 + n] = (T.sin(angles_frag[cs, n]) * q_h1_frag[cs, n]
                                            + T.cos(angles_frag[cs, n]) * q_h2_frag[cs, n])

            dA_cs_frag = T.alloc_fragment([chunk_size], T.float32)
            T.copy(DA_CS[i_b, i_h, chunk_start:chunk_start + chunk_size], dA_cs_frag)
            dout_frag = T.alloc_fragment([chunk_size, P], dtype)
            for cs, p in T.Parallel(chunk_size, P):
                dout_frag[cs, p] = (DOUT[i_b, lane, chunk_start + cs, i_h, p]
                                    * GATE[i_b, chunk_start + cs, i_h, p]
                                    * T.exp(dA_cs_frag[cs]))
            T.copy(dout_frag, dout_shared)

            g_frag = T.alloc_fragment([N, P], accum_dtype)
            T.gemm(q_shared, dout_shared, g_frag, transpose_A=True, clear_accum=True)
            T.copy(g_frag, G_OUT[i_b, lane, i_h, chunk_idx, :, :])

    return mamba_chunkparallel_bwd_dstates_local_kernel


@tilelang.jit(
    out_idx=[],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    })
def mamba_chunkparallel_bwd_dstates_combine(
    B,
    H,
    NP,
    lane_grid,
    nchunks,
    chunk_size: int = 32,
    np_tile: int = 256,
    dtype: str = 'bfloat16',
    threads: int = 256,
):
    """Pass B fused: DS_IN = tri @ G with the strictly-upper triangular decay
    matrix tri[c, c'] = exp(LE_{c'-1} - LE_c) built IN-KERNEL from the
    cumulative chunk log-decay LE, tiled over the flattened [N*P] state axis;
    the ddA state-passing term part3[c] = e_c * <STATES_c, DS_IN_c> is folded
    into the same tile loop (each DS tile is consumed for the dot right after
    it is stored), and DDA is written chunk-broadcast. Replaces the torch tri
    matmul + the 2 GB fp32 part3 einsum. One CTA per (head, batch, lane)."""
    accum_dtype = 'float32'
    LG = lane_grid
    S = nchunks * chunk_size
    assert NP % np_tile == 0
    n_tiles = NP // np_tile

    @T.prim_func
    def mamba_chunkparallel_bwd_dstates_combine_kernel(
            G_IN: T.Tensor((B, LG, H, nchunks, NP), dtype),  # type: ignore
            LE: T.Tensor([B, H, nchunks], T.float32),  # type: ignore
            STATES: T.Tensor([B, H, nchunks, NP], dtype),  # type: ignore
            DS_IN: T.Tensor((B, LG, H, nchunks, NP), dtype),  # type: ignore
            DDA: T.Tensor((B, LG, H, S), T.float32),  # type: ignore
            ):
        with T.Kernel(H, B, LG, threads=threads) as (i_h, i_b, i_l):
            le_shared = T.alloc_shared([nchunks], T.float32)
            T.copy(LE[i_b, i_h, :], le_shared)
            tri_frag = T.alloc_fragment([nchunks, nchunks], dtype)
            for c, cp in T.Parallel(nchunks, nchunks):
                # decay product EXCLUDES the source chunk: exp(LE_{c'-1} - LE_c)
                tri_frag[c, cp] = T.if_then_else(
                    cp > c,
                    T.exp(T.if_then_else(cp >= 1, le_shared[cp - 1], 0.0) - le_shared[c]),
                    0.0)
            tri_shared = T.alloc_shared([nchunks, nchunks], dtype)
            T.copy(tri_frag, tri_shared)
            e_frag = T.alloc_fragment([nchunks], T.float32)
            for c in T.Parallel(nchunks):
                e_frag[c] = T.exp(le_shared[c] - T.if_then_else(c >= 1, le_shared[c - 1], 0.0))

            part3_frag = T.alloc_fragment([nchunks], T.float32)
            T.clear(part3_frag)
            g_tile_shared = T.alloc_shared([nchunks, np_tile], dtype)
            s_tile_shared = T.alloc_shared([nchunks, np_tile], dtype)
            ds_frag = T.alloc_fragment([nchunks, np_tile], accum_dtype)
            red_frag = T.alloc_fragment([nchunks], T.float32)
            for t in T.serial(n_tiles):
                T.copy(G_IN[i_b, i_l, i_h, :, t * np_tile:(t + 1) * np_tile], g_tile_shared)
                T.gemm(tri_shared, g_tile_shared, ds_frag, clear_accum=True)
                T.copy(ds_frag, DS_IN[i_b, i_l, i_h, :, t * np_tile:(t + 1) * np_tile])
                T.copy(STATES[i_b, i_h, :, t * np_tile:(t + 1) * np_tile], s_tile_shared)
                for c, tp in T.Parallel(nchunks, np_tile):
                    ds_frag[c, tp] = ds_frag[c, tp] * s_tile_shared[c, tp]
                T.reduce_sum(ds_frag, red_frag, dim=-1, clear=True)
                for c in T.Parallel(nchunks):
                    part3_frag[c] += red_frag[c]

            for c, cs in T.Parallel(nchunks, chunk_size):
                DDA[i_b, i_l, i_h, c * chunk_size + cs] = part3_frag[c] * e_frag[c]

    return mamba_chunkparallel_bwd_dstates_combine_kernel
