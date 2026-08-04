"""Lane-tile (in-tile P-batched) SISO backward kernel.

Specialization of `mamba_mimo_bwd_bwd` for the LBI P-batched pullback:
R = 1, no Z (gate handled outside), reduceO = False, MIMO_V == 1 (so PsiV == v
and the DMIMO_V output is dropped). Each CTA processes LT cotangent lanes with
ALL forward tiles (q/k/v/states/scalars/lkq mask) loaded and preprocessed ONCE
per chunk -- the lane loop is python-unrolled and reuses one set of per-lane
temporaries, so shared memory is nearly LT-independent. The only per-lane
persistent state is the fp32 `dstates` accumulator ([N, P] registers per lane),
which is the intrinsic cost of P independent backward scans.

R = 1 also collapses the `dqk_from_diag` GEMM: only its DIAGONAL is consumed
(dgamma_diag and the q/k diagonal correction), so it reduces to a row-wise dot
`sum_p dout * v` -- one [cs, cs] fragment and one GEMM removed vs the general
kernel.

Contract identical to the lane_grid mode of `mamba_mimo_bwd_bwd` (same DOUT
[B, LG, S, H, P] cotangent layout, same per-lane output shapes, same bwd_fwd
STATES/QK_DOT inputs); grid is (H, B, LG // LT).
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
def mamba_lanetile_bwd_bwd(
    B,
    S,
    H,
    G,
    N,
    P,
    lane_grid,
    lane_tile,
    hasD,
    chunk_size: int = 64,
    rotary_dim_divisor: int = 4,
    dtype: str = 'bfloat16',
    threads: int = 256,
    num_stages: int = 0,
):
    accum_dtype = 'float32'
    assert lane_grid % lane_tile == 0, "lane_tile must divide lane_grid"
    LT = lane_tile
    LG = lane_grid
    LO = LG // LT  # lane-outer grid extent
    nchunks = tilelang.cdiv(S, chunk_size)
    n_rot = N // rotary_dim_divisor

    @T.prim_func
    def mamba_lanetile_bwd_bwd_kernel(
            DOUT: T.Tensor((B, LG, S, H, P), dtype),  # type: ignore
            Q: T.Tensor([B, S, 1, G, N], dtype),  # type: ignore
            K: T.Tensor([B, S, 1, G, N], dtype),  # type: ignore
            V: T.Tensor([B, S, H, P], dtype),  # type: ignore
            Q_BIAS: T.Tensor([H, 1, N], T.float32),  # type: ignore
            K_BIAS: T.Tensor([H, 1, N], T.float32),  # type: ignore
            DK: T.Tensor((B, LG, S, H, N), dtype),  # type: ignore
            DV: T.Tensor((B, LG, S, H, P), dtype),  # type: ignore
            STATES: T.Tensor([B, H, nchunks, N, P], dtype),  # type: ignore
            DQ: T.Tensor((B, LG, S, H, N), dtype),  # type: ignore
            ANGLES: T.Tensor([B, S, H, n_rot], T.float32),  # type: ignore
            DA_CS: T.Tensor([B, H, S], T.float32),  # type: ignore
            DA_CS_REV: T.Tensor([B, H, S], T.float32),  # type: ignore
            DT: T.Tensor([B, H, S], T.float32),  # type: ignore
            TRAP: T.Tensor([B, H, S], dtype),  # type: ignore
            DFACTOR: T.Tensor((B, LG, H, S), T.float32),  # type: ignore
            DGAMMA_DIAG: T.Tensor((B, LG, H, S), T.float32),  # type: ignore
            DANGLES: T.Tensor((B, LG, S, H, n_rot), T.float32),  # type: ignore
            D: T.Tensor([H], T.float32),  # type: ignore
            DD: T.Tensor((B, LG, H), T.float32),  # type: ignore
            QK_DOT: T.Tensor([B, H, S, 1, 1], dtype),  # type: ignore
            DDA: T.Tensor((B, LG, H, S), T.float32),  # type: ignore
            DSSDA: T.Tensor((B, LG, H, nchunks, chunk_size, chunk_size), T.float32),  # type: ignore
            DDA_CS_REV: T.Tensor((B, LG, H, S), T.float32),  # type: ignore
            DDA_CS: T.Tensor((B, LG, H, S), T.float32),  # type: ignore
            SEGSUM: T.Tensor([B, H, nchunks, chunk_size, chunk_size], T.float32),  # type: ignore
            ):
        with T.Kernel(H, B, LO, threads=threads) as (i_h, i_b, i_lo):
            i_h_qk = i_h // (H // G)

            # --- Shared (lane-free) buffers ---
            q_shared = T.alloc_shared([chunk_size, N], dtype)          # rotated q (+bias)
            q_pre_rot_shared = T.alloc_shared([chunk_size, N], dtype)  # biased, unrotated
            k_shared = T.alloc_shared([chunk_size, N], dtype)          # rotated + trap-scaled
            k_pre_rot_shared = T.alloc_shared([chunk_size, N], dtype)  # biased, unrotated
            k_pre_trap_shared = T.alloc_shared([chunk_size, N], dtype)  # rotated, unscaled
            v_shared = T.alloc_shared([chunk_size, P], dtype)
            states_shared = T.alloc_shared([N, P], dtype)
            lkq_unmasked_shared = T.alloc_shared([chunk_size, chunk_size], dtype)
            lkq_masked_shared = T.alloc_shared([chunk_size, chunk_size], dtype)

            # --- Per-lane temporaries (ONE copy, reused across the unrolled lane loop) ---
            dout_shared = T.alloc_shared([chunk_size, P], dtype)
            dstates_op_shared = T.alloc_shared([N, P], dtype)          # bf16 gemm operand
            dkq_masked_shared = T.alloc_shared([chunk_size, chunk_size], dtype)
            dk_stage_shared = T.alloc_shared([chunk_size, N], dtype)
            dq_stage_shared = T.alloc_shared([chunk_size, N], dtype)

            # --- Per-lane PERSISTENT state (fp32 register accumulators) ---
            dstates_frags = T.alloc_fragment([LT, N, P], accum_dtype)
            T.clear(dstates_frags)
            if hasD:
                dD_frags = T.alloc_fragment([LT], accum_dtype)
                T.clear(dD_frags)

            q_bias_frag = T.alloc_fragment([1, N], dtype)
            k_bias_frag = T.alloc_fragment([1, N], dtype)
            T.copy(Q_BIAS[i_h, :, :], q_bias_frag)
            T.copy(K_BIAS[i_h, :, :], k_bias_frag)

            for chunk_idx_rev in T.Pipelined(0, nchunks, num_stages=num_stages):
                chunk_idx = nchunks - 1 - chunk_idx_rev
                chunk_start = chunk_idx * chunk_size

                # ============ SHARED PROLOGUE (once per chunk) ============
                # Discretization scalars.
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
                da_cs_sum = T.alloc_var(T.float32)
                T.copy(DA_CS[i_b, i_h, chunk_start + chunk_size - 1], da_cs_sum)

                # q: load + bias -> pre-rot save -> rotary (multi-offset
                # accesses route through SHARED; fragments need one pattern).
                q_frag = T.alloc_fragment([chunk_size, N], dtype)
                for cs, n in T.Parallel(chunk_size, N):
                    q_frag[cs, n] = Q[i_b, chunk_start + cs, 0, i_h_qk, n] + q_bias_frag[0, n]
                T.copy(q_frag, q_pre_rot_shared)
                T.copy(q_frag, q_shared)
                angles_frag = T.alloc_fragment([chunk_size, n_rot], T.float32)
                T.copy(ANGLES[i_b, chunk_start:chunk_start + chunk_size, i_h, :], angles_frag)
                q_h1_frag = T.alloc_fragment([chunk_size, n_rot], dtype)
                q_h2_frag = T.alloc_fragment([chunk_size, n_rot], dtype)
                for cs, n in T.Parallel(chunk_size, n_rot):
                    q_h1_frag[cs, n] = q_pre_rot_shared[cs, n]
                    q_h2_frag[cs, n] = q_pre_rot_shared[cs, N // 2 + n]
                for cs, n in T.Parallel(chunk_size, n_rot):
                    q_shared[cs, n] = (T.cos(angles_frag[cs, n]) * q_h1_frag[cs, n]
                                       - T.sin(angles_frag[cs, n]) * q_h2_frag[cs, n])
                    q_shared[cs, N // 2 + n] = (T.sin(angles_frag[cs, n]) * q_h1_frag[cs, n]
                                                + T.cos(angles_frag[cs, n]) * q_h2_frag[cs, n])

                # k: load + bias -> pre-rot save -> rotary (pre-trap save) -> trap scale.
                k_frag = T.alloc_fragment([chunk_size, N], dtype)
                for cs, n in T.Parallel(chunk_size, N):
                    k_frag[cs, n] = K[i_b, chunk_start + cs, 0, i_h_qk, n] + k_bias_frag[0, n]
                T.copy(k_frag, k_pre_rot_shared)
                T.copy(k_frag, k_pre_trap_shared)
                k_h1_frag = T.alloc_fragment([chunk_size, n_rot], dtype)
                k_h2_frag = T.alloc_fragment([chunk_size, n_rot], dtype)
                for cs, n in T.Parallel(chunk_size, n_rot):
                    k_h1_frag[cs, n] = k_pre_rot_shared[cs, n]
                    k_h2_frag[cs, n] = k_pre_rot_shared[cs, N // 2 + n]
                for cs, n in T.Parallel(chunk_size, n_rot):
                    k_pre_trap_shared[cs, n] = (T.cos(angles_frag[cs, n]) * k_h1_frag[cs, n]
                                                - T.sin(angles_frag[cs, n]) * k_h2_frag[cs, n])
                    k_pre_trap_shared[cs, N // 2 + n] = (T.sin(angles_frag[cs, n]) * k_h1_frag[cs, n]
                                                         + T.cos(angles_frag[cs, n]) * k_h2_frag[cs, n])
                k_scaled_frag = T.alloc_fragment([chunk_size, N], dtype)
                T.copy(k_pre_trap_shared, k_scaled_frag)
                for cs, n in T.Parallel(chunk_size, N):
                    k_scaled_frag[cs, n] *= trap_scale_shared[cs]
                T.copy(k_scaled_frag, k_shared)

                T.copy(V[i_b, chunk_start:chunk_start + chunk_size, i_h, :], v_shared)

                # lkq = k_scaled @ q_rot^T; save unmasked; mask with reverse-causal segsum.
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
                states_frag = T.alloc_fragment([N, P], T.float32)
                T.copy(states_shared, states_frag)

                if hasD:
                    D_var = T.alloc_var(T.float32)
                    T.copy(D[i_h], D_var)

                # Per-lane temps allocated ONCE per chunk, reused across the
                # unrolled lane loop (dependencies serialize the reuse).
                dout_frag = T.alloc_fragment([chunk_size, P], dtype)
                dPsiV_frag = T.alloc_fragment([chunk_size, P], accum_dtype)
                dv_out_frag = T.alloc_fragment([chunk_size, P], dtype)
                dqk_diag_pre_frag = T.alloc_fragment([chunk_size, P], accum_dtype)
                dqk_diag_frag = T.alloc_fragment([chunk_size], accum_dtype)
                dgamma_diag_frag = T.alloc_fragment([chunk_size], accum_dtype)
                dqk_diag_gamma_frag = T.alloc_fragment([chunk_size], accum_dtype)
                dk_frag_l = T.alloc_fragment([chunk_size, N], accum_dtype)
                ddacsrev_pre_frag = T.alloc_fragment([chunk_size, N], accum_dtype)
                ddacsrev_frag = T.alloc_fragment([chunk_size], accum_dtype)
                dkq_frag = T.alloc_fragment([chunk_size, chunk_size], accum_dtype)
                dssda_frag = T.alloc_fragment([chunk_size, chunk_size], accum_dtype)
                dkq_masked_frag = T.alloc_fragment([chunk_size, chunk_size], dtype)
                dfactor_pre_frag = T.alloc_fragment([chunk_size, N], accum_dtype)
                dfactor_frag = T.alloc_fragment([chunk_size], accum_dtype)
                dda_sp_pre_frag = T.alloc_fragment([N, P], T.float32)
                dda_sp_frag = T.alloc_fragment([1], T.float32)
                dda_bcast_frag = T.alloc_fragment([chunk_size], T.float32)
                dq_frag_l = T.alloc_fragment([chunk_size, N], accum_dtype)
                ddacs_pre_frag = T.alloc_fragment([chunk_size, N], accum_dtype)
                ddacs_frag = T.alloc_fragment([chunk_size], accum_dtype)
                dk_h1_l_frag = T.alloc_fragment([chunk_size, n_rot], dtype)
                dk_h2_l_frag = T.alloc_fragment([chunk_size, n_rot], dtype)
                dq_h1_l_frag = T.alloc_fragment([chunk_size, n_rot], dtype)
                dq_h2_l_frag = T.alloc_fragment([chunk_size, n_rot], dtype)
                kpre_h1_frag = T.alloc_fragment([chunk_size, n_rot], dtype)
                kpre_h2_frag = T.alloc_fragment([chunk_size, n_rot], dtype)
                qpre_h1_frag = T.alloc_fragment([chunk_size, n_rot], dtype)
                qpre_h2_frag = T.alloc_fragment([chunk_size, n_rot], dtype)
                dangle_frag = T.alloc_fragment([chunk_size, n_rot], T.float32)
                dk_out_frag = T.alloc_fragment([chunk_size, N], accum_dtype)
                dq_out_frag = T.alloc_fragment([chunk_size, N], accum_dtype)
                dout_scaled_frag = T.alloc_fragment([chunk_size, P], dtype)

                # ============ LANE LOOP (serial; buffers reused per lane) ============
                for lt in T.serial(LT):
                    lane = i_lo * LT + lt

                    # dout tile for this lane; bf16 dstates operand from the
                    # lane's fp32 accumulator (previous-chunk value).
                    for cs, p in T.Parallel(chunk_size, P):
                        dout_shared[cs, p] = DOUT[i_b, lane, chunk_start + cs, i_h, p]
                    for n, p in T.Parallel(N, P):
                        dstates_op_shared[n, p] = dstates_frags[lt, n, p]
                    T.copy(dout_shared, dout_frag)

                    if hasD:
                        dD_pre_frag = T.alloc_fragment([chunk_size, P], accum_dtype)
                        for cs, p in T.Parallel(chunk_size, P):
                            dD_pre_frag[cs, p] = dout_frag[cs, p] * v_shared[cs, p]
                        dD_tmp_frag = T.alloc_fragment([1], accum_dtype)
                        T.reduce_sum(T.view(dD_pre_frag, shape=[chunk_size * P]),
                                     dD_tmp_frag, clear=True)
                        dD_frags[lt] += dD_tmp_frag[0]

                    # dPsiV: interchunk (k_scaled @ dstates) * exp(dA_cs_rev)
                    #        + intrachunk (lkq_masked @ dout) + D/qk_dot diagonal.
                    T.gemm(k_shared, dstates_op_shared, dPsiV_frag, clear_accum=True)
                    for cs, p in T.Parallel(chunk_size, P):
                        dPsiV_frag[cs, p] *= exp_dA_cs_rev_frag[cs]
                    T.gemm(lkq_masked_shared, dout_shared, dPsiV_frag, clear_accum=False)
                    if hasD:
                        for cs, p in T.Parallel(chunk_size, P):
                            dPsiV_frag[cs, p] += dout_frag[cs, p] * D_var
                    for cs, p in T.Parallel(chunk_size, P):
                        dPsiV_frag[cs, p] += dout_frag[cs, p] * qk_dot_frag[cs] * gamma_frag[cs]
                    T.copy(dPsiV_frag, dv_out_frag)
                    T.copy(dv_out_frag, DV[i_b, lane, chunk_start:chunk_start + chunk_size, i_h, :])

                    # Diagonal qk correction (R=1: only the diagonal of
                    # dout @ (v)^T is consumed): dqk_diag = sum_p dout * v.
                    for cs, p in T.Parallel(chunk_size, P):
                        dqk_diag_pre_frag[cs, p] = dout_frag[cs, p] * v_shared[cs, p]
                    T.reduce_sum(dqk_diag_pre_frag, dqk_diag_frag, dim=-1, clear=True)
                    for cs in T.Parallel(chunk_size):
                        dgamma_diag_frag[cs] = qk_dot_frag[cs] * dqk_diag_frag[cs]
                    T.copy(dgamma_diag_frag, DGAMMA_DIAG[i_b, lane, i_h, chunk_start:chunk_start + chunk_size])
                    for cs in T.Parallel(chunk_size):
                        dqk_diag_gamma_frag[cs] = dqk_diag_frag[cs] * gamma_frag[cs]

                    # dk interchunk: (PsiV == v) @ dstates^T; ddA_cs_rev before scaling.
                    T.gemm(v_shared, dstates_op_shared, dk_frag_l, transpose_B=True, clear_accum=True)
                    for cs, n in T.Parallel(chunk_size, N):
                        ddacsrev_pre_frag[cs, n] = k_shared[cs, n] * dk_frag_l[cs, n]
                    T.reduce_sum(ddacsrev_pre_frag, ddacsrev_frag, dim=-1, clear=True)
                    T.copy(ddacsrev_frag, DDA_CS_REV[i_b, lane, i_h, chunk_start:chunk_start + chunk_size])
                    for cs, n in T.Parallel(chunk_size, N):
                        dk_frag_l[cs, n] *= exp_dA_cs_rev_frag[cs]

                    # dkq intrachunk: v @ dout^T; DSSDA = lkq_unmasked * dkq; mask.
                    T.gemm(v_shared, dout_shared, dkq_frag, transpose_B=True, clear_accum=True)
                    for cs_i, cs_j in T.Parallel(chunk_size, chunk_size):
                        dssda_frag[cs_i, cs_j] = lkq_unmasked_shared[cs_i, cs_j] * dkq_frag[cs_i, cs_j]
                    T.copy(dssda_frag, DSSDA[i_b, lane, i_h, chunk_idx, :, :])
                    T.copy(dkq_frag, dkq_masked_frag)
                    for cs_i, cs_j in T.Parallel(chunk_size, chunk_size):
                        dkq_masked_frag[cs_i, cs_j] = T.if_then_else(
                            cs_i < cs_j,
                            dkq_masked_frag[cs_i, cs_j] * T.exp(SEGSUM[i_b, i_h, chunk_idx, cs_j, cs_i]),
                            0.0)
                    T.copy(dkq_masked_frag, dkq_masked_shared)

                    # dk_nodiag = dk + dkq_masked @ q_rot; dfactor; trap scale.
                    # (shared roundtrip: normalize the accumulator layout between
                    # the two differently-shaped gemms, as in the general kernel)
                    T.copy(dk_frag_l, dk_stage_shared)
                    T.copy(dk_stage_shared, dk_frag_l)
                    T.gemm(dkq_masked_shared, q_shared, dk_frag_l, clear_accum=False)
                    for cs, n in T.Parallel(chunk_size, N):
                        dfactor_pre_frag[cs, n] = k_pre_trap_shared[cs, n] * dk_frag_l[cs, n]
                    T.reduce_sum(dfactor_pre_frag, dfactor_frag, dim=-1, clear=True)
                    T.copy(dfactor_frag, DFACTOR[i_b, lane, i_h, chunk_start:chunk_start + chunk_size])
                    for cs, n in T.Parallel(chunk_size, N):
                        dk_frag_l[cs, n] *= trap_scale_shared[cs]

                    # ddA state passing (uses the PRE-update dstates).
                    for n, p in T.Parallel(N, P):
                        dda_sp_pre_frag[n, p] = states_frag[n, p] * dstates_frags[lt, n, p] * T.exp(da_cs_sum)
                    T.reduce_sum(T.view(dda_sp_pre_frag, shape=[N * P]), dda_sp_frag, dim=-1, clear=True)
                    for cs in T.Parallel(chunk_size):
                        dda_bcast_frag[cs] = dda_sp_frag[0]
                    T.copy(dda_bcast_frag, DDA[i_b, lane, i_h, chunk_start:chunk_start + chunk_size])

                    # dq interchunk: dout @ states^T; ddA_cs; scale; + dkq_masked^T @ k_scaled.
                    T.gemm(dout_shared, states_shared, dq_frag_l, transpose_B=True, clear_accum=True)
                    for cs, n in T.Parallel(chunk_size, N):
                        ddacs_pre_frag[cs, n] = q_shared[cs, n] * dq_frag_l[cs, n]
                    T.reduce_sum(ddacs_pre_frag, ddacs_frag, dim=-1, clear=True)
                    T.copy(ddacs_frag, DDA_CS[i_b, lane, i_h, chunk_start:chunk_start + chunk_size])
                    for cs, n in T.Parallel(chunk_size, N):
                        dq_frag_l[cs, n] *= T.exp(dA_cs_shared[cs])
                    T.copy(dq_frag_l, dq_stage_shared)
                    T.copy(dq_stage_shared, dq_frag_l)
                    T.gemm(dkq_masked_shared, k_shared, dq_frag_l, transpose_A=True, clear_accum=False)

                    # Inverse rotary + dangles (dk first, then dq), then the
                    # diagonal qk corrections against the PRE-ROTATED q/k.
                    # Multi-offset accesses route through SHARED buffers.
                    T.copy(dk_frag_l, dk_stage_shared)
                    for cs, n in T.Parallel(chunk_size, n_rot):
                        dk_h1_l_frag[cs, n] = dk_stage_shared[cs, n]
                        dk_h2_l_frag[cs, n] = dk_stage_shared[cs, N // 2 + n]
                        kpre_h1_frag[cs, n] = k_pre_rot_shared[cs, n]
                        kpre_h2_frag[cs, n] = k_pre_rot_shared[cs, N // 2 + n]
                    for cs, n in T.Parallel(chunk_size, n_rot):
                        dangle_frag[cs, n] = (
                            dk_h1_l_frag[cs, n] * (-kpre_h1_frag[cs, n] * T.sin(angles_frag[cs, n])
                                                   - kpre_h2_frag[cs, n] * T.cos(angles_frag[cs, n]))
                            + dk_h2_l_frag[cs, n] * (kpre_h1_frag[cs, n] * T.cos(angles_frag[cs, n])
                                                     - kpre_h2_frag[cs, n] * T.sin(angles_frag[cs, n])))
                    for cs, n in T.Parallel(chunk_size, n_rot):
                        dk_stage_shared[cs, n] = (T.cos(angles_frag[cs, n]) * dk_h1_l_frag[cs, n]
                                                  + T.sin(angles_frag[cs, n]) * dk_h2_l_frag[cs, n])
                        dk_stage_shared[cs, N // 2 + n] = (-T.sin(angles_frag[cs, n]) * dk_h1_l_frag[cs, n]
                                                           + T.cos(angles_frag[cs, n]) * dk_h2_l_frag[cs, n])
                    for cs, n in T.Parallel(chunk_size, N):
                        dk_out_frag[cs, n] = dk_stage_shared[cs, n] + dqk_diag_gamma_frag[cs] * q_pre_rot_shared[cs, n]
                    T.copy(dk_out_frag, DK[i_b, lane, chunk_start:chunk_start + chunk_size, i_h, :])

                    T.copy(dq_frag_l, dq_stage_shared)
                    for cs, n in T.Parallel(chunk_size, n_rot):
                        dq_h1_l_frag[cs, n] = dq_stage_shared[cs, n]
                        dq_h2_l_frag[cs, n] = dq_stage_shared[cs, N // 2 + n]
                        qpre_h1_frag[cs, n] = q_pre_rot_shared[cs, n]
                        qpre_h2_frag[cs, n] = q_pre_rot_shared[cs, N // 2 + n]
                    for cs, n in T.Parallel(chunk_size, n_rot):
                        dangle_frag[cs, n] += (
                            dq_h1_l_frag[cs, n] * (-qpre_h1_frag[cs, n] * T.sin(angles_frag[cs, n])
                                                   - qpre_h2_frag[cs, n] * T.cos(angles_frag[cs, n]))
                            + dq_h2_l_frag[cs, n] * (qpre_h1_frag[cs, n] * T.cos(angles_frag[cs, n])
                                                     - qpre_h2_frag[cs, n] * T.sin(angles_frag[cs, n])))
                    T.copy(dangle_frag, DANGLES[i_b, lane, chunk_start:chunk_start + chunk_size, i_h, :])
                    for cs, n in T.Parallel(chunk_size, n_rot):
                        dq_stage_shared[cs, n] = (T.cos(angles_frag[cs, n]) * dq_h1_l_frag[cs, n]
                                                  + T.sin(angles_frag[cs, n]) * dq_h2_l_frag[cs, n])
                        dq_stage_shared[cs, N // 2 + n] = (-T.sin(angles_frag[cs, n]) * dq_h1_l_frag[cs, n]
                                                           + T.cos(angles_frag[cs, n]) * dq_h2_l_frag[cs, n])
                    for cs, n in T.Parallel(chunk_size, N):
                        dq_out_frag[cs, n] = dq_stage_shared[cs, n] + dqk_diag_gamma_frag[cs] * k_pre_rot_shared[cs, n]
                    T.copy(dq_out_frag, DQ[i_b, lane, chunk_start:chunk_start + chunk_size, i_h, :])

                    # dstates update (for the next, earlier chunk):
                    # dstates = dstates * exp(da_cs_sum) + q_rot^T @ (dout * exp(dA_cs)).
                    # (gemm into a 2D temp, then elementwise-accumulate into the
                    # lane slice -- gemm into a 3D-fragment slice is unsupported)
                    for cs, p in T.Parallel(chunk_size, P):
                        dout_scaled_frag[cs, p] = dout_frag[cs, p] * T.exp(dA_cs_shared[cs])
                    T.copy(dout_scaled_frag, dout_shared)
                    dstates_upd_frag = T.alloc_fragment([N, P], accum_dtype)
                    T.gemm(q_shared, dout_shared, dstates_upd_frag, transpose_A=True, clear_accum=True)
                    for n, p in T.Parallel(N, P):
                        dstates_frags[lt, n, p] = dstates_frags[lt, n, p] * T.exp(da_cs_sum) + dstates_upd_frag[n, p]

            if hasD:
                for lt in T.serial(LT):
                    DD[i_b, i_lo * LT + lt, i_h] = dD_frags[lt]

    return mamba_lanetile_bwd_bwd_kernel
