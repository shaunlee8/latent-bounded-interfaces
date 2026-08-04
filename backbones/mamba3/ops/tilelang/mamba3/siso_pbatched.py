"""P-batched SISO scan pullback via the tilelang MIMO backward.

The MIMO recurrence is multi-input/multi-output over one shared state:

    S_t = alpha S_{t-1} + beta (k_{t-1} x v_{t-1}) + gamma (k_t x v_t),
    with k x v = sum_r k_r (x) v_r,      out_r = q_r . S_t

With Q/K broadcast across R and `mimo_v = 1/R`, every channel's forward is the
SISO forward, and a rank-carrying cotangent drives the backward for all P
lanes in one kernel with shared tiles. The shared state means the unmodified
kernel lane-resolves only the q side; the lane-resolved kernels (lanetile,
chunk-parallel) widen the state-cotangent chain with the lane dimension.

This path emits scan-input-space gradients; its parity target is the
registered Triton backward (`mamba3_siso_combined`). The per-lane loop
reference (`mamba3_siso_scan_input_pullback_basis`) is NOT comparable at this
boundary: its intermediates are an internal decomposition that only composes
to true gradients downstream, so a direct comparison reports a false mismatch.

Contract notes:
  * ANGLES = cumulative rotary phases (`angle_dt_fwd` output); DANGLES is
    converted to raw dAngles + the dt-angle term by `angle_dt_bwd`.
  * Z-gating and the D-skip are handled outside the kernel (per-lane
    `compute_dzdo` / `D*do`), so `z=None, D=None` inside.
  * `ddt` from the kernel is the decay-path DT grad; `angle_dt_bwd` adds the
    angle-path term.
"""

from __future__ import annotations

import torch

from backbones.mamba3.ops.triton.mamba3.angle_dt import angle_dt_bwd, angle_dt_fwd
from backbones.mamba3.ops.triton.mamba3.mamba3_mimo_utils import (
    bwd_dadt_fused_triton,
    bwd_dtrap_ddt_triton,
    compute_dacs_segsum_triton,
)
from backbones.mamba3.ops.tilelang.mamba3.mamba3_mimo_bwd import (
    mamba_mimo_bwd_bwd,
    mamba_mimo_bwd_combined,
    mamba_mimo_bwd_fwd,
)


def _deinterleave(x: torch.Tensor) -> torch.Tensor:
    """Rotary-convention bridge: our SISO Triton kernels rotate INTERLEAVED pairs
    (2i, 2i+1); the tilelang MIMO kernels rotate HALF-SPLIT pairs (i, N/2+i).
    Mapping feature 2i -> i and 2i+1 -> N/2+i makes the two rotations identical
    (validated: forward parity 5.9e-3 bf16 with, 9.5e-2 without)."""
    return torch.cat([x[..., 0::2], x[..., 1::2]], dim=-1)


def _interleave(y: torch.Tensor) -> torch.Tensor:
    """Inverse of `_deinterleave` (applied to dq/dk coming back)."""
    n = y.shape[-1]
    out = torch.empty_like(y)
    out[..., 0::2] = y[..., : n // 2]
    out[..., 1::2] = y[..., n // 2 :]
    return out


def mamba3_siso_pbatched_pullback(
    *,
    mixer,
    cache,
    output_cotangent_basis: torch.Tensor,
    tl_chunk_size: int = 64,
) -> dict[str, torch.Tensor]:
    """Stage B1: PER-LANE adjoints for ALL eight scan inputs in ONE kernel launch
    pair, by folding the P lanes into the batch dimension (R=1 per lane instance).

    Each lane is an independent SISO backward, so batching lanes as batch entries
    is exact (no state coupling across lanes). Costs R x memory traffic on the
    replicated forward tiles (vs the future lane-blocked in-tile kernel that
    shares them), but eliminates the per-lane launch cascade: 2 tilelang launches
    + glue instead of ~6 Triton launches x P lanes. R=1 also lifts the
    shared-memory wall (chunk 64 fine).

    Returns the per-lane adjoint dict {Q,K,V,ADT,DT,Trap,Angles,Z}, each
    [B, P, ...scan shape...] in the scan input's dtype -- TRUE scan-input-space
    gradients (parity target: the registered backward)."""
    from backends.mamba3 import _mamba3_scan_inputs_from_cache
    from backbones.mamba3.ops.triton.mamba3.mamba3_siso_bwd import compute_dzdo
    from backbones.mamba3.ops.triton.mamba3.mamba3_siso_fwd import mamba3_siso_fwd

    scan = _mamba3_scan_inputs_from_cache(mixer, cache)
    bsz, num_p, seqlen, heads, hd = output_cotangent_basis.shape
    n_state = scan["Q"].shape[-1]
    cs = mixer.chunk_size
    bp = bsz * num_p

    # Shared forward quantities (cotangent-independent, computed once).
    angles_cumsum, _ = angle_dt_fwd(
        scan["Angles"], scan["DT"], init_state=None, chunk_size=cs,
        return_output_state=True, cu_seqlens=None,
    )
    q_bias = mixer.C_bias.squeeze(1)
    k_bias = mixer.B_bias.squeeze(1)
    _out, out_v, *_rest = mamba3_siso_fwd(
        scan["Q"], scan["K"], scan["V"], scan["ADT"], scan["DT"], scan["Trap"],
        q_bias, k_bias, angles_cumsum, mixer.D, scan["Z"], None,
        chunk_size=cs, store_states_adt_outv=True, return_final_states=False,
    )

    # Per-lane Z-gate VJP outside the kernel.
    do = output_cotangent_basis.to(dtype=scan["V"].dtype)
    if scan["Z"] is not None:
        dz_lanes, do_scaled = [], []
        for j in range(num_p):
            dz_j, do_j = compute_dzdo(do[:, j].contiguous(), scan["Z"], out_v, chunk_size=cs)
            dz_lanes.append(dz_j.unsqueeze(1))
            do_scaled.append(do_j.unsqueeze(1))
        dz = torch.cat(dz_lanes, dim=1)
        do = torch.cat(do_scaled, dim=1)
    else:
        dz = None

    def rep(x, batch_dim=0):
        # replicate a [B, ...] tensor to [B*P, ...] (lane-major within batch)
        return x.unsqueeze(1).expand(bsz, num_p, *x.shape[1:]).reshape(bp, *x.shape[1:]).contiguous()

    q = _deinterleave(scan["Q"])
    k = _deinterleave(scan["K"])
    q_r = rep(q).unsqueeze(2)          # [BP, S, R=1, G, N]
    k_r = rep(k).unsqueeze(2)
    v_r = rep(scan["V"].contiguous())
    qb = _deinterleave(q_bias).unsqueeze(1).contiguous().float()   # [H, 1, N]
    kb = _deinterleave(k_bias).unsqueeze(1).contiguous().float()
    mimo_v = torch.ones((heads, 1, hd), device=do.device, dtype=torch.float32)
    angles_r = rep(angles_cumsum.float().contiguous())
    dt_r = rep(scan["DT"].float().contiguous())
    trap_r = rep(scan["Trap"].contiguous())
    dA_r = rep(scan["ADT"].float().contiguous())
    dA_cs, dA_cs_rev, segsum = compute_dacs_segsum_triton(dA_r, tl_chunk_size)
    dout = do.reshape(bp, seqlen, heads, hd).unsqueeze(2).contiguous()  # [BP, S, 1, H, hd]

    (dq, dk, dv, ddA, ddt, dtrap, _qb, _kb, _mv, _mz, _mo, dangles, _dD, _dz) = mamba_mimo_bwd_combined(
        dout, q_r, k_r, v_r, qb, kb, mimo_v, None,
        None, None,
        angles_r, dA_cs, dA_cs_rev, dt_r, trap_r, None,
        segsum, tl_chunk_size, mixer.rotary_dim_divisor, "bfloat16",
    )

    # D-skip contribution to dV (excluded via D=None): per-lane D * do_j.
    dv = dv.reshape(bsz, num_p, seqlen, heads, hd) + torch.einsum(
        "bplhd,h->bplhd", do.float(), mixer.D.float()
    ).to(dv.dtype)

    # Angle epilogue per lane (batched over BP): cumsum-phase grads -> raw dAngles
    # + the dt-angle term.
    dangles_raw, ddt_angle, _ = angle_dt_bwd(
        grad_out=dangles, angle=rep(scan["Angles"].contiguous()), dt=dt_r,
        has_init_state=False, chunk_size=tl_chunk_size, grad_output_state=None,
    )

    def unrep(x):
        return x.reshape(bsz, num_p, *x.shape[1:])

    return {
        "Q": _interleave(dq.squeeze(2)).reshape(bsz, num_p, seqlen, 1, n_state).to(dtype=scan["Q"].dtype),
        "K": _interleave(dk.squeeze(2)).reshape(bsz, num_p, seqlen, 1, n_state).to(dtype=scan["K"].dtype),
        "V": dv.to(dtype=scan["V"].dtype),
        "ADT": unrep(ddA).to(dtype=scan["ADT"].dtype),
        "DT": unrep(ddt + ddt_angle).to(dtype=scan["DT"].dtype),
        "Trap": unrep(dtrap).to(dtype=scan["Trap"].dtype),
        "Angles": unrep(dangles_raw).to(dtype=scan["Angles"].dtype),
        "Z": dz,
    }


def mamba3_siso_pbatched_pullback_lanegrid(
    *,
    mixer,
    cache,
    output_cotangent_basis: torch.Tensor,
    tl_chunk_size: int = 64,
    lane_tile: int = 0,
    chunk_parallel: bool = False,
) -> dict[str, torch.Tensor]:
    """Stage B3 (P-in-grid): PER-LANE adjoints for ALL EIGHT scan inputs with ZERO
    input replication.

    The cotangent-independent pass (`bwd_fwd`: STATES + QK_DOT) runs ONCE; the
    per-lane pass (`bwd_bwd` with the `lane_grid` axis) reads all forward tensors
    lane-free (shared across lane CTAs via cache) and writes every gradient with a
    lane dimension. R = 1 per lane, so the full tuned chunk size (64) applies --
    no shared-memory wall. Returns per-lane {Q,K,V,ADT,DT,Trap,Angles,Z}.

    `lane_tile > 0` switches pass 2 to the IN-TILE kernel
    (`mamba_lanetile_bwd_bwd`): each CTA processes `lane_tile` lanes with the
    forward tiles loaded and preprocessed ONCE per chunk (the MIMO-style
    amortization); must divide the number of lanes. Same outputs.

    `chunk_parallel=True` switches pass 2 to the CHUNK-PARALLEL three-pass
    form (removes the serial chunk chain that leaves the serial kernel
    latency-bound at ~5% FLOP utilization): pass A computes the chunk-local
    state-cotangent contributions G_c = q_rot^T (dout * exp(dA_cs)) in torch,
    pass B runs the tiny inter-chunk reverse scan, pass C
    (`mamba_chunkparallel_bwd_bwd`) computes every gradient with one CTA per
    (head, batch, lane, chunk). Use tl_chunk_size=32 so pass C fits 2 CTAs/SM.
    Same outputs."""
    from backends.mamba3 import _mamba3_scan_inputs_from_cache
    from backbones.mamba3.ops.triton.mamba3.mamba3_siso_bwd import compute_dzdo
    from backbones.mamba3.ops.triton.mamba3.mamba3_siso_fwd import mamba3_siso_fwd

    scan = _mamba3_scan_inputs_from_cache(mixer, cache)
    bsz, num_p, seqlen, heads, hd = output_cotangent_basis.shape
    n_state = scan["Q"].shape[-1]
    cs = mixer.chunk_size
    rdd = mixer.rotary_dim_divisor
    tcs = int(tl_chunk_size)
    nch = seqlen // tcs
    dev = output_cotangent_basis.device
    n_rope = n_state // rdd

    # Shared forward quantities.
    angles_cumsum, _ = angle_dt_fwd(
        scan["Angles"], scan["DT"], init_state=None, chunk_size=cs,
        return_output_state=True, cu_seqlens=None,
    )
    q_bias = mixer.C_bias.squeeze(1)
    k_bias = mixer.B_bias.squeeze(1)
    _out, out_v, *_rest = mamba3_siso_fwd(
        scan["Q"], scan["K"], scan["V"], scan["ADT"], scan["DT"], scan["Trap"],
        q_bias, k_bias, angles_cumsum, mixer.D, scan["Z"], None,
        chunk_size=cs, store_states_adt_outv=True, return_final_states=False,
    )
    do = output_cotangent_basis.to(dtype=scan["V"].dtype).contiguous()
    gate = None
    if scan["Z"] is not None and chunk_parallel:
        # Z-gate VJP fused into the kernels: the gate factors are LANE-FREE
        # elementwise fields, so `do` stays raw -- passes A and C multiply
        # dout by GATE = silu(z) on load, and dZ = do * out_v * silu'(z) is
        # one broadcast multiply (replaces compute_dzdo and its lane-tiled
        # Z/out_v copies, ~2-3 ms at the representative shape).
        z_f = scan["Z"].float()
        sig = torch.sigmoid(z_f)
        gate = (z_f * sig).to(torch.bfloat16).contiguous()           # silu(z)
        zfac = (out_v.float() * sig * (1.0 + z_f * (1.0 - sig))).to(torch.bfloat16)
        dz = (do * zfac.unsqueeze(1)).view(bsz, num_p, seqlen, heads, hd)
    elif scan["Z"] is not None:
        # batched Z-gate VJP: lanes folded into batch (zero-copy view of do);
        # Z/out_v tiled once.
        def tile_bp(x):
            return x.unsqueeze(1).expand(bsz, num_p, *x.shape[1:]).reshape(
                bsz * num_p, *x.shape[1:]).contiguous()

        dz_f, do_f = compute_dzdo(do.view(bsz * num_p, seqlen, heads, hd),
                                  tile_bp(scan["Z"].contiguous()), tile_bp(out_v), chunk_size=cs)
        dz = dz_f.view(bsz, num_p, seqlen, heads, hd)
        do = do_f.view(bsz, num_p, seqlen, heads, hd)
    else:
        dz = None
    if chunk_parallel and gate is None:
        gate = torch.ones(bsz, seqlen, heads, hd, dtype=torch.bfloat16, device=do.device)

    # Lane-free inputs (SHARED across lane CTAs -- the whole point).
    q_c = _deinterleave(scan["Q"]).unsqueeze(2).contiguous()      # [B,S,1,1,N]
    k_c = _deinterleave(scan["K"]).unsqueeze(2).contiguous()
    v_c = scan["V"].contiguous()
    qb = _deinterleave(q_bias).unsqueeze(1).contiguous().float()  # [H,1,N]
    kb = _deinterleave(k_bias).unsqueeze(1).contiguous().float()
    mimo_v = torch.ones((heads, 1, hd), device=dev, dtype=torch.float32)
    angles_c = angles_cumsum.float().contiguous()
    dt_c = scan["DT"].float().contiguous()
    trap_c = scan["Trap"].contiguous()
    dA_cs, dA_cs_rev, segsum = compute_dacs_segsum_triton(scan["ADT"].float().contiguous(), tcs)

    bf16, f32 = torch.bfloat16, torch.float32
    states = torch.empty([bsz, heads, nch, n_state, hd], dtype=bf16, device=dev)
    qk_dot = torch.zeros([bsz, heads, seqlen, 1, 1], dtype=bf16, device=dev)
    dout1 = torch.zeros(bsz, seqlen, 1, heads, hd, device=dev, dtype=bf16)
    dmo = torch.empty([bsz, heads, 1, hd], dtype=f32, device=dev)
    z_dum = torch.empty([bsz, seqlen, heads, hd], dtype=bf16, device=dev)
    mz_dum = torch.empty([heads, 1, hd], dtype=f32, device=dev)
    dz_dum = torch.empty([bsz, seqlen, heads, hd], dtype=bf16, device=dev)
    dmz_dum = torch.empty([bsz, heads, 1, hd], dtype=f32, device=dev)
    d_dum = torch.empty([heads], dtype=f32, device=dev)

    # Pass 1 ONCE: cotangent-independent STATES + QK_DOT.
    fwd_k = mamba_mimo_bwd_fwd(bsz, seqlen, heads, 1, n_state, hd, 1, False, False, False,
                               tcs, rdd, "bfloat16", 128, 0)
    fwd_k(dout1, q_c, k_c, v_c, qb, kb, mimo_v, mz_dum, dmo, states, z_dum, mz_dum,
          dz_dum, dmz_dum, angles_c, dA_cs, dA_cs_rev, dt_c, trap_c, d_dum, qk_dot, segsum)

    # Pass 2: lane-grid backward (per-lane outputs, zero replication).
    lg = num_p
    if chunk_parallel:
        # head-summed in-kernel via fp32 atomics -> reduced shape, MUST be zeros
        dq = torch.zeros([bsz, lg, seqlen, n_state], dtype=f32, device=dev)
        dk = torch.zeros_like(dq)
    else:
        dq = torch.empty([bsz, lg, seqlen, heads, n_state], dtype=bf16, device=dev)
        dk = torch.empty_like(dq)
    dv = torch.empty([bsz, lg, seqlen, heads, hd], dtype=bf16, device=dev)
    dmv = torch.empty([bsz, lg, heads, 1, hd], dtype=f32, device=dev)
    dd = torch.empty([bsz, lg, heads], dtype=f32, device=dev)
    # kernel fully overwrites every output -> empty, not zeros (dSS alone is GBs)
    dangles = torch.empty([bsz, lg, seqlen, heads, n_rope], dtype=f32, device=dev)
    dfactor = torch.empty([bsz, lg, heads, seqlen], dtype=f32, device=dev)
    dgamma = torch.empty([bsz, lg, heads, seqlen], dtype=f32, device=dev)
    ddA = torch.empty([bsz, lg, heads, seqlen], dtype=f32, device=dev)
    dSS = torch.empty([bsz, lg, heads, nch, tcs, tcs], dtype=f32, device=dev)
    ddacr = torch.empty([bsz, lg, heads, seqlen], dtype=f32, device=dev)
    ddacs = torch.empty([bsz, lg, heads, seqlen], dtype=f32, device=dev)
    dout = do  # already [B, LG, S, H, hd] -- the kernel's lane DOUT layout (no copy)

    # hasD=True: the D-skip dv term is computed in-kernel (valid in lane-grid --
    # R=1 uses the true v); the per-lane dD output is discarded.
    if chunk_parallel:
        from backbones.mamba3.ops.tilelang.mamba3.mamba3_chunkparallel_bwd import (
            mamba_chunkparallel_bwd_bwd,
            mamba_chunkparallel_bwd_dstates_local,
        )

        # Pass A (kernel): chunk-local dstates contributions
        # G_c = q_rot^T (dout*exp(dA_cs)), one CTA per (head, batch, lane, chunk).
        g_loc = torch.empty(bsz, lg, heads, nch, n_state, hd, dtype=bf16, device=dev)
        pa_k = mamba_chunkparallel_bwd_dstates_local(bsz, seqlen, heads, 1, n_state, hd,
                                                     lg, tcs, rdd, "bfloat16", 128)
        pa_k(dout, gate, q_c, qb, angles_c, dA_cs, g_loc)

        # Pass B (fused kernel): DS_IN = tri @ G with the strictly-upper
        # triangular decay matrix (tri[c, c'] = exp(LE_{c'-1} - LE_c) -- the
        # decay product EXCLUDES the source chunk) built in-kernel from the
        # cumulative chunk log-decay, PLUS the ddA state-passing term
        # (exp(da_sum_c) * <STATES_c, IN_c>) folded into the tile loop and
        # written chunk-broadcast into `ddA` directly.
        from backbones.mamba3.ops.tilelang.mamba3.mamba3_chunkparallel_bwd import (
            mamba_chunkparallel_bwd_dstates_combine,
        )

        le = dA_cs.view(bsz, heads, nch, tcs)[..., -1].cumsum(dim=-1).contiguous()
        if nch >= 16:
            np_flat = n_state * hd
            np_tile = 512 if nch <= 32 else 256
            ds_in = torch.empty(bsz, lg, heads, nch, n_state, hd, dtype=bf16, device=dev)
            pb_k = mamba_chunkparallel_bwd_dstates_combine(bsz, heads, np_flat, lg, nch,
                                                           tcs, np_tile, "bfloat16", 256)
            pb_k(g_loc.view(bsz, lg, heads, nch, np_flat), le,
                 states.view(bsz, heads, nch, np_flat),
                 ds_in.view(bsz, lg, heads, nch, np_flat), ddA)
        else:
            # Torch fallback for tiny chunk counts (the fused kernel's tri gemm
            # needs M = nch >= 16 for MMA warp partitioning).
            le_col = torch.nn.functional.pad(le, (1, 0))[..., :-1]
            d_mat = le_col.unsqueeze(-2) - le.unsqueeze(-1)
            mask = torch.ones(nch, nch, device=dev, dtype=torch.bool).triu(1)
            tri = torch.where(mask, torch.exp(d_mat), torch.zeros((), device=dev)).to(bf16)
            ds_in = tri.unsqueeze(1).matmul(
                g_loc.reshape(bsz, lg, heads, nch, n_state * hd)
            ).view(bsz, lg, heads, nch, n_state, hd).contiguous()
            e_chunk = torch.exp(dA_cs.view(bsz, heads, nch, tcs)[..., -1])
            part3 = torch.einsum("bhcnp,blhcnp->blhc", states.float(), ds_in.float())
            part3 = part3 * e_chunk.unsqueeze(1)
            ddA.copy_(part3.unsqueeze(-1).expand(bsz, lg, heads, nch, tcs)
                      .reshape(bsz, lg, heads, seqlen))

        # Pass C: all gradients, one CTA per (head, batch, lane, chunk).
        # chunk_parallel == "cuda" routes to the hand-CUDA kernel;
        # anything truthy else uses the tilelang kernel.
        dd_cp = torch.empty([bsz, lg, heads, nch], dtype=f32, device=dev)
        pass_c_args = (dout, gate, q_c, k_c, v_c, qb, kb, dk, dv, states, dq, ds_in,
                       angles_c, dA_cs, dA_cs_rev, dt_c, trap_c,
                       dfactor, dgamma, dangles, mixer.D.float().contiguous(), dd_cp,
                       qk_dot, dSS, ddacr, ddacs, segsum)
        if chunk_parallel in ("cuda", "cuda_mma"):
            from cuda.mamba3 import chunkparallel_pass_c

            chunkparallel_pass_c(*pass_c_args, use_mma=(chunk_parallel == "cuda_mma"))
        else:
            bwd_k = mamba_chunkparallel_bwd_bwd(bsz, seqlen, heads, 1, n_state, hd, lg, True,
                                                tcs, rdd, "bfloat16", 128)
            bwd_k(*pass_c_args)
    elif lane_tile:
        from backbones.mamba3.ops.tilelang.mamba3.mamba3_lanetile_bwd import (
            mamba_lanetile_bwd_bwd,
        )

        bwd_k = mamba_lanetile_bwd_bwd(bsz, seqlen, heads, 1, n_state, hd, lg,
                                       int(lane_tile), True, tcs, rdd, "bfloat16", 256, 0)
        bwd_k(dout, q_c, k_c, v_c, qb, kb, dk, dv, states, dq,
              angles_c, dA_cs, dA_cs_rev, dt_c, trap_c,
              dfactor, dgamma, dangles, mixer.D.float().contiguous(), dd, qk_dot,
              ddA, dSS, ddacr, ddacs, segsum)
    else:
        bwd_k = mamba_mimo_bwd_bwd(bsz, seqlen, heads, 1, n_state, hd, 1, False, True, False,
                                   tcs, rdd, "bfloat16", 256, 0, lane_scalars=False, lane_grid=lg)
        bwd_k(dout, q_c, k_c, v_c, qb, kb, mimo_v, mz_dum, dk, dv, dmv, states, dq,
              z_dum, mz_dum, angles_c, dA_cs, dA_cs_rev, dt_c, trap_c,
              dfactor, dgamma, dangles, mixer.D.float().contiguous(), dd, qk_dot, ddA, dSS, ddacr, ddacs, segsum)

    # Per-lane epilogues (small [B,H,S]-sized Triton calls per lane).
    # Batched epilogues: fold lanes into the BATCH dim -- lane tensors are
    # zero-copy [B*LG, ...] views of the kernel's [B, LG, ...] outputs; the small
    # shared tensors are tiled once (the Triton utils require packed slices --
    # strided lane views IMA -- so per-lane strided calls are not an option).
    def tile_b(x):
        # stride-0 lane-broadcast VIEW (no materialization): the Triton
        # epilogues accept a stride-0 batch (validated -- values match the
        # tiled path and no IMA; the packed-slice restriction applies to
        # strided lane SLICES, not broadcast batches).
        return x.unsqueeze(1).expand(bsz, lg, *x.shape[1:]).reshape(bsz * lg, *x.shape[1:])

    blg = bsz * lg
    ddt_f, dtrap_f = bwd_dtrap_ddt_triton(
        tile_b(trap_c), tile_b(dt_c), dfactor.view(blg, heads, seqlen),
        dgamma.view(blg, heads, seqlen), tcs,
    )
    ddA_f = ddA.view(blg, heads, seqlen) + bwd_dadt_fused_triton(
        dSS.view(blg, heads, nch, tcs, tcs), tile_b(segsum),
        ddacs.view(blg, heads, seqlen), ddacr.view(blg, heads, seqlen),
        tile_b(dA_cs), tile_b(dA_cs_rev), tcs,
    )
    dang_f, ddt_ang_f, _ = angle_dt_bwd(
        grad_out=dangles.view(blg, seqlen, heads, n_rope),
        angle=tile_b(scan["Angles"].contiguous()), dt=tile_b(scan["DT"].float().contiguous()),
        has_init_state=False, chunk_size=tcs, grad_output_state=None,
    )
    ddt_all = (ddt_f + ddt_ang_f).view(bsz, lg, heads, seqlen)
    dtrap_all = dtrap_f.view(bsz, lg, heads, seqlen)
    ddA_all = ddA_f.view(bsz, lg, heads, seqlen)
    dang_all = dang_f.view(bsz, lg, seqlen, heads, n_rope)

    # dq/dk: G=1 -> sum over heads (fp32 accumulation, no materialized casts);
    # interleave rotary bridge back.
    if chunk_parallel:
        dq_l = _interleave(dq).unsqueeze(3)   # already head-summed in-kernel
        dk_l = _interleave(dk).unsqueeze(3)
    else:
        dq_l = _interleave(dq.sum(dim=3, dtype=torch.float32)).unsqueeze(3)   # [B,P,S,1,N]
        dk_l = _interleave(dk.sum(dim=3, dtype=torch.float32)).unsqueeze(3)
    dv_l = dv  # D-skip term already included in-kernel (hasD=True)

    return {
        "Q": dq_l.to(dtype=scan["Q"].dtype),
        "K": dk_l.to(dtype=scan["K"].dtype),
        "V": dv_l.to(dtype=scan["V"].dtype),
        "ADT": ddA_all.to(dtype=scan["ADT"].dtype),
        "DT": ddt_all.to(dtype=scan["DT"].dtype),
        "Trap": dtrap_all.to(dtype=scan["Trap"].dtype),
        "Angles": dang_all.to(dtype=scan["Angles"].dtype),
        "Z": dz,
    }


def _rmsnorm_input_vjp(x_f32: torch.Tensor, weight: torch.Tensor, g_y: torch.Tensor,
                       eps: float) -> torch.Tensor:
    """Input-VJP of weight-only RMSNorm (fp32 upcast, group = full last dim),
    batched over a lane dim: x [B,L,N], g_y [B,P,L,N] -> [B,P,L,N] fp32."""
    n = x_f32.shape[-1]
    rstd = torch.rsqrt(x_f32.square().mean(dim=-1, keepdim=True) + eps)  # [B,L,1]
    h = g_y.float() * weight.float()                                    # [B,P,L,N]
    xb, rb = x_f32.unsqueeze(1), rstd.unsqueeze(1)
    return rb * h - xb * (rb.pow(3) / n) * (h * xb).sum(dim=-1, keepdim=True)


def mamba3_mixer_input_pullback_pbatched(
    *,
    mixer,
    cache,
    output_cotangent_basis: torch.Tensor,
    fused_epilogue: bool = True,
) -> torch.Tensor:
    """Stage C: mixer-level P-batched input pullback with TRUE preprocess-VJP
    epilogues.

    Maps a P-batched cotangent on the MIXER output [B, P, L, D] to the cotangent
    on the mixer input u, parameters held fixed. Composition: out_proj^T -> the
    lane-grid P-batched scan pullback (TRUE per-lane scan-input gradients) ->
    the TRUE preprocess VJP (in_proj/split/norms/softplus-clamp/angle-expand).
    NOTE: the custom epilogues in `mamba3_mixer_input_pullback_native` consume
    the per-lane Triton chain's MID-SPACE intermediates and must NOT be used
    here -- the lane-grid pullback emits true gradients.

    `fused_epilogue=True` (default) uses the closed-form lane-BATCHED preprocess
    VJP (one pass for all P lanes); False falls back to scoped-local autograd
    with one backward per lane (the validation oracle for the fused path)."""
    from einops import rearrange
    import torch.nn.functional as F

    if output_cotangent_basis.dim() != 4:
        raise ValueError("output_cotangent_basis must have shape [B, P, L, D]")
    bsz, num_p, seqlen, _ = output_cotangent_basis.shape
    heads, hd = mixer.nheads, mixer.headdim
    d_inner, d_state = mixer.d_inner, mixer.d_state

    # 1. out_proj^T -> scan-output cotangent lanes [B, P, L, H, hd]
    g_yinner = output_cotangent_basis.to(mixer.out_proj.weight.dtype) @ mixer.out_proj.weight
    g_scan_out = g_yinner.reshape(bsz, num_p, seqlen, heads, hd).contiguous()

    # 2. P-batched TRUE scan pullback (per-lane all eight adjoints)
    adj = mamba3_siso_pbatched_pullback_lanegrid(
        mixer=mixer, cache=cache, output_cotangent_basis=g_scan_out
    )

    u = cache.input_u.detach()
    sizes = [d_inner, d_inner, d_state, d_state, heads, heads, heads, mixer.num_rope_angles]

    if fused_epilogue:
        # 3. Closed-form preprocess VJP, ALL lanes in one batched pass. The
        # forward re-run is grad-free; each branch's adjoint is hand-derived
        # and concatenated back into zx-space, then one matmul to u-space.
        with torch.no_grad():
            zx = mixer.in_proj(u)
            _z, _x, b_pre, c_pre, dd_dt, dd_A, _trap, _angles = torch.split(zx, sizes, dim=-1)
            dt_arg = (dd_dt + mixer.dt_bias).float()
            sig_dt = torch.sigmoid(dt_arg)                       # [B,L,H]
            dt_f = F.softplus(dt_arg)
            neg_sp_A = -F.softplus(dd_A.float())
            a_neg = torch.clamp(neg_sp_A, max=-mixer.A_floor)
            clamp_pass = (neg_sp_A <= -mixer.A_floor).float()    # d clamp(v,max=M)/dv (grad passes at equality, matching torch)
            sig_A = torch.sigmoid(dd_A.float())

            g_q = adj["Q"].float().squeeze(3)                    # [B,P,L,N]
            g_k = adj["K"].float().squeeze(3)
            g_c_pre = _rmsnorm_input_vjp(c_pre.float(), mixer.C_norm.weight, g_q, mixer.C_norm.eps)
            g_b_pre = _rmsnorm_input_vjp(b_pre.float(), mixer.B_norm.weight, g_k, mixer.B_norm.eps)

            g_adt = adj["ADT"].float().permute(0, 1, 3, 2)       # [B,P,H,S] -> [B,P,L,H]
            g_dt = adj["DT"].float().permute(0, 1, 3, 2)
            au = a_neg.unsqueeze(1)
            g_dd_dt = (g_dt + g_adt * au) * sig_dt.unsqueeze(1)
            g_dd_A = g_adt * dt_f.unsqueeze(1) * (-sig_A * clamp_pass).unsqueeze(1)

            # Assemble zx-space adjoint in the projection dtype (matches the true
            # graph: the split backward concatenates bf16 grads). Slice-writes
            # into one buffer avoid the fp32 torch.cat blowup (GBs at scale).
            w = mixer.in_proj.weight
            d_tot = sum(sizes)
            g_zx = torch.empty(bsz, num_p, seqlen, d_tot, device=u.device, dtype=w.dtype)
            offs = [0]
            for s in sizes:
                offs.append(offs[-1] + s)
            g_zx[..., offs[0]:offs[1]] = adj["Z"].reshape(bsz, num_p, seqlen, d_inner)
            g_zx[..., offs[1]:offs[2]] = adj["V"].reshape(bsz, num_p, seqlen, d_inner)
            g_zx[..., offs[2]:offs[3]] = g_b_pre
            g_zx[..., offs[3]:offs[4]] = g_c_pre
            g_zx[..., offs[4]:offs[5]] = g_dd_dt
            g_zx[..., offs[5]:offs[6]] = g_dd_A
            g_zx[..., offs[6]:offs[7]] = adj["Trap"].permute(0, 1, 3, 2)
            g_zx[..., offs[7]:offs[8]] = adj["Angles"].float().sum(dim=3)  # expand over H -> sum
            g_u = g_zx @ w
        return g_u.to(dtype=output_cotangent_basis.dtype)

    # 3'. Oracle path: scoped-local autograd, one backward per lane.
    with torch.enable_grad():
        u_leaf = u.requires_grad_(True)
        zx = mixer.in_proj(u_leaf)
        z, x, b_pre, c_pre, dd_dt, dd_A, trap, angles = torch.split(zx, sizes, dim=-1)
        z = rearrange(z, "b l (h p) -> b l h p", p=hd)
        x = rearrange(x, "b l (h p) -> b l h p", p=hd)
        b_n = mixer.B_norm(rearrange(b_pre, "b l (r g n) -> b l r g n", r=1, g=1))
        c_n = mixer.C_norm(rearrange(c_pre, "b l (r g n) -> b l r g n", r=1, g=1))
        trap_t = rearrange(trap, "b l h -> b h l")
        a_neg = torch.clamp(-F.softplus(dd_A.to(torch.float32)), max=-mixer.A_floor)
        dt_t = F.softplus(dd_dt + mixer.dt_bias)
        adt_t = rearrange(a_neg * dt_t, "b l n -> b n l")
        dt_t = rearrange(dt_t, "b l n -> b n l")
        angles_e = angles.unsqueeze(-2).expand(-1, -1, heads, -1)
        outputs = [c_n.squeeze(2), b_n.squeeze(2), x, adt_t, dt_t, trap_t, angles_e, z]

        lanes = []
        for j in range(num_p):
            seeds = [
                adj["Q"][:, j].to(outputs[0].dtype),
                adj["K"][:, j].to(outputs[1].dtype),
                adj["V"][:, j].to(outputs[2].dtype),
                adj["ADT"][:, j].to(outputs[3].dtype),
                adj["DT"][:, j].to(outputs[4].dtype),
                adj["Trap"][:, j].to(outputs[5].dtype),
                adj["Angles"][:, j].to(outputs[6].dtype),
                adj["Z"][:, j].to(outputs[7].dtype),
            ]
            (g_u,) = torch.autograd.grad(outputs, u_leaf, grad_outputs=seeds, retain_graph=True)
            lanes.append(g_u.unsqueeze(1))
    return torch.cat(lanes, dim=1).to(dtype=output_cotangent_basis.dtype)

