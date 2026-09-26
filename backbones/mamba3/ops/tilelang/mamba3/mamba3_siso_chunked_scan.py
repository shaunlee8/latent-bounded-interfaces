"""Fused tilelang chunked-scan kernels for the forward-mode SISO JVP.

The kernels carry the augmented (S, dS) state through the chunked scan on
tensor cores (chunk-local quadratic plus inter-chunk (S, dS) passing) and
emit the scan output and its JVP for all r directions at once; variants
generate the tangents on chip from raw group-level tangents (gen), fold the
finalize and sequence mean into the epilogue (gen_mean), or batch several
directions per launch. Field preparation and the torch-side finalize live in
`_prepare_dual_fields` and `_finalize`."""

import builtins

import torch
import torch.nn.functional as F

try:  # tilelang is only installed in the kernel-dev env; keep import optional.
    import tilelang
    import tilelang.language as T
    _HAS_TILELANG = True
except Exception:  # pragma: no cover - environment without tilelang
    _HAS_TILELANG = False

from backbones.mamba3.ops.triton.mamba3.mamba3_siso_chunked_scan_ref import (
    _prepare_dual_fields,
    _prepare_primal_fields,
    _prepare_tangent_fields,
    _prepare_tangent_fields_batched,
    _finalize,
    _finalize_batched,
)

_KERNEL_CACHE: dict = {}
_COMPILED_CACHE: dict = {}


def _maybe_compile(fn, name: str):
    """torch.compile the batched field prep so elementwise tangent chains fuse
    and [r, S, BH, N] intermediates never materialize. Compile failures latch
    an eager fallback at call time."""
    if name not in _COMPILED_CACHE:
        try:
            compiled = torch.compile(fn, dynamic=False)
        except Exception:
            _COMPILED_CACHE[name] = fn
            return fn
        state = {"failed": False}

        def wrapper(*args, **kwargs):
            if state["failed"]:
                return fn(*args, **kwargs)
            try:
                return compiled(*args, **kwargs)
            except Exception:
                state["failed"] = True
                return fn(*args, **kwargs)

        _COMPILED_CACHE[name] = wrapper
    return _COMPILED_CACHE[name]


if _HAS_TILELANG:
    @tilelang.jit(out_idx=[8, 9], pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    })
    def _chunked_scan_kernel(B, S, H, N, P, chunk_size, dtype='bfloat16', threads=256, num_stages=0):
        # threads=256: fragments spread over more threads halve per-thread
        # registers.
        acc = 'float32'
        assert S % chunk_size == 0
        nc = S // chunk_size
        cs = chunk_size

        @T.prim_func
        def kernel(
            QR: T.Tensor([B, S, H, N], dtype),
            DQR: T.Tensor([B, S, H, N], dtype),
            KSC: T.Tensor([B, S, H, N], dtype),
            DKSC: T.Tensor([B, S, H, N], dtype),
            V: T.Tensor([B, S, H, P], dtype),
            DV: T.Tensor([B, S, H, P], dtype),
            L: T.Tensor([B, H, S], acc),
            DL: T.Tensor([B, H, S], acc),
            OUT: T.Tensor([B, S, H, P], acc),
            DOUT: T.Tensor([B, S, H, P], acc),
        ):
            with T.Kernel(H, B, threads=threads) as (i_h, i_b):
                S_f = T.alloc_fragment([N, P], acc)
                dS_f = T.alloc_fragment([N, P], acc)
                T.clear(S_f); T.clear(dS_f)

                qr_s = T.alloc_shared([cs, N], dtype); dqr_s = T.alloc_shared([cs, N], dtype)
                ksc_s = T.alloc_shared([cs, N], dtype); dksc_s = T.alloc_shared([cs, N], dtype)
                v_s = T.alloc_shared([cs, P], dtype); dv_s = T.alloc_shared([cs, P], dtype)
                S_s = T.alloc_shared([N, P], dtype); dS_s = T.alloc_shared([N, P], dtype)
                L_s = T.alloc_shared([cs], acc); dL_s = T.alloc_shared([cs], acc)
                WQK_s = T.alloc_shared([cs, cs], dtype); dWQK_s = T.alloc_shared([cs, cs], dtype)
                wksc_s = T.alloc_shared([cs, N], dtype); scr_s = T.alloc_shared([cs, N], dtype)

                # Swizzled smem layouts on the tensor-core GEMM operands
                # (bank-conflict reduction) plus L2 row rasterization.
                T.annotate_layout({
                    qr_s: tilelang.layout.make_swizzled_layout(qr_s),
                    dqr_s: tilelang.layout.make_swizzled_layout(dqr_s),
                    ksc_s: tilelang.layout.make_swizzled_layout(ksc_s),
                    dksc_s: tilelang.layout.make_swizzled_layout(dksc_s),
                    v_s: tilelang.layout.make_swizzled_layout(v_s),
                    dv_s: tilelang.layout.make_swizzled_layout(dv_s),
                    S_s: tilelang.layout.make_swizzled_layout(S_s),
                    dS_s: tilelang.layout.make_swizzled_layout(dS_s),
                    WQK_s: tilelang.layout.make_swizzled_layout(WQK_s),
                    dWQK_s: tilelang.layout.make_swizzled_layout(dWQK_s),
                    wksc_s: tilelang.layout.make_swizzled_layout(wksc_s),
                    scr_s: tilelang.layout.make_swizzled_layout(scr_s),
                })
                T.use_swizzle(10, "row")

                for i in T.Pipelined(0, nc, num_stages=num_stages):
                    c0 = i * cs
                    T.copy(L[i_b, i_h, c0:c0 + cs], L_s)
                    T.copy(DL[i_b, i_h, c0:c0 + cs], dL_s)
                    T.copy(QR[i_b, c0:c0 + cs, i_h, :], qr_s)
                    T.copy(DQR[i_b, c0:c0 + cs, i_h, :], dqr_s)
                    T.copy(KSC[i_b, c0:c0 + cs, i_h, :], ksc_s)
                    T.copy(DKSC[i_b, c0:c0 + cs, i_h, :], dksc_s)
                    T.copy(V[i_b, c0:c0 + cs, i_h, :], v_s)
                    T.copy(DV[i_b, c0:c0 + cs, i_h, :], dv_s)

                    # intra QK / dQK  [cs, cs]
                    QK = T.alloc_fragment([cs, cs], acc)
                    dQK = T.alloc_fragment([cs, cs], acc)
                    T.gemm(qr_s, ksc_s, QK, transpose_B=True, clear_accum=True)
                    T.gemm(dqr_s, ksc_s, dQK, transpose_B=True, clear_accum=True)
                    T.gemm(qr_s, dksc_s, dQK, transpose_B=True, clear_accum=False)

                    WQK = T.alloc_fragment([cs, cs], acc)
                    dWQK = T.alloc_fragment([cs, cs], acc)
                    for ii, jj in T.Parallel(cs, cs):
                        w = T.if_then_else(ii >= jj, T.exp(L_s[ii] - L_s[jj]), 0.0)
                        WQK[ii, jj] = w * QK[ii, jj]
                        dWQK[ii, jj] = w * (dL_s[ii] - dL_s[jj]) * QK[ii, jj] + w * dQK[ii, jj]
                    T.copy(WQK, WQK_s)
                    T.copy(dWQK, dWQK_s)

                    # intra out / dout  [cs, P]
                    out_f = T.alloc_fragment([cs, P], acc)
                    dout_f = T.alloc_fragment([cs, P], acc)
                    T.gemm(WQK_s, v_s, out_f, clear_accum=True)
                    T.gemm(dWQK_s, v_s, dout_f, clear_accum=True)
                    T.gemm(WQK_s, dv_s, dout_f, clear_accum=False)

                    # inter: qsin / dqsin / qdsin  [cs, P]
                    T.copy(S_f, S_s); T.copy(dS_f, dS_s)
                    qsin = T.alloc_fragment([cs, P], acc)
                    dqsin = T.alloc_fragment([cs, P], acc)
                    qdsin = T.alloc_fragment([cs, P], acc)
                    T.gemm(qr_s, S_s, qsin, clear_accum=True)
                    T.gemm(dqr_s, S_s, dqsin, clear_accum=True)
                    T.gemm(qr_s, dS_s, qdsin, clear_accum=True)
                    for ii, pp in T.Parallel(cs, P):
                        e = T.exp(L_s[ii])
                        out_f[ii, pp] += e * qsin[ii, pp]
                        dout_f[ii, pp] += e * (dL_s[ii] * qsin[ii, pp] + dqsin[ii, pp] + qdsin[ii, pp])

                    T.copy(out_f, OUT[i_b, c0:c0 + cs, i_h, :])
                    T.copy(dout_f, DOUT[i_b, c0:c0 + cs, i_h, :])

                    # --- state update (dS uses the previous S_f for the dchunk_decay term) ---
                    Sc = T.alloc_fragment([N, P], acc)
                    dSc = T.alloc_fragment([N, P], acc)
                    for jj, nn in T.Parallel(cs, N):
                        wl = T.exp(L_s[cs - 1] - L_s[jj])
                        wksc_s[jj, nn] = wl * ksc_s[jj, nn]
                    T.gemm(wksc_s, v_s, Sc, transpose_A=True, clear_accum=True)
                    for jj, nn in T.Parallel(cs, N):
                        wl = T.exp(L_s[cs - 1] - L_s[jj])
                        scr_s[jj, nn] = wl * (dL_s[cs - 1] - dL_s[jj]) * ksc_s[jj, nn]
                    T.gemm(scr_s, v_s, dSc, transpose_A=True, clear_accum=True)
                    for jj, nn in T.Parallel(cs, N):
                        wl = T.exp(L_s[cs - 1] - L_s[jj])
                        scr_s[jj, nn] = wl * dksc_s[jj, nn]
                    T.gemm(scr_s, v_s, dSc, transpose_A=True, clear_accum=False)
                    T.gemm(wksc_s, dv_s, dSc, transpose_A=True, clear_accum=False)

                    for nn, pp in T.Parallel(N, P):
                        cd = T.exp(L_s[cs - 1])
                        dcd = cd * dL_s[cs - 1]
                        dS_new = cd * dS_f[nn, pp] + dcd * S_f[nn, pp] + dSc[nn, pp]
                        S_new = cd * S_f[nn, pp] + Sc[nn, pp]
                        dS_f[nn, pp] = dS_new
                        S_f[nn, pp] = S_new

        return kernel

    @tilelang.jit(out_idx=[8, 9], pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    })
    def _chunked_scan_lbi_kernel(B, S, H, N, P, chunk_size, rb, dtype='bfloat16', threads=256):
        """Lane-ILP base kernel: rb tangent directions share the primal (operands
        loaded and primal GEMMs computed once) with per-direction buffers/fragments
        (no WAR hazards); the direction loop is T.unroll(rb) so the directions' pipelines
        interleave. Bit-composes with rb=1 (`_chunked_scan_kernel`) up to bf16
        GEMM scheduling."""
        acc = 'float32'
        assert S % chunk_size == 0
        assert rb == 2, "direction-ILP spike is inlined for rb=2"
        nc = S // chunk_size
        cs = chunk_size

        @T.prim_func
        def kernel(
            QR: T.Tensor([B, S, H, N], dtype),
            KSC: T.Tensor([B, S, H, N], dtype),
            V: T.Tensor([B, S, H, P], dtype),
            DQR: T.Tensor([rb, B, S, H, N], dtype),
            DKSC: T.Tensor([rb, B, S, H, N], dtype),
            DV: T.Tensor([rb, B, S, H, P], dtype),
            L: T.Tensor([B, H, S], acc),
            DL: T.Tensor([rb, B, H, S], acc),
            OUT: T.Tensor([B, S, H, P], acc),
            DOUT: T.Tensor([rb, B, S, H, P], acc),
        ):
            with T.Kernel(H, B, threads=threads) as (i_h, i_b):
                # shared primal state + per-direction tangent state as python lists
                # of 2D buffers (gemm-C must be 2D; sliced fragments rejected).
                RB = tuple(builtins.range(rb))
                S_f = T.alloc_fragment([N, P], acc); T.clear(S_f)
                dS_f = [T.alloc_fragment([N, P], acc) for _ in RB]
                qr_s = T.alloc_shared([cs, N], dtype); ksc_s = T.alloc_shared([cs, N], dtype)
                v_s = T.alloc_shared([cs, P], dtype)
                S_s = T.alloc_shared([N, P], dtype)
                L_s = T.alloc_shared([cs], acc); WQK_s = T.alloc_shared([cs, cs], dtype)
                wksc_s = T.alloc_shared([cs, N], dtype)
                # per-direction INDEPENDENT smem + accumulator fragments (no reuse).
                dqr_s = [T.alloc_shared([cs, N], dtype) for _ in RB]
                dksc_s = [T.alloc_shared([cs, N], dtype) for _ in RB]
                dv_s = [T.alloc_shared([cs, P], dtype) for _ in RB]
                dS_s = [T.alloc_shared([N, P], dtype) for _ in RB]
                dL_s = [T.alloc_shared([cs], acc) for _ in RB]
                dWQK_s = [T.alloc_shared([cs, cs], dtype) for _ in RB]
                scr_s = [T.alloc_shared([cs, N], dtype) for _ in RB]
                dQK = [T.alloc_fragment([cs, cs], acc) for _ in RB]
                dWQK = [T.alloc_fragment([cs, cs], acc) for _ in RB]
                dout_f = [T.alloc_fragment([cs, P], acc) for _ in RB]
                dqsin = [T.alloc_fragment([cs, P], acc) for _ in RB]
                qdsin = [T.alloc_fragment([cs, P], acc) for _ in RB]
                dSc = [T.alloc_fragment([N, P], acc) for _ in RB]
                _ = [T.clear(dS_f[b]) for b in RB]

                _sw = tilelang.layout.make_swizzled_layout
                lyt = {b: _sw(b) for b in [qr_s, ksc_s, v_s, S_s, WQK_s, wksc_s]}
                lyt.update({buf: _sw(buf)
                            for lst in (dqr_s, dksc_s, dv_s, dS_s, dWQK_s, scr_s)
                            for buf in lst})
                T.annotate_layout(lyt)
                T.use_swizzle(10, "row")

                for i in T.Pipelined(0, nc, num_stages=0):
                    c0 = i * cs
                    # Shared primal loads and GEMMs run once per direction-block.
                    T.copy(L[i_b, i_h, c0:c0 + cs], L_s)
                    T.copy(QR[i_b, c0:c0 + cs, i_h, :], qr_s)
                    T.copy(KSC[i_b, c0:c0 + cs, i_h, :], ksc_s)
                    T.copy(V[i_b, c0:c0 + cs, i_h, :], v_s)
                    QK = T.alloc_fragment([cs, cs], acc)
                    T.gemm(qr_s, ksc_s, QK, transpose_B=True, clear_accum=True)
                    WQK = T.alloc_fragment([cs, cs], acc)
                    for ii, jj in T.Parallel(cs, cs):
                        w = T.if_then_else(ii >= jj, T.exp(L_s[ii] - L_s[jj]), 0.0)
                        WQK[ii, jj] = w * QK[ii, jj]
                    T.copy(WQK, WQK_s)
                    out_f = T.alloc_fragment([cs, P], acc)
                    T.gemm(WQK_s, v_s, out_f, clear_accum=True)
                    T.copy(S_f, S_s)
                    qsin = T.alloc_fragment([cs, P], acc)
                    T.gemm(qr_s, S_s, qsin, clear_accum=True)
                    for ii, pp in T.Parallel(cs, P):
                        out_f[ii, pp] += T.exp(L_s[ii]) * qsin[ii, pp]
                    T.copy(out_f, OUT[i_b, c0:c0 + cs, i_h, :])
                    Sc = T.alloc_fragment([N, P], acc)
                    for jj, nn in T.Parallel(cs, N):
                        wl = T.exp(L_s[cs - 1] - L_s[jj])
                        wksc_s[jj, nn] = wl * ksc_s[jj, nn]
                    T.gemm(wksc_s, v_s, Sc, transpose_A=True, clear_accum=True)

                    # both directions' independent loads issue first so the two
                    # pipelines interleave.
                    T.copy(DL[0, i_b, i_h, c0:c0 + cs], dL_s[0]); T.copy(DL[1, i_b, i_h, c0:c0 + cs], dL_s[1])
                    T.copy(DQR[0, i_b, c0:c0 + cs, i_h, :], dqr_s[0]); T.copy(DQR[1, i_b, c0:c0 + cs, i_h, :], dqr_s[1])
                    T.copy(DKSC[0, i_b, c0:c0 + cs, i_h, :], dksc_s[0]); T.copy(DKSC[1, i_b, c0:c0 + cs, i_h, :], dksc_s[1])
                    T.copy(DV[0, i_b, c0:c0 + cs, i_h, :], dv_s[0]); T.copy(DV[1, i_b, c0:c0 + cs, i_h, :], dv_s[1])
                    T.copy(dS_f[0], dS_s[0]); T.copy(dS_f[1], dS_s[1])

                    # dQK: both directions (each 2 gemms into its OWN accumulator).
                    T.gemm(dqr_s[0], ksc_s, dQK[0], transpose_B=True, clear_accum=True)
                    T.gemm(qr_s, dksc_s[0], dQK[0], transpose_B=True, clear_accum=False)
                    T.gemm(dqr_s[1], ksc_s, dQK[1], transpose_B=True, clear_accum=True)
                    T.gemm(qr_s, dksc_s[1], dQK[1], transpose_B=True, clear_accum=False)
                    for ii, jj in T.Parallel(cs, cs):
                        w = T.if_then_else(ii >= jj, T.exp(L_s[ii] - L_s[jj]), 0.0)
                        dWQK[0][ii, jj] = w * (dL_s[0][ii] - dL_s[0][jj]) * QK[ii, jj] + w * dQK[0][ii, jj]
                        dWQK[1][ii, jj] = w * (dL_s[1][ii] - dL_s[1][jj]) * QK[ii, jj] + w * dQK[1][ii, jj]
                    T.copy(dWQK[0], dWQK_s[0]); T.copy(dWQK[1], dWQK_s[1])
                    # dout: both directions.
                    T.gemm(dWQK_s[0], v_s, dout_f[0], clear_accum=True)
                    T.gemm(WQK_s, dv_s[0], dout_f[0], clear_accum=False)
                    T.gemm(dWQK_s[1], v_s, dout_f[1], clear_accum=True)
                    T.gemm(WQK_s, dv_s[1], dout_f[1], clear_accum=False)
                    # dqsin / qdsin: both directions (state read).
                    T.gemm(dqr_s[0], S_s, dqsin[0], clear_accum=True)
                    T.gemm(qr_s, dS_s[0], qdsin[0], clear_accum=True)
                    T.gemm(dqr_s[1], S_s, dqsin[1], clear_accum=True)
                    T.gemm(qr_s, dS_s[1], qdsin[1], clear_accum=True)
                    for ii, pp in T.Parallel(cs, P):
                        e = T.exp(L_s[ii])
                        dout_f[0][ii, pp] += e * (dL_s[0][ii] * qsin[ii, pp] + dqsin[0][ii, pp] + qdsin[0][ii, pp])
                        dout_f[1][ii, pp] += e * (dL_s[1][ii] * qsin[ii, pp] + dqsin[1][ii, pp] + qdsin[1][ii, pp])
                    T.copy(dout_f[0], DOUT[0, i_b, c0:c0 + cs, i_h, :]); T.copy(dout_f[1], DOUT[1, i_b, c0:c0 + cs, i_h, :])
                    # dSc: both directions (three weighted gemms each into own accum).
                    for jj, nn in T.Parallel(cs, N):
                        wl = T.exp(L_s[cs - 1] - L_s[jj])
                        scr_s[0][jj, nn] = wl * (dL_s[0][cs - 1] - dL_s[0][jj]) * ksc_s[jj, nn]
                        scr_s[1][jj, nn] = wl * (dL_s[1][cs - 1] - dL_s[1][jj]) * ksc_s[jj, nn]
                    T.gemm(scr_s[0], v_s, dSc[0], transpose_A=True, clear_accum=True)
                    T.gemm(scr_s[1], v_s, dSc[1], transpose_A=True, clear_accum=True)
                    for jj, nn in T.Parallel(cs, N):
                        wl = T.exp(L_s[cs - 1] - L_s[jj])
                        scr_s[0][jj, nn] = wl * dksc_s[0][jj, nn]
                        scr_s[1][jj, nn] = wl * dksc_s[1][jj, nn]
                    T.gemm(scr_s[0], v_s, dSc[0], transpose_A=True, clear_accum=False)
                    T.gemm(scr_s[1], v_s, dSc[1], transpose_A=True, clear_accum=False)
                    T.gemm(wksc_s, dv_s[0], dSc[0], transpose_A=True, clear_accum=False)
                    T.gemm(wksc_s, dv_s[1], dSc[1], transpose_A=True, clear_accum=False)
                    for nn, pp in T.Parallel(N, P):
                        cd = T.exp(L_s[cs - 1])
                        dcd = cd * dL_s[0][cs - 1]
                        dS_f[0][nn, pp] = cd * dS_f[0][nn, pp] + dcd * S_f[nn, pp] + dSc[0][nn, pp]
                    for nn, pp in T.Parallel(N, P):
                        cd = T.exp(L_s[cs - 1])
                        dcd = cd * dL_s[1][cs - 1]
                        dS_f[1][nn, pp] = cd * dS_f[1][nn, pp] + dcd * S_f[nn, pp] + dSc[1][nn, pp]

                    for nn, pp in T.Parallel(N, P):
                        S_f[nn, pp] = T.exp(L_s[cs - 1]) * S_f[nn, pp] + Sc[nn, pp]

        return kernel


if _HAS_TILELANG:
    @tilelang.jit(out_idx=[6, 7], pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    })
    def _chunked_scan_passA_kernel(B, S, H, N, P, chunk_size, dtype='bfloat16', threads=128):
        """Chunk-parallel pass A: per-chunk (S, dS) state CONTRIBUTIONS,
        one CTA per (h, b, chunk). Fully parallel; no serial chunk chain."""
        acc = 'float32'
        assert S % chunk_size == 0
        nc = S // chunk_size
        cs = chunk_size

        @T.prim_func
        def kernel(
            KSC: T.Tensor([B, S, H, N], dtype),
            DKSC: T.Tensor([B, S, H, N], dtype),
            V: T.Tensor([B, S, H, P], dtype),
            DV: T.Tensor([B, S, H, P], dtype),
            L: T.Tensor([B, H, S], acc),
            DL: T.Tensor([B, H, S], acc),
            SC: T.Tensor([B, H, nc, N, P], acc),
            DSC: T.Tensor([B, H, nc, N, P], acc),
        ):
            with T.Kernel(H, B, nc, threads=threads) as (i_h, i_b, i_c):
                ksc_s = T.alloc_shared([cs, N], dtype); dksc_s = T.alloc_shared([cs, N], dtype)
                v_s = T.alloc_shared([cs, P], dtype); dv_s = T.alloc_shared([cs, P], dtype)
                wksc_s = T.alloc_shared([cs, N], dtype); scr_s = T.alloc_shared([cs, N], dtype)
                L_s = T.alloc_shared([cs], acc); dL_s = T.alloc_shared([cs], acc)
                T.annotate_layout({
                    ksc_s: tilelang.layout.make_swizzled_layout(ksc_s),
                    dksc_s: tilelang.layout.make_swizzled_layout(dksc_s),
                    v_s: tilelang.layout.make_swizzled_layout(v_s),
                    dv_s: tilelang.layout.make_swizzled_layout(dv_s),
                    wksc_s: tilelang.layout.make_swizzled_layout(wksc_s),
                    scr_s: tilelang.layout.make_swizzled_layout(scr_s),
                })
                T.use_swizzle(10, "row")

                c0 = i_c * cs
                T.copy(L[i_b, i_h, c0:c0 + cs], L_s)
                T.copy(DL[i_b, i_h, c0:c0 + cs], dL_s)
                T.copy(KSC[i_b, c0:c0 + cs, i_h, :], ksc_s)
                T.copy(DKSC[i_b, c0:c0 + cs, i_h, :], dksc_s)
                T.copy(V[i_b, c0:c0 + cs, i_h, :], v_s)
                T.copy(DV[i_b, c0:c0 + cs, i_h, :], dv_s)

                Sc = T.alloc_fragment([N, P], acc)
                dSc = T.alloc_fragment([N, P], acc)
                for jj, nn in T.Parallel(cs, N):
                    wl = T.exp(L_s[cs - 1] - L_s[jj])
                    wksc_s[jj, nn] = wl * ksc_s[jj, nn]
                T.gemm(wksc_s, v_s, Sc, transpose_A=True, clear_accum=True)
                for jj, nn in T.Parallel(cs, N):
                    wl = T.exp(L_s[cs - 1] - L_s[jj])
                    scr_s[jj, nn] = wl * (dL_s[cs - 1] - dL_s[jj]) * ksc_s[jj, nn]
                T.gemm(scr_s, v_s, dSc, transpose_A=True, clear_accum=True)
                for jj, nn in T.Parallel(cs, N):
                    wl = T.exp(L_s[cs - 1] - L_s[jj])
                    scr_s[jj, nn] = wl * dksc_s[jj, nn]
                T.gemm(scr_s, v_s, dSc, transpose_A=True, clear_accum=False)
                T.gemm(wksc_s, dv_s, dSc, transpose_A=True, clear_accum=False)

                T.copy(Sc, SC[i_b, i_h, i_c, :, :])
                T.copy(dSc, DSC[i_b, i_h, i_c, :, :])

        return kernel

    @tilelang.jit(out_idx=[10, 11], pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    })
    def _chunked_scan_passC_kernel(B, S, H, N, P, chunk_size, dtype='bfloat16', threads=128):
        """Chunk-parallel pass C: intra (chunk-local quadratic) + inter
        (entering-state readout from the pass-B scanned S_in/dS_in), one CTA per
        (h, b, chunk). Fully parallel."""
        acc = 'float32'
        assert S % chunk_size == 0
        nc = S // chunk_size
        cs = chunk_size

        @T.prim_func
        def kernel(
            QR: T.Tensor([B, S, H, N], dtype),
            DQR: T.Tensor([B, S, H, N], dtype),
            KSC: T.Tensor([B, S, H, N], dtype),
            DKSC: T.Tensor([B, S, H, N], dtype),
            V: T.Tensor([B, S, H, P], dtype),
            DV: T.Tensor([B, S, H, P], dtype),
            L: T.Tensor([B, H, S], acc),
            DL: T.Tensor([B, H, S], acc),
            SIN: T.Tensor([B, H, nc, N, P], acc),
            DSIN: T.Tensor([B, H, nc, N, P], acc),
            OUT: T.Tensor([B, S, H, P], acc),
            DOUT: T.Tensor([B, S, H, P], acc),
        ):
            with T.Kernel(H, B, nc, threads=threads) as (i_h, i_b, i_c):
                qr_s = T.alloc_shared([cs, N], dtype); dqr_s = T.alloc_shared([cs, N], dtype)
                ksc_s = T.alloc_shared([cs, N], dtype); dksc_s = T.alloc_shared([cs, N], dtype)
                v_s = T.alloc_shared([cs, P], dtype); dv_s = T.alloc_shared([cs, P], dtype)
                S_s = T.alloc_shared([N, P], dtype); dS_s = T.alloc_shared([N, P], dtype)
                L_s = T.alloc_shared([cs], acc); dL_s = T.alloc_shared([cs], acc)
                WQK_s = T.alloc_shared([cs, cs], dtype); dWQK_s = T.alloc_shared([cs, cs], dtype)
                T.annotate_layout({
                    qr_s: tilelang.layout.make_swizzled_layout(qr_s),
                    dqr_s: tilelang.layout.make_swizzled_layout(dqr_s),
                    ksc_s: tilelang.layout.make_swizzled_layout(ksc_s),
                    dksc_s: tilelang.layout.make_swizzled_layout(dksc_s),
                    v_s: tilelang.layout.make_swizzled_layout(v_s),
                    dv_s: tilelang.layout.make_swizzled_layout(dv_s),
                    S_s: tilelang.layout.make_swizzled_layout(S_s),
                    dS_s: tilelang.layout.make_swizzled_layout(dS_s),
                    WQK_s: tilelang.layout.make_swizzled_layout(WQK_s),
                    dWQK_s: tilelang.layout.make_swizzled_layout(dWQK_s),
                })
                T.use_swizzle(10, "row")

                c0 = i_c * cs
                T.copy(L[i_b, i_h, c0:c0 + cs], L_s)
                T.copy(DL[i_b, i_h, c0:c0 + cs], dL_s)
                T.copy(QR[i_b, c0:c0 + cs, i_h, :], qr_s)
                T.copy(DQR[i_b, c0:c0 + cs, i_h, :], dqr_s)
                T.copy(KSC[i_b, c0:c0 + cs, i_h, :], ksc_s)
                T.copy(DKSC[i_b, c0:c0 + cs, i_h, :], dksc_s)
                T.copy(V[i_b, c0:c0 + cs, i_h, :], v_s)
                T.copy(DV[i_b, c0:c0 + cs, i_h, :], dv_s)
                T.copy(SIN[i_b, i_h, i_c, :, :], S_s)
                T.copy(DSIN[i_b, i_h, i_c, :, :], dS_s)

                # intra QK / dQK  [cs, cs]
                QK = T.alloc_fragment([cs, cs], acc)
                dQK = T.alloc_fragment([cs, cs], acc)
                T.gemm(qr_s, ksc_s, QK, transpose_B=True, clear_accum=True)
                T.gemm(dqr_s, ksc_s, dQK, transpose_B=True, clear_accum=True)
                T.gemm(qr_s, dksc_s, dQK, transpose_B=True, clear_accum=False)

                WQK = T.alloc_fragment([cs, cs], acc)
                dWQK = T.alloc_fragment([cs, cs], acc)
                for ii, jj in T.Parallel(cs, cs):
                    w = T.if_then_else(ii >= jj, T.exp(L_s[ii] - L_s[jj]), 0.0)
                    WQK[ii, jj] = w * QK[ii, jj]
                    dWQK[ii, jj] = w * (dL_s[ii] - dL_s[jj]) * QK[ii, jj] + w * dQK[ii, jj]
                T.copy(WQK, WQK_s)
                T.copy(dWQK, dWQK_s)

                out_f = T.alloc_fragment([cs, P], acc)
                dout_f = T.alloc_fragment([cs, P], acc)
                T.gemm(WQK_s, v_s, out_f, clear_accum=True)
                T.gemm(dWQK_s, v_s, dout_f, clear_accum=True)
                T.gemm(WQK_s, dv_s, dout_f, clear_accum=False)

                # inter from the scanned entering state
                qsin = T.alloc_fragment([cs, P], acc)
                dqsin = T.alloc_fragment([cs, P], acc)
                qdsin = T.alloc_fragment([cs, P], acc)
                T.gemm(qr_s, S_s, qsin, clear_accum=True)
                T.gemm(dqr_s, S_s, dqsin, clear_accum=True)
                T.gemm(qr_s, dS_s, qdsin, clear_accum=True)
                for ii, pp in T.Parallel(cs, P):
                    e = T.exp(L_s[ii])
                    out_f[ii, pp] += e * qsin[ii, pp]
                    dout_f[ii, pp] += e * (dL_s[ii] * qsin[ii, pp] + dqsin[ii, pp] + qdsin[ii, pp])

                T.copy(out_f, OUT[i_b, c0:c0 + cs, i_h, :])
                T.copy(dout_f, DOUT[i_b, c0:c0 + cs, i_h, :])

        return kernel

    # Chunked direction-grid gen passes: one direction per CTA, latency hidden by
    # grid size (nc*r*H*B CTAs), chunk states streamed through HBM.

    @tilelang.jit(out_idx=[4], pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    })
    def _gen_passA_primal_kernel(B, S, H, N, P, chunk_size, dtype='bfloat16', threads=128):
        """Pass A (primal, direction-free): per-chunk primal state contribution Sc
        from KR (rotated unscaled K) + SCALE + V. Grid (nc, H, B)."""
        acc = 'float32'
        assert S % chunk_size == 0
        nc = S // chunk_size
        cs = chunk_size

        @T.prim_func
        def kernel(
            KR: T.Tensor([B, S, H, N], dtype),
            V: T.Tensor([B, S, H, P], dtype),
            SCALE: T.Tensor([B, H, S], acc),
            L: T.Tensor([B, H, S], acc),
            SC: T.Tensor([B, H, nc, N, P], acc),
        ):
            with T.Kernel(nc, H, B, threads=threads) as (i_c, i_h, i_b):
                kr_s = T.alloc_shared([cs, N], dtype)
                v_s = T.alloc_shared([cs, P], dtype)
                wksc_s = T.alloc_shared([cs, N], dtype)
                L_s = T.alloc_shared([cs], acc); sc_s = T.alloc_shared([cs], acc)
                T.annotate_layout({
                    kr_s: tilelang.layout.make_swizzled_layout(kr_s),
                    v_s: tilelang.layout.make_swizzled_layout(v_s),
                    wksc_s: tilelang.layout.make_swizzled_layout(wksc_s),
                })
                T.use_swizzle(10, "row")
                c0 = i_c * cs
                T.copy(L[i_b, i_h, c0:c0 + cs], L_s)
                T.copy(SCALE[i_b, i_h, c0:c0 + cs], sc_s)
                T.copy(KR[i_b, c0:c0 + cs, i_h, :], kr_s)
                T.copy(V[i_b, c0:c0 + cs, i_h, :], v_s)
                Sc = T.alloc_fragment([N, P], acc)
                for jj, nn in T.Parallel(cs, N):
                    wl = T.exp(L_s[cs - 1] - L_s[jj])
                    wksc_s[jj, nn] = wl * sc_s[jj] * kr_s[jj, nn]
                T.gemm(wksc_s, v_s, Sc, transpose_A=True, clear_accum=True)
                T.copy(Sc, SC[i_b, i_h, i_c, :, :])

        return kernel

    @tilelang.jit(out_idx=[11], pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    })
    def _gen_passA_tan_kernel(B, S, H, G, N, Da, P, chunk_size, R, dtype='bfloat16', threads=128):
        """Pass A (tangent, direction-grid): per-(chunk, direction) tangent state
        contribution dSc, with on-chip generation of dK_sc in the prologue.
        Grid (nc*R, H, B), direction fastest (L2 tile adjacency)."""
        acc = 'float32'
        assert S % chunk_size == 0
        nc = S // chunk_size
        cs = chunk_size
        HG = H // G

        @T.prim_func
        def kernel(
            KR: T.Tensor([B, S, H, N], dtype),
            V: T.Tensor([B, S, H, P], dtype),
            DKRAW: T.Tensor([R, B, S, G, N], dtype),
            DV: T.Tensor([R, B, S, H, P], dtype),
            DTHETA: T.Tensor([R, B, H, S, Da], dtype),
            COS: T.Tensor([B, H, S, Da], dtype),
            SIN: T.Tensor([B, H, S, Da], dtype),
            SCALE: T.Tensor([B, H, S], acc),
            DSCALE: T.Tensor([R, B, H, S], acc),
            L: T.Tensor([B, H, S], acc),
            DL: T.Tensor([R, B, H, S], acc),
            DSC: T.Tensor([R, B, H, nc, N, P], acc),
        ):
            with T.Kernel(nc * R, H, B, threads=threads) as (i_cl, i_h, i_b):
                i_c = i_cl // R
                i_l = i_cl % R
                kr_s = T.alloc_shared([cs, N], dtype)
                dksc_s = T.alloc_shared([cs, N], dtype)
                v_s = T.alloc_shared([cs, P], dtype); dv_s = T.alloc_shared([cs, P], dtype)
                cos_s = T.alloc_shared([cs, Da], dtype); sin_s = T.alloc_shared([cs, Da], dtype)
                dth_s = T.alloc_shared([cs, Da], dtype)
                scr_s = T.alloc_shared([cs, N], dtype)
                L_s = T.alloc_shared([cs], acc); dL_s = T.alloc_shared([cs], acc)
                sc_s = T.alloc_shared([cs], acc); dsc_s = T.alloc_shared([cs], acc)
                T.annotate_layout({
                    kr_s: tilelang.layout.make_swizzled_layout(kr_s),
                    dksc_s: tilelang.layout.make_swizzled_layout(dksc_s),
                    v_s: tilelang.layout.make_swizzled_layout(v_s),
                    dv_s: tilelang.layout.make_swizzled_layout(dv_s),
                    scr_s: tilelang.layout.make_swizzled_layout(scr_s),
                })
                T.use_swizzle(10, "row")
                c0 = i_c * cs
                T.copy(L[i_b, i_h, c0:c0 + cs], L_s)
                T.copy(DL[i_l, i_b, i_h, c0:c0 + cs], dL_s)
                T.copy(SCALE[i_b, i_h, c0:c0 + cs], sc_s)
                T.copy(DSCALE[i_l, i_b, i_h, c0:c0 + cs], dsc_s)
                T.copy(KR[i_b, c0:c0 + cs, i_h, :], kr_s)
                T.copy(V[i_b, c0:c0 + cs, i_h, :], v_s)
                T.copy(DV[i_l, i_b, c0:c0 + cs, i_h, :], dv_s)
                T.copy(DKRAW[i_l, i_b, c0:c0 + cs, i_h // HG, :], dksc_s)
                T.copy(COS[i_b, i_h, c0:c0 + cs, :], cos_s)
                T.copy(SIN[i_b, i_h, c0:c0 + cs, :], sin_s)
                T.copy(DTHETA[i_l, i_b, i_h, c0:c0 + cs, :], dth_s)

                # tangent-gen prologue: raw dk -> dK_r (rotary JVP, in place) -> dK_sc.
                for ii, pp in T.Parallel(cs, N // 2):
                    c = T.if_then_else(pp < Da, cos_s[ii, pp % Da], 1.0)
                    sn = T.if_then_else(pp < Da, sin_s[ii, pp % Da], 0.0)
                    dt = T.if_then_else(pp < Da, dth_s[ii, pp % Da], 0.0)
                    d0 = dksc_s[ii, 2 * pp]
                    d1 = dksc_s[ii, 2 * pp + 1]
                    k0 = kr_s[ii, 2 * pp]
                    k1 = kr_s[ii, 2 * pp + 1]
                    dksc_s[ii, 2 * pp] = d0 * c - d1 * sn - dt * k1
                    dksc_s[ii, 2 * pp + 1] = d0 * sn + d1 * c + dt * k0
                for ii, nn in T.Parallel(cs, N):
                    dksc_s[ii, nn] = dksc_s[ii, nn] * sc_s[ii] + kr_s[ii, nn] * dsc_s[ii]

                dSc = T.alloc_fragment([N, P], acc)
                for jj, nn in T.Parallel(cs, N):
                    wl = T.exp(L_s[cs - 1] - L_s[jj])
                    scr_s[jj, nn] = wl * (dL_s[cs - 1] - dL_s[jj]) * sc_s[jj] * kr_s[jj, nn]
                T.gemm(scr_s, v_s, dSc, transpose_A=True, clear_accum=True)
                for jj, nn in T.Parallel(cs, N):
                    wl = T.exp(L_s[cs - 1] - L_s[jj])
                    scr_s[jj, nn] = wl * dksc_s[jj, nn]
                T.gemm(scr_s, v_s, dSc, transpose_A=True, clear_accum=False)
                for jj, nn in T.Parallel(cs, N):
                    wl = T.exp(L_s[cs - 1] - L_s[jj])
                    scr_s[jj, nn] = wl * sc_s[jj] * kr_s[jj, nn]
                T.gemm(scr_s, dv_s, dSc, transpose_A=True, clear_accum=False)
                T.copy(dSc, DSC[i_l, i_b, i_h, i_c, :, :])

        return kernel

    @tilelang.jit(out_idx=[15, 16], pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    })
    def _gen_passC_kernel(B, S, H, G, N, Da, P, chunk_size, R, dtype='bfloat16', threads=128):
        """Pass C (full, direction-grid): intra dual quadratic + inter readout from
        scanned entering states, tangent-gen prologue (dQ_r and dK_sc).
        Grid (nc*R, H, B), direction fastest. OUT is written redundantly by every
        direction (identical values); DOUT per direction."""
        acc = 'float32'
        assert S % chunk_size == 0
        nc = S // chunk_size
        cs = chunk_size
        HG = H // G

        @T.prim_func
        def kernel(
            QR: T.Tensor([B, S, H, N], dtype),
            KR: T.Tensor([B, S, H, N], dtype),
            V: T.Tensor([B, S, H, P], dtype),
            DQRAW: T.Tensor([R, B, S, G, N], dtype),
            DKRAW: T.Tensor([R, B, S, G, N], dtype),
            DV: T.Tensor([R, B, S, H, P], dtype),
            DTHETA: T.Tensor([R, B, H, S, Da], dtype),
            COS: T.Tensor([B, H, S, Da], dtype),
            SIN: T.Tensor([B, H, S, Da], dtype),
            SCALE: T.Tensor([B, H, S], acc),
            DSCALE: T.Tensor([R, B, H, S], acc),
            L: T.Tensor([B, H, S], acc),
            DL: T.Tensor([R, B, H, S], acc),
            SIN_S: T.Tensor([B, H, nc, N, P], acc),
            DSIN_S: T.Tensor([R, B, H, nc, N, P], acc),
            OUT: T.Tensor([B, S, H, P], acc),
            DOUT: T.Tensor([R, B, S, H, P], acc),
        ):
            with T.Kernel(nc * R, H, B, threads=threads) as (i_cl, i_h, i_b):
                i_c = i_cl // R
                i_l = i_cl % R
                qr_s = T.alloc_shared([cs, N], dtype); dqr_s = T.alloc_shared([cs, N], dtype)
                kr_s = T.alloc_shared([cs, N], dtype)
                ksc_s = T.alloc_shared([cs, N], dtype); dksc_s = T.alloc_shared([cs, N], dtype)
                v_s = T.alloc_shared([cs, P], dtype); dv_s = T.alloc_shared([cs, P], dtype)
                S_s = T.alloc_shared([N, P], dtype); dS_s = T.alloc_shared([N, P], dtype)
                cos_s = T.alloc_shared([cs, Da], dtype); sin_s = T.alloc_shared([cs, Da], dtype)
                dth_s = T.alloc_shared([cs, Da], dtype)
                L_s = T.alloc_shared([cs], acc); dL_s = T.alloc_shared([cs], acc)
                sc_s = T.alloc_shared([cs], acc); dsc_s = T.alloc_shared([cs], acc)
                WQK_s = T.alloc_shared([cs, cs], dtype); dWQK_s = T.alloc_shared([cs, cs], dtype)
                T.annotate_layout({
                    qr_s: tilelang.layout.make_swizzled_layout(qr_s),
                    dqr_s: tilelang.layout.make_swizzled_layout(dqr_s),
                    ksc_s: tilelang.layout.make_swizzled_layout(ksc_s),
                    dksc_s: tilelang.layout.make_swizzled_layout(dksc_s),
                    v_s: tilelang.layout.make_swizzled_layout(v_s),
                    dv_s: tilelang.layout.make_swizzled_layout(dv_s),
                    S_s: tilelang.layout.make_swizzled_layout(S_s),
                    dS_s: tilelang.layout.make_swizzled_layout(dS_s),
                    WQK_s: tilelang.layout.make_swizzled_layout(WQK_s),
                    dWQK_s: tilelang.layout.make_swizzled_layout(dWQK_s),
                })
                T.use_swizzle(10, "row")
                c0 = i_c * cs
                T.copy(L[i_b, i_h, c0:c0 + cs], L_s)
                T.copy(DL[i_l, i_b, i_h, c0:c0 + cs], dL_s)
                T.copy(SCALE[i_b, i_h, c0:c0 + cs], sc_s)
                T.copy(DSCALE[i_l, i_b, i_h, c0:c0 + cs], dsc_s)
                T.copy(QR[i_b, c0:c0 + cs, i_h, :], qr_s)
                T.copy(KR[i_b, c0:c0 + cs, i_h, :], kr_s)
                T.copy(V[i_b, c0:c0 + cs, i_h, :], v_s)
                T.copy(DV[i_l, i_b, c0:c0 + cs, i_h, :], dv_s)
                T.copy(DQRAW[i_l, i_b, c0:c0 + cs, i_h // HG, :], dqr_s)
                T.copy(DKRAW[i_l, i_b, c0:c0 + cs, i_h // HG, :], dksc_s)
                T.copy(COS[i_b, i_h, c0:c0 + cs, :], cos_s)
                T.copy(SIN[i_b, i_h, c0:c0 + cs, :], sin_s)
                T.copy(DTHETA[i_l, i_b, i_h, c0:c0 + cs, :], dth_s)
                T.copy(SIN_S[i_b, i_h, i_c, :, :], S_s)
                T.copy(DSIN_S[i_l, i_b, i_h, i_c, :, :], dS_s)

                # tangent-gen prologue (verbatim from the serial gen kernel).
                for ii, pp in T.Parallel(cs, N // 2):
                    c = T.if_then_else(pp < Da, cos_s[ii, pp % Da], 1.0)
                    sn = T.if_then_else(pp < Da, sin_s[ii, pp % Da], 0.0)
                    dt = T.if_then_else(pp < Da, dth_s[ii, pp % Da], 0.0)
                    d0 = dqr_s[ii, 2 * pp]
                    d1 = dqr_s[ii, 2 * pp + 1]
                    q0 = qr_s[ii, 2 * pp]
                    q1 = qr_s[ii, 2 * pp + 1]
                    dqr_s[ii, 2 * pp] = d0 * c - d1 * sn - dt * q1
                    dqr_s[ii, 2 * pp + 1] = d0 * sn + d1 * c + dt * q0
                for ii, pp in T.Parallel(cs, N // 2):
                    c = T.if_then_else(pp < Da, cos_s[ii, pp % Da], 1.0)
                    sn = T.if_then_else(pp < Da, sin_s[ii, pp % Da], 0.0)
                    dt = T.if_then_else(pp < Da, dth_s[ii, pp % Da], 0.0)
                    d0 = dksc_s[ii, 2 * pp]
                    d1 = dksc_s[ii, 2 * pp + 1]
                    k0 = kr_s[ii, 2 * pp]
                    k1 = kr_s[ii, 2 * pp + 1]
                    dksc_s[ii, 2 * pp] = d0 * c - d1 * sn - dt * k1
                    dksc_s[ii, 2 * pp + 1] = d0 * sn + d1 * c + dt * k0
                for ii, nn in T.Parallel(cs, N):
                    dksc_s[ii, nn] = dksc_s[ii, nn] * sc_s[ii] + kr_s[ii, nn] * dsc_s[ii]
                    ksc_s[ii, nn] = kr_s[ii, nn] * sc_s[ii]

                # intra dual quadratic + inter readout (verbatim passC body).
                QK = T.alloc_fragment([cs, cs], acc)
                dQK = T.alloc_fragment([cs, cs], acc)
                T.gemm(qr_s, ksc_s, QK, transpose_B=True, clear_accum=True)
                T.gemm(dqr_s, ksc_s, dQK, transpose_B=True, clear_accum=True)
                T.gemm(qr_s, dksc_s, dQK, transpose_B=True, clear_accum=False)
                WQK = T.alloc_fragment([cs, cs], acc)
                dWQK = T.alloc_fragment([cs, cs], acc)
                for ii, jj in T.Parallel(cs, cs):
                    w = T.if_then_else(ii >= jj, T.exp(L_s[ii] - L_s[jj]), 0.0)
                    WQK[ii, jj] = w * QK[ii, jj]
                    dWQK[ii, jj] = w * (dL_s[ii] - dL_s[jj]) * QK[ii, jj] + w * dQK[ii, jj]
                T.copy(WQK, WQK_s)
                T.copy(dWQK, dWQK_s)
                out_f = T.alloc_fragment([cs, P], acc)
                dout_f = T.alloc_fragment([cs, P], acc)
                T.gemm(WQK_s, v_s, out_f, clear_accum=True)
                T.gemm(dWQK_s, v_s, dout_f, clear_accum=True)
                T.gemm(WQK_s, dv_s, dout_f, clear_accum=False)
                qsin = T.alloc_fragment([cs, P], acc)
                dqsin = T.alloc_fragment([cs, P], acc)
                qdsin = T.alloc_fragment([cs, P], acc)
                T.gemm(qr_s, S_s, qsin, clear_accum=True)
                T.gemm(dqr_s, S_s, dqsin, clear_accum=True)
                T.gemm(qr_s, dS_s, qdsin, clear_accum=True)
                for ii, pp in T.Parallel(cs, P):
                    e = T.exp(L_s[ii])
                    out_f[ii, pp] += e * qsin[ii, pp]
                    dout_f[ii, pp] += e * (dL_s[ii] * qsin[ii, pp] + dqsin[ii, pp] + qdsin[ii, pp])
                T.copy(out_f, OUT[i_b, c0:c0 + cs, i_h, :])
                T.copy(dout_f, DOUT[i_l, i_b, c0:c0 + cs, i_h, :])

        return kernel

    @tilelang.jit(out_idx=[20], pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    })
    def _gen_passC_mean_kernel(B, S, H, G, N, Da, P, chunk_size, R, s_true,
                               dtype='bfloat16', threads=128):
        """Pass C (mean epilogue, direction-grid): pass C + finalize (D-skip, QK-dot,
        Z-gate) + in-CTA masked row-sum -> chunk partial PART [R, B, H, nc, P]
        (caller sums nc and divides by s_true)."""
        acc = 'float32'
        assert S % chunk_size == 0
        nc = S // chunk_size
        cs = chunk_size
        HG = H // G

        @T.prim_func
        def kernel(
            QR: T.Tensor([B, S, H, N], dtype),
            KR: T.Tensor([B, S, H, N], dtype),
            V: T.Tensor([B, S, H, P], dtype),
            DQRAW: T.Tensor([R, B, S, G, N], dtype),
            DKRAW: T.Tensor([R, B, S, G, N], dtype),
            DV: T.Tensor([R, B, S, H, P], dtype),
            DTHETA: T.Tensor([R, B, H, S, Da], dtype),
            COS: T.Tensor([B, H, S, Da], dtype),
            SIN: T.Tensor([B, H, S, Da], dtype),
            SCALE: T.Tensor([B, H, S], acc),
            DSCALE: T.Tensor([R, B, H, S], acc),
            L: T.Tensor([B, H, S], acc),
            DL: T.Tensor([R, B, H, S], acc),
            Z: T.Tensor([B, S, H, P], dtype),
            DZ: T.Tensor([R, B, S, H, P], dtype),
            QKDOT: T.Tensor([B, H, S], acc),
            DQKDOT: T.Tensor([R, B, H, S], acc),
            DSKIP: T.Tensor([H], acc),
            SIN_S: T.Tensor([B, H, nc, N, P], acc),
            DSIN_S: T.Tensor([R, B, H, nc, N, P], acc),
            PART: T.Tensor([R, B, H, nc, P], acc),
        ):
            with T.Kernel(nc * R, H, B, threads=threads) as (i_cl, i_h, i_b):
                i_c = i_cl // R
                i_l = i_cl % R
                qr_s = T.alloc_shared([cs, N], dtype); dqr_s = T.alloc_shared([cs, N], dtype)
                kr_s = T.alloc_shared([cs, N], dtype)
                ksc_s = T.alloc_shared([cs, N], dtype); dksc_s = T.alloc_shared([cs, N], dtype)
                v_s = T.alloc_shared([cs, P], dtype); dv_s = T.alloc_shared([cs, P], dtype)
                z_s = T.alloc_shared([cs, P], dtype); dz_s = T.alloc_shared([cs, P], dtype)
                S_s = T.alloc_shared([N, P], dtype); dS_s = T.alloc_shared([N, P], dtype)
                cos_s = T.alloc_shared([cs, Da], dtype); sin_s = T.alloc_shared([cs, Da], dtype)
                dth_s = T.alloc_shared([cs, Da], dtype)
                L_s = T.alloc_shared([cs], acc); dL_s = T.alloc_shared([cs], acc)
                sc_s = T.alloc_shared([cs], acc); dsc_s = T.alloc_shared([cs], acc)
                qk_s = T.alloc_shared([cs], acc); dqk_s = T.alloc_shared([cs], acc)
                WQK_s = T.alloc_shared([cs, cs], dtype); dWQK_s = T.alloc_shared([cs, cs], dtype)
                T.annotate_layout({
                    qr_s: tilelang.layout.make_swizzled_layout(qr_s),
                    dqr_s: tilelang.layout.make_swizzled_layout(dqr_s),
                    ksc_s: tilelang.layout.make_swizzled_layout(ksc_s),
                    dksc_s: tilelang.layout.make_swizzled_layout(dksc_s),
                    v_s: tilelang.layout.make_swizzled_layout(v_s),
                    dv_s: tilelang.layout.make_swizzled_layout(dv_s),
                    S_s: tilelang.layout.make_swizzled_layout(S_s),
                    dS_s: tilelang.layout.make_swizzled_layout(dS_s),
                    WQK_s: tilelang.layout.make_swizzled_layout(WQK_s),
                    dWQK_s: tilelang.layout.make_swizzled_layout(dWQK_s),
                })
                T.use_swizzle(10, "row")
                c0 = i_c * cs
                T.copy(L[i_b, i_h, c0:c0 + cs], L_s)
                T.copy(DL[i_l, i_b, i_h, c0:c0 + cs], dL_s)
                T.copy(SCALE[i_b, i_h, c0:c0 + cs], sc_s)
                T.copy(DSCALE[i_l, i_b, i_h, c0:c0 + cs], dsc_s)
                T.copy(QKDOT[i_b, i_h, c0:c0 + cs], qk_s)
                T.copy(DQKDOT[i_l, i_b, i_h, c0:c0 + cs], dqk_s)
                T.copy(QR[i_b, c0:c0 + cs, i_h, :], qr_s)
                T.copy(KR[i_b, c0:c0 + cs, i_h, :], kr_s)
                T.copy(V[i_b, c0:c0 + cs, i_h, :], v_s)
                T.copy(DV[i_l, i_b, c0:c0 + cs, i_h, :], dv_s)
                T.copy(Z[i_b, c0:c0 + cs, i_h, :], z_s)
                T.copy(DZ[i_l, i_b, c0:c0 + cs, i_h, :], dz_s)
                T.copy(DQRAW[i_l, i_b, c0:c0 + cs, i_h // HG, :], dqr_s)
                T.copy(DKRAW[i_l, i_b, c0:c0 + cs, i_h // HG, :], dksc_s)
                T.copy(COS[i_b, i_h, c0:c0 + cs, :], cos_s)
                T.copy(SIN[i_b, i_h, c0:c0 + cs, :], sin_s)
                T.copy(DTHETA[i_l, i_b, i_h, c0:c0 + cs, :], dth_s)
                T.copy(SIN_S[i_b, i_h, i_c, :, :], S_s)
                T.copy(DSIN_S[i_l, i_b, i_h, i_c, :, :], dS_s)

                for ii, pp in T.Parallel(cs, N // 2):
                    c = T.if_then_else(pp < Da, cos_s[ii, pp % Da], 1.0)
                    sn = T.if_then_else(pp < Da, sin_s[ii, pp % Da], 0.0)
                    dt = T.if_then_else(pp < Da, dth_s[ii, pp % Da], 0.0)
                    d0 = dqr_s[ii, 2 * pp]
                    d1 = dqr_s[ii, 2 * pp + 1]
                    q0 = qr_s[ii, 2 * pp]
                    q1 = qr_s[ii, 2 * pp + 1]
                    dqr_s[ii, 2 * pp] = d0 * c - d1 * sn - dt * q1
                    dqr_s[ii, 2 * pp + 1] = d0 * sn + d1 * c + dt * q0
                for ii, pp in T.Parallel(cs, N // 2):
                    c = T.if_then_else(pp < Da, cos_s[ii, pp % Da], 1.0)
                    sn = T.if_then_else(pp < Da, sin_s[ii, pp % Da], 0.0)
                    dt = T.if_then_else(pp < Da, dth_s[ii, pp % Da], 0.0)
                    d0 = dksc_s[ii, 2 * pp]
                    d1 = dksc_s[ii, 2 * pp + 1]
                    k0 = kr_s[ii, 2 * pp]
                    k1 = kr_s[ii, 2 * pp + 1]
                    dksc_s[ii, 2 * pp] = d0 * c - d1 * sn - dt * k1
                    dksc_s[ii, 2 * pp + 1] = d0 * sn + d1 * c + dt * k0
                for ii, nn in T.Parallel(cs, N):
                    dksc_s[ii, nn] = dksc_s[ii, nn] * sc_s[ii] + kr_s[ii, nn] * dsc_s[ii]
                    ksc_s[ii, nn] = kr_s[ii, nn] * sc_s[ii]

                QK = T.alloc_fragment([cs, cs], acc)
                dQK = T.alloc_fragment([cs, cs], acc)
                T.gemm(qr_s, ksc_s, QK, transpose_B=True, clear_accum=True)
                T.gemm(dqr_s, ksc_s, dQK, transpose_B=True, clear_accum=True)
                T.gemm(qr_s, dksc_s, dQK, transpose_B=True, clear_accum=False)
                WQK = T.alloc_fragment([cs, cs], acc)
                dWQK = T.alloc_fragment([cs, cs], acc)
                for ii, jj in T.Parallel(cs, cs):
                    w = T.if_then_else(ii >= jj, T.exp(L_s[ii] - L_s[jj]), 0.0)
                    WQK[ii, jj] = w * QK[ii, jj]
                    dWQK[ii, jj] = w * (dL_s[ii] - dL_s[jj]) * QK[ii, jj] + w * dQK[ii, jj]
                T.copy(WQK, WQK_s)
                T.copy(dWQK, dWQK_s)
                out_f = T.alloc_fragment([cs, P], acc)
                dout_f = T.alloc_fragment([cs, P], acc)
                T.gemm(WQK_s, v_s, out_f, clear_accum=True)
                T.gemm(dWQK_s, v_s, dout_f, clear_accum=True)
                T.gemm(WQK_s, dv_s, dout_f, clear_accum=False)
                qsin = T.alloc_fragment([cs, P], acc)
                dqsin = T.alloc_fragment([cs, P], acc)
                qdsin = T.alloc_fragment([cs, P], acc)
                T.gemm(qr_s, S_s, qsin, clear_accum=True)
                T.gemm(dqr_s, S_s, dqsin, clear_accum=True)
                T.gemm(qr_s, dS_s, qdsin, clear_accum=True)
                accm = T.alloc_fragment([cs, P], acc)
                for ii, pp in T.Parallel(cs, P):
                    e = T.exp(L_s[ii])
                    out_f[ii, pp] += e * qsin[ii, pp]
                    dout_f[ii, pp] += e * (dL_s[ii] * qsin[ii, pp] + dqsin[ii, pp] + qdsin[ii, pp])
                for ii, pp in T.Parallel(cs, P):
                    o = out_f[ii, pp] + DSKIP[i_h] * v_s[ii, pp] - v_s[ii, pp] * qk_s[ii]
                    do = dout_f[ii, pp] + DSKIP[i_h] * dv_s[ii, pp] \
                        - (dv_s[ii, pp] * qk_s[ii] + v_s[ii, pp] * dqk_s[ii])
                    zf = z_s[ii, pp]
                    sg = 1.0 / (1.0 + T.exp(-zf))
                    gate = zf * sg
                    dgate = sg * (1.0 + zf * (1.0 - sg)) * dz_s[ii, pp]
                    accm[ii, pp] = T.if_then_else(
                        c0 + ii < s_true, do * gate + o * dgate, 0.0)
                part = T.alloc_fragment([P], acc)
                T.reduce_sum(accm, part, dim=0)
                T.copy(part, PART[i_l, i_b, i_h, i_c, :])

        return kernel


if _HAS_TILELANG:
    @tilelang.jit(out_idx=[13, 14], pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    })
    def _chunked_scan_gen_kernel(B, S, H, G, N, Da, P, chunk_size, dtype='bfloat16', threads=256):
        """On-chip tangent generation: consumes raw group-level tangents
        (DQRAW/DKRAW [B,S,G,N]) + small per-head scalar fields, computing the
        rotary-JVP (dQ_r = rot(dq) + dtheta (x) rot90(Q_r)) and the scale-JVP
        (dK_sc = dK_r*scale + K_r*dscale) in the prologue; the [r,B,S,H,N]
        tangent fields never exist in HBM. Dual-scan core byte-identical to
        `_chunked_scan_kernel`."""
        acc = 'float32'
        assert S % chunk_size == 0
        nc = S // chunk_size
        cs = chunk_size
        HG = H // G

        @T.prim_func
        def kernel(
            QR: T.Tensor([B, S, H, N], dtype),       # rotated Q (primal)
            KR: T.Tensor([B, S, H, N], dtype),       # rotated UNSCALED K (primal)
            V: T.Tensor([B, S, H, P], dtype),
            DQRAW: T.Tensor([B, S, G, N], dtype),    # raw group-level tangents
            DKRAW: T.Tensor([B, S, G, N], dtype),
            DV: T.Tensor([B, S, H, P], dtype),
            DTHETA: T.Tensor([B, H, S, Da], dtype),  # tangent rotary phase (cumsum'd)
            COS: T.Tensor([B, H, S, Da], dtype),     # primal rotary cos/sin
            SIN: T.Tensor([B, H, S, Da], dtype),
            SCALE: T.Tensor([B, H, S], acc),
            DSCALE: T.Tensor([B, H, S], acc),
            L: T.Tensor([B, H, S], acc),
            DL: T.Tensor([B, H, S], acc),
            OUT: T.Tensor([B, S, H, P], acc),
            DOUT: T.Tensor([B, S, H, P], acc),
        ):
            with T.Kernel(H, B, threads=threads) as (i_h, i_b):
                S_f = T.alloc_fragment([N, P], acc)
                dS_f = T.alloc_fragment([N, P], acc)
                T.clear(S_f); T.clear(dS_f)

                qr_s = T.alloc_shared([cs, N], dtype); dqr_s = T.alloc_shared([cs, N], dtype)
                kr_s = T.alloc_shared([cs, N], dtype)
                ksc_s = T.alloc_shared([cs, N], dtype); dksc_s = T.alloc_shared([cs, N], dtype)
                v_s = T.alloc_shared([cs, P], dtype); dv_s = T.alloc_shared([cs, P], dtype)
                S_s = T.alloc_shared([N, P], dtype); dS_s = T.alloc_shared([N, P], dtype)
                L_s = T.alloc_shared([cs], acc); dL_s = T.alloc_shared([cs], acc)
                sc_s = T.alloc_shared([cs], acc); dsc_s = T.alloc_shared([cs], acc)
                cos_s = T.alloc_shared([cs, Da], dtype); sin_s = T.alloc_shared([cs, Da], dtype)
                dth_s = T.alloc_shared([cs, Da], dtype)
                WQK_s = T.alloc_shared([cs, cs], dtype); dWQK_s = T.alloc_shared([cs, cs], dtype)
                wksc_s = T.alloc_shared([cs, N], dtype); scr_s = T.alloc_shared([cs, N], dtype)

                T.annotate_layout({
                    qr_s: tilelang.layout.make_swizzled_layout(qr_s),
                    dqr_s: tilelang.layout.make_swizzled_layout(dqr_s),
                    ksc_s: tilelang.layout.make_swizzled_layout(ksc_s),
                    dksc_s: tilelang.layout.make_swizzled_layout(dksc_s),
                    v_s: tilelang.layout.make_swizzled_layout(v_s),
                    dv_s: tilelang.layout.make_swizzled_layout(dv_s),
                    S_s: tilelang.layout.make_swizzled_layout(S_s),
                    dS_s: tilelang.layout.make_swizzled_layout(dS_s),
                    WQK_s: tilelang.layout.make_swizzled_layout(WQK_s),
                    dWQK_s: tilelang.layout.make_swizzled_layout(dWQK_s),
                    wksc_s: tilelang.layout.make_swizzled_layout(wksc_s),
                    scr_s: tilelang.layout.make_swizzled_layout(scr_s),
                })
                T.use_swizzle(10, "row")

                for i in T.Pipelined(0, nc, num_stages=0):
                    c0 = i * cs
                    T.copy(L[i_b, i_h, c0:c0 + cs], L_s)
                    T.copy(DL[i_b, i_h, c0:c0 + cs], dL_s)
                    T.copy(SCALE[i_b, i_h, c0:c0 + cs], sc_s)
                    T.copy(DSCALE[i_b, i_h, c0:c0 + cs], dsc_s)
                    T.copy(QR[i_b, c0:c0 + cs, i_h, :], qr_s)
                    T.copy(KR[i_b, c0:c0 + cs, i_h, :], kr_s)
                    T.copy(V[i_b, c0:c0 + cs, i_h, :], v_s)
                    T.copy(DV[i_b, c0:c0 + cs, i_h, :], dv_s)
                    T.copy(DQRAW[i_b, c0:c0 + cs, i_h // HG, :], dqr_s)
                    T.copy(DKRAW[i_b, c0:c0 + cs, i_h // HG, :], dksc_s)
                    T.copy(COS[i_b, i_h, c0:c0 + cs, :], cos_s)
                    T.copy(SIN[i_b, i_h, c0:c0 + cs, :], sin_s)
                    T.copy(DTHETA[i_b, i_h, c0:c0 + cs, :], dth_s)

                    # on-chip tangent gen, in place per pair: dQ_r = rot(dq) +
                    # dtheta (x) rot90(Q_r); tail pairs use cos=1/sin=0/dtheta=0.
                    for ii, pp in T.Parallel(cs, N // 2):
                        c = T.if_then_else(pp < Da, cos_s[ii, pp % Da], 1.0)
                        sn = T.if_then_else(pp < Da, sin_s[ii, pp % Da], 0.0)
                        dt = T.if_then_else(pp < Da, dth_s[ii, pp % Da], 0.0)
                        d0 = dqr_s[ii, 2 * pp]
                        d1 = dqr_s[ii, 2 * pp + 1]
                        q0 = qr_s[ii, 2 * pp]
                        q1 = qr_s[ii, 2 * pp + 1]
                        dqr_s[ii, 2 * pp] = d0 * c - d1 * sn - dt * q1
                        dqr_s[ii, 2 * pp + 1] = d0 * sn + d1 * c + dt * q0
                    # dksc_s holds raw dk -> dK_r in place, then combine with scale.
                    for ii, pp in T.Parallel(cs, N // 2):
                        c = T.if_then_else(pp < Da, cos_s[ii, pp % Da], 1.0)
                        sn = T.if_then_else(pp < Da, sin_s[ii, pp % Da], 0.0)
                        dt = T.if_then_else(pp < Da, dth_s[ii, pp % Da], 0.0)
                        d0 = dksc_s[ii, 2 * pp]
                        d1 = dksc_s[ii, 2 * pp + 1]
                        k0 = kr_s[ii, 2 * pp]
                        k1 = kr_s[ii, 2 * pp + 1]
                        dksc_s[ii, 2 * pp] = d0 * c - d1 * sn - dt * k1
                        dksc_s[ii, 2 * pp + 1] = d0 * sn + d1 * c + dt * k0
                    for ii, nn in T.Parallel(cs, N):
                        dksc_s[ii, nn] = dksc_s[ii, nn] * sc_s[ii] + kr_s[ii, nn] * dsc_s[ii]
                        ksc_s[ii, nn] = kr_s[ii, nn] * sc_s[ii]

                    # --- (S, dS) scan core (identical to _chunked_scan_kernel) ---
                    QK = T.alloc_fragment([cs, cs], acc)
                    dQK = T.alloc_fragment([cs, cs], acc)
                    T.gemm(qr_s, ksc_s, QK, transpose_B=True, clear_accum=True)
                    T.gemm(dqr_s, ksc_s, dQK, transpose_B=True, clear_accum=True)
                    T.gemm(qr_s, dksc_s, dQK, transpose_B=True, clear_accum=False)

                    WQK = T.alloc_fragment([cs, cs], acc)
                    dWQK = T.alloc_fragment([cs, cs], acc)
                    for ii, jj in T.Parallel(cs, cs):
                        w = T.if_then_else(ii >= jj, T.exp(L_s[ii] - L_s[jj]), 0.0)
                        WQK[ii, jj] = w * QK[ii, jj]
                        dWQK[ii, jj] = w * (dL_s[ii] - dL_s[jj]) * QK[ii, jj] + w * dQK[ii, jj]
                    T.copy(WQK, WQK_s)
                    T.copy(dWQK, dWQK_s)

                    out_f = T.alloc_fragment([cs, P], acc)
                    dout_f = T.alloc_fragment([cs, P], acc)
                    T.gemm(WQK_s, v_s, out_f, clear_accum=True)
                    T.gemm(dWQK_s, v_s, dout_f, clear_accum=True)
                    T.gemm(WQK_s, dv_s, dout_f, clear_accum=False)

                    T.copy(S_f, S_s); T.copy(dS_f, dS_s)
                    qsin = T.alloc_fragment([cs, P], acc)
                    dqsin = T.alloc_fragment([cs, P], acc)
                    qdsin = T.alloc_fragment([cs, P], acc)
                    T.gemm(qr_s, S_s, qsin, clear_accum=True)
                    T.gemm(dqr_s, S_s, dqsin, clear_accum=True)
                    T.gemm(qr_s, dS_s, qdsin, clear_accum=True)
                    for ii, pp in T.Parallel(cs, P):
                        e = T.exp(L_s[ii])
                        out_f[ii, pp] += e * qsin[ii, pp]
                        dout_f[ii, pp] += e * (dL_s[ii] * qsin[ii, pp] + dqsin[ii, pp] + qdsin[ii, pp])

                    T.copy(out_f, OUT[i_b, c0:c0 + cs, i_h, :])
                    T.copy(dout_f, DOUT[i_b, c0:c0 + cs, i_h, :])

                    Sc = T.alloc_fragment([N, P], acc)
                    dSc = T.alloc_fragment([N, P], acc)
                    for jj, nn in T.Parallel(cs, N):
                        wl = T.exp(L_s[cs - 1] - L_s[jj])
                        wksc_s[jj, nn] = wl * ksc_s[jj, nn]
                    T.gemm(wksc_s, v_s, Sc, transpose_A=True, clear_accum=True)
                    for jj, nn in T.Parallel(cs, N):
                        wl = T.exp(L_s[cs - 1] - L_s[jj])
                        scr_s[jj, nn] = wl * (dL_s[cs - 1] - dL_s[jj]) * ksc_s[jj, nn]
                    T.gemm(scr_s, v_s, dSc, transpose_A=True, clear_accum=True)
                    for jj, nn in T.Parallel(cs, N):
                        wl = T.exp(L_s[cs - 1] - L_s[jj])
                        scr_s[jj, nn] = wl * dksc_s[jj, nn]
                    T.gemm(scr_s, v_s, dSc, transpose_A=True, clear_accum=False)
                    T.gemm(wksc_s, dv_s, dSc, transpose_A=True, clear_accum=False)

                    for nn, pp in T.Parallel(N, P):
                        cd = T.exp(L_s[cs - 1])
                        dcd = cd * dL_s[cs - 1]
                        dS_new = cd * dS_f[nn, pp] + dcd * S_f[nn, pp] + dSc[nn, pp]
                        S_new = cd * S_f[nn, pp] + Sc[nn, pp]
                        dS_f[nn, pp] = dS_new
                        S_f[nn, pp] = S_new

        return kernel

    @tilelang.jit(out_idx=[18], pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    })
    def _chunked_scan_gen_mean_kernel(B, S, H, G, N, Da, P, chunk_size, s_true,
                                  dtype='bfloat16', threads=256, min_blocks=1):
        """Mean-epilogue gen kernel: the finalize (D-skip, QK-dot skip, Z-gate)
        folds into the chunk epilogue and the tangent output row-sums on-chip,
        emitting only ACC [B, cs, H, P]; the caller does .sum(cs)/S_true,
        the sequence mean the interface meanpool needs. DOUT/OUT never reach HBM;
        `s_true` masks padded rows. Prologue + core byte-identical to
        `_chunked_scan_gen_kernel`."""
        acc = 'float32'
        assert S % chunk_size == 0
        nc = S // chunk_size
        cs = chunk_size
        HG = H // G

        @T.prim_func
        def kernel(
            QR: T.Tensor([B, S, H, N], dtype),
            KR: T.Tensor([B, S, H, N], dtype),
            V: T.Tensor([B, S, H, P], dtype),
            DQRAW: T.Tensor([B, S, G, N], dtype),
            DKRAW: T.Tensor([B, S, G, N], dtype),
            DV: T.Tensor([B, S, H, P], dtype),
            DTHETA: T.Tensor([B, H, S, Da], dtype),
            COS: T.Tensor([B, H, S, Da], dtype),
            SIN: T.Tensor([B, H, S, Da], dtype),
            SCALE: T.Tensor([B, H, S], acc),
            DSCALE: T.Tensor([B, H, S], acc),
            L: T.Tensor([B, H, S], acc),
            DL: T.Tensor([B, H, S], acc),
            Z: T.Tensor([B, S, H, P], dtype),        # gate primal
            DZ: T.Tensor([B, S, H, P], dtype),       # gate tangent (per direction)
            QKDOT: T.Tensor([B, H, S], acc),         # QK-dot skip (primal)
            DQKDOT: T.Tensor([B, H, S], acc),        # QK-dot skip (tangent)
            DSKIP: T.Tensor([H], acc),               # D skip weights
            ACC: T.Tensor([B, cs, H, P], acc),       # per-slot partial sums
        ):
            with T.Kernel(H, B, threads=threads) as (i_h, i_b):
                if min_blocks > 1:
                    # __launch_bounds__(threads, min_blocks) caps regs/thread so
                    # other kernels can backfill the SMs during the recurrence.
                    T.annotate_min_blocks_per_sm(min_blocks)
                S_f = T.alloc_fragment([N, P], acc)
                dS_f = T.alloc_fragment([N, P], acc)
                accm_f = T.alloc_fragment([cs, P], acc)
                T.clear(S_f); T.clear(dS_f); T.clear(accm_f)

                qr_s = T.alloc_shared([cs, N], dtype); dqr_s = T.alloc_shared([cs, N], dtype)
                kr_s = T.alloc_shared([cs, N], dtype)
                ksc_s = T.alloc_shared([cs, N], dtype); dksc_s = T.alloc_shared([cs, N], dtype)
                v_s = T.alloc_shared([cs, P], dtype); dv_s = T.alloc_shared([cs, P], dtype)
                z_s = T.alloc_shared([cs, P], acc); dz_s = T.alloc_shared([cs, P], acc)
                S_s = T.alloc_shared([N, P], dtype); dS_s = T.alloc_shared([N, P], dtype)
                L_s = T.alloc_shared([cs], acc); dL_s = T.alloc_shared([cs], acc)
                sc_s = T.alloc_shared([cs], acc); dsc_s = T.alloc_shared([cs], acc)
                qk_s = T.alloc_shared([cs], acc); dqk_s = T.alloc_shared([cs], acc)
                cos_s = T.alloc_shared([cs, Da], dtype); sin_s = T.alloc_shared([cs, Da], dtype)
                dth_s = T.alloc_shared([cs, Da], dtype)
                WQK_s = T.alloc_shared([cs, cs], dtype); dWQK_s = T.alloc_shared([cs, cs], dtype)
                wksc_s = T.alloc_shared([cs, N], dtype); scr_s = T.alloc_shared([cs, N], dtype)

                T.annotate_layout({
                    qr_s: tilelang.layout.make_swizzled_layout(qr_s),
                    dqr_s: tilelang.layout.make_swizzled_layout(dqr_s),
                    ksc_s: tilelang.layout.make_swizzled_layout(ksc_s),
                    dksc_s: tilelang.layout.make_swizzled_layout(dksc_s),
                    v_s: tilelang.layout.make_swizzled_layout(v_s),
                    dv_s: tilelang.layout.make_swizzled_layout(dv_s),
                    S_s: tilelang.layout.make_swizzled_layout(S_s),
                    dS_s: tilelang.layout.make_swizzled_layout(dS_s),
                    WQK_s: tilelang.layout.make_swizzled_layout(WQK_s),
                    dWQK_s: tilelang.layout.make_swizzled_layout(dWQK_s),
                    wksc_s: tilelang.layout.make_swizzled_layout(wksc_s),
                    scr_s: tilelang.layout.make_swizzled_layout(scr_s),
                })
                T.use_swizzle(10, "row")

                for i in T.Pipelined(0, nc, num_stages=0):
                    c0 = i * cs
                    T.copy(L[i_b, i_h, c0:c0 + cs], L_s)
                    T.copy(DL[i_b, i_h, c0:c0 + cs], dL_s)
                    T.copy(SCALE[i_b, i_h, c0:c0 + cs], sc_s)
                    T.copy(DSCALE[i_b, i_h, c0:c0 + cs], dsc_s)
                    T.copy(QKDOT[i_b, i_h, c0:c0 + cs], qk_s)
                    T.copy(DQKDOT[i_b, i_h, c0:c0 + cs], dqk_s)
                    T.copy(QR[i_b, c0:c0 + cs, i_h, :], qr_s)
                    T.copy(KR[i_b, c0:c0 + cs, i_h, :], kr_s)
                    T.copy(V[i_b, c0:c0 + cs, i_h, :], v_s)
                    T.copy(DV[i_b, c0:c0 + cs, i_h, :], dv_s)
                    T.copy(Z[i_b, c0:c0 + cs, i_h, :], z_s)
                    T.copy(DZ[i_b, c0:c0 + cs, i_h, :], dz_s)
                    T.copy(DQRAW[i_b, c0:c0 + cs, i_h // HG, :], dqr_s)
                    T.copy(DKRAW[i_b, c0:c0 + cs, i_h // HG, :], dksc_s)
                    T.copy(COS[i_b, i_h, c0:c0 + cs, :], cos_s)
                    T.copy(SIN[i_b, i_h, c0:c0 + cs, :], sin_s)
                    T.copy(DTHETA[i_b, i_h, c0:c0 + cs, :], dth_s)

                    # --- on-chip tangent generation (identical to gen kernel) ---
                    for ii, pp in T.Parallel(cs, N // 2):
                        c = T.if_then_else(pp < Da, cos_s[ii, pp % Da], 1.0)
                        sn = T.if_then_else(pp < Da, sin_s[ii, pp % Da], 0.0)
                        dt = T.if_then_else(pp < Da, dth_s[ii, pp % Da], 0.0)
                        d0 = dqr_s[ii, 2 * pp]
                        d1 = dqr_s[ii, 2 * pp + 1]
                        q0 = qr_s[ii, 2 * pp]
                        q1 = qr_s[ii, 2 * pp + 1]
                        dqr_s[ii, 2 * pp] = d0 * c - d1 * sn - dt * q1
                        dqr_s[ii, 2 * pp + 1] = d0 * sn + d1 * c + dt * q0
                    for ii, pp in T.Parallel(cs, N // 2):
                        c = T.if_then_else(pp < Da, cos_s[ii, pp % Da], 1.0)
                        sn = T.if_then_else(pp < Da, sin_s[ii, pp % Da], 0.0)
                        dt = T.if_then_else(pp < Da, dth_s[ii, pp % Da], 0.0)
                        d0 = dksc_s[ii, 2 * pp]
                        d1 = dksc_s[ii, 2 * pp + 1]
                        k0 = kr_s[ii, 2 * pp]
                        k1 = kr_s[ii, 2 * pp + 1]
                        dksc_s[ii, 2 * pp] = d0 * c - d1 * sn - dt * k1
                        dksc_s[ii, 2 * pp + 1] = d0 * sn + d1 * c + dt * k0
                    for ii, nn in T.Parallel(cs, N):
                        dksc_s[ii, nn] = dksc_s[ii, nn] * sc_s[ii] + kr_s[ii, nn] * dsc_s[ii]
                        ksc_s[ii, nn] = kr_s[ii, nn] * sc_s[ii]

                    # --- (S, dS) scan core (identical to _chunked_scan_kernel) ---
                    QK = T.alloc_fragment([cs, cs], acc)
                    dQK = T.alloc_fragment([cs, cs], acc)
                    T.gemm(qr_s, ksc_s, QK, transpose_B=True, clear_accum=True)
                    T.gemm(dqr_s, ksc_s, dQK, transpose_B=True, clear_accum=True)
                    T.gemm(qr_s, dksc_s, dQK, transpose_B=True, clear_accum=False)

                    WQK = T.alloc_fragment([cs, cs], acc)
                    dWQK = T.alloc_fragment([cs, cs], acc)
                    for ii, jj in T.Parallel(cs, cs):
                        w = T.if_then_else(ii >= jj, T.exp(L_s[ii] - L_s[jj]), 0.0)
                        WQK[ii, jj] = w * QK[ii, jj]
                        dWQK[ii, jj] = w * (dL_s[ii] - dL_s[jj]) * QK[ii, jj] + w * dQK[ii, jj]
                    T.copy(WQK, WQK_s)
                    T.copy(dWQK, dWQK_s)

                    out_f = T.alloc_fragment([cs, P], acc)
                    dout_f = T.alloc_fragment([cs, P], acc)
                    T.gemm(WQK_s, v_s, out_f, clear_accum=True)
                    T.gemm(dWQK_s, v_s, dout_f, clear_accum=True)
                    T.gemm(WQK_s, dv_s, dout_f, clear_accum=False)

                    T.copy(S_f, S_s); T.copy(dS_f, dS_s)
                    qsin = T.alloc_fragment([cs, P], acc)
                    dqsin = T.alloc_fragment([cs, P], acc)
                    qdsin = T.alloc_fragment([cs, P], acc)
                    T.gemm(qr_s, S_s, qsin, clear_accum=True)
                    T.gemm(dqr_s, S_s, dqsin, clear_accum=True)
                    T.gemm(qr_s, dS_s, qdsin, clear_accum=True)
                    for ii, pp in T.Parallel(cs, P):
                        e = T.exp(L_s[ii])
                        out_f[ii, pp] += e * qsin[ii, pp]
                        dout_f[ii, pp] += e * (dL_s[ii] * qsin[ii, pp] + dqsin[ii, pp] + qdsin[ii, pp])

                    # on-chip finalize: out = quad + D*v - v*qkdot; dout = dquad
                    # + D*dv - (dv*qkdot + v*dqkdot); dfin = dout*gate + out*dgate.

                    # gate = z*sig(z), dgate = sig(z)(1 + z(1-sig(z)))*dz; padded
                    # rows (>= s_true) masked out of the accumulator.
                    for ii, pp in T.Parallel(cs, P):
                        o = out_f[ii, pp] + DSKIP[i_h] * v_s[ii, pp] - v_s[ii, pp] * qk_s[ii]
                        do = dout_f[ii, pp] + DSKIP[i_h] * dv_s[ii, pp] \
                            - (dv_s[ii, pp] * qk_s[ii] + v_s[ii, pp] * dqk_s[ii])
                        sg = 1.0 / (1.0 + T.exp(-z_s[ii, pp]))
                        gate = z_s[ii, pp] * sg
                        dgate = sg * (1.0 + z_s[ii, pp] * (1.0 - sg)) * dz_s[ii, pp]
                        accm_f[ii, pp] += T.if_then_else(
                            c0 + ii < s_true, do * gate + o * dgate, 0.0)

                    Sc = T.alloc_fragment([N, P], acc)
                    dSc = T.alloc_fragment([N, P], acc)
                    for jj, nn in T.Parallel(cs, N):
                        wl = T.exp(L_s[cs - 1] - L_s[jj])
                        wksc_s[jj, nn] = wl * ksc_s[jj, nn]
                    T.gemm(wksc_s, v_s, Sc, transpose_A=True, clear_accum=True)
                    for jj, nn in T.Parallel(cs, N):
                        wl = T.exp(L_s[cs - 1] - L_s[jj])
                        scr_s[jj, nn] = wl * (dL_s[cs - 1] - dL_s[jj]) * ksc_s[jj, nn]
                    T.gemm(scr_s, v_s, dSc, transpose_A=True, clear_accum=True)
                    for jj, nn in T.Parallel(cs, N):
                        wl = T.exp(L_s[cs - 1] - L_s[jj])
                        scr_s[jj, nn] = wl * dksc_s[jj, nn]
                    T.gemm(scr_s, v_s, dSc, transpose_A=True, clear_accum=False)
                    T.gemm(wksc_s, dv_s, dSc, transpose_A=True, clear_accum=False)

                    for nn, pp in T.Parallel(N, P):
                        cd = T.exp(L_s[cs - 1])
                        dcd = cd * dL_s[cs - 1]
                        dS_new = cd * dS_f[nn, pp] + dcd * S_f[nn, pp] + dSc[nn, pp]
                        S_new = cd * S_f[nn, pp] + Sc[nn, pp]
                        dS_f[nn, pp] = dS_new
                        S_f[nn, pp] = S_new

                T.copy(accm_f, ACC[i_b, :, i_h, :])

        return kernel

    @tilelang.jit(out_idx=[18], pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    })
    def _chunked_scan_gen_mean_lb_kernel(B, S, H, G, N, Da, P, chunk_size, s_true, rb,
                                     dtype='bfloat16', threads=256):
        """Direction-batched mean-epilogue kernel: `_chunked_scan_gen_mean_kernel`
                with `rb` tangent directions per launch. The primal forward half is
                computed once per chunk and reused across the rb directions; only the
                per-direction tangent GEMMs loop. Emits ACC [rb, B, cs, H, P]."""
        acc = 'float32'
        assert S % chunk_size == 0
        nc = S // chunk_size
        cs = chunk_size
        HG = H // G

        @T.prim_func
        def kernel(
            QR: T.Tensor([B, S, H, N], dtype),
            KR: T.Tensor([B, S, H, N], dtype),
            V: T.Tensor([B, S, H, P], dtype),
            DQRAW: T.Tensor([rb, B, S, G, N], dtype),
            DKRAW: T.Tensor([rb, B, S, G, N], dtype),
            DV: T.Tensor([rb, B, S, H, P], dtype),
            DTHETA: T.Tensor([rb, B, H, S, Da], dtype),
            COS: T.Tensor([B, H, S, Da], dtype),
            SIN: T.Tensor([B, H, S, Da], dtype),
            SCALE: T.Tensor([B, H, S], acc),
            DSCALE: T.Tensor([rb, B, H, S], acc),
            L: T.Tensor([B, H, S], acc),
            DL: T.Tensor([rb, B, H, S], acc),
            Z: T.Tensor([B, S, H, P], dtype),
            DZ: T.Tensor([rb, B, S, H, P], dtype),
            QKDOT: T.Tensor([B, H, S], acc),
            DQKDOT: T.Tensor([rb, B, H, S], acc),
            DSKIP: T.Tensor([H], acc),
            ACC: T.Tensor([rb, B, cs, H, P], acc),
        ):
            with T.Kernel(H, B, threads=threads) as (i_h, i_b):
                # Primal fragments and buffers are shared, one set for all directions.
                S_f = T.alloc_fragment([N, P], acc); T.clear(S_f)
                qr_s = T.alloc_shared([cs, N], dtype)
                kr_s = T.alloc_shared([cs, N], dtype)
                ksc_s = T.alloc_shared([cs, N], dtype)
                v_s = T.alloc_shared([cs, P], dtype)
                z_s = T.alloc_shared([cs, P], acc)
                S_s = T.alloc_shared([N, P], dtype)
                L_s = T.alloc_shared([cs], acc); sc_s = T.alloc_shared([cs], acc)
                qk_s = T.alloc_shared([cs], acc)
                cos_s = T.alloc_shared([cs, Da], dtype); sin_s = T.alloc_shared([cs, Da], dtype)
                WQK_s = T.alloc_shared([cs, cs], dtype)
                wksc_s = T.alloc_shared([cs, N], dtype); scr_s = T.alloc_shared([cs, N], dtype)

                # persistent per-direction state (leading rb dim, carried across
                # chunks); all other per-direction buffers are single reused scratch.
                dS_f = T.alloc_fragment([rb, N, P], acc)
                accm_f = T.alloc_fragment([rb, cs, P], acc)
                T.clear(dS_f); T.clear(accm_f)
                dqr_s = T.alloc_shared([cs, N], dtype)
                dksc_s = T.alloc_shared([cs, N], dtype)
                dv_s = T.alloc_shared([cs, P], dtype); dz_s = T.alloc_shared([cs, P], acc)
                dS_s = T.alloc_shared([N, P], dtype)
                dL_s = T.alloc_shared([cs], acc); dsc_s = T.alloc_shared([cs], acc)
                dqk_s = T.alloc_shared([cs], acc); dth_s = T.alloc_shared([cs, Da], dtype)
                dWQK_s = T.alloc_shared([cs, cs], dtype)

                T.annotate_layout({
                    qr_s: tilelang.layout.make_swizzled_layout(qr_s),
                    ksc_s: tilelang.layout.make_swizzled_layout(ksc_s),
                    v_s: tilelang.layout.make_swizzled_layout(v_s),
                    S_s: tilelang.layout.make_swizzled_layout(S_s),
                    WQK_s: tilelang.layout.make_swizzled_layout(WQK_s),
                    wksc_s: tilelang.layout.make_swizzled_layout(wksc_s),
                    scr_s: tilelang.layout.make_swizzled_layout(scr_s),
                    dqr_s: tilelang.layout.make_swizzled_layout(dqr_s),
                    dksc_s: tilelang.layout.make_swizzled_layout(dksc_s),
                    dv_s: tilelang.layout.make_swizzled_layout(dv_s),
                    dS_s: tilelang.layout.make_swizzled_layout(dS_s),
                    dWQK_s: tilelang.layout.make_swizzled_layout(dWQK_s),
                })
                T.use_swizzle(10, "row")

                for i in T.Pipelined(0, nc, num_stages=0):
                    c0 = i * cs
                    # --- shared primal loads (once per direction-block) ---
                    T.copy(L[i_b, i_h, c0:c0 + cs], L_s)
                    T.copy(SCALE[i_b, i_h, c0:c0 + cs], sc_s)
                    T.copy(QKDOT[i_b, i_h, c0:c0 + cs], qk_s)
                    T.copy(QR[i_b, c0:c0 + cs, i_h, :], qr_s)
                    T.copy(KR[i_b, c0:c0 + cs, i_h, :], kr_s)
                    T.copy(V[i_b, c0:c0 + cs, i_h, :], v_s)
                    T.copy(Z[i_b, c0:c0 + cs, i_h, :], z_s)
                    T.copy(COS[i_b, i_h, c0:c0 + cs, :], cos_s)
                    T.copy(SIN[i_b, i_h, c0:c0 + cs, :], sin_s)
                    for ii, nn in T.Parallel(cs, N):
                        ksc_s[ii, nn] = kr_s[ii, nn] * sc_s[ii]

                    # --- primal QK/WQK/out_f/Sc (shared, computed ONCE) ---
                    QK = T.alloc_fragment([cs, cs], acc)
                    T.gemm(qr_s, ksc_s, QK, transpose_B=True, clear_accum=True)
                    WQK = T.alloc_fragment([cs, cs], acc)
                    for ii, jj in T.Parallel(cs, cs):
                        w = T.if_then_else(ii >= jj, T.exp(L_s[ii] - L_s[jj]), 0.0)
                        WQK[ii, jj] = w * QK[ii, jj]
                    T.copy(WQK, WQK_s)
                    out_f = T.alloc_fragment([cs, P], acc)
                    T.gemm(WQK_s, v_s, out_f, clear_accum=True)
                    T.copy(S_f, S_s)
                    qsin = T.alloc_fragment([cs, P], acc)
                    T.gemm(qr_s, S_s, qsin, clear_accum=True)
                    for ii, pp in T.Parallel(cs, P):
                        out_f[ii, pp] += T.exp(L_s[ii]) * qsin[ii, pp]
                    Sc = T.alloc_fragment([N, P], acc)
                    for jj, nn in T.Parallel(cs, N):
                        wl = T.exp(L_s[cs - 1] - L_s[jj])
                        wksc_s[jj, nn] = wl * ksc_s[jj, nn]
                    T.gemm(wksc_s, v_s, Sc, transpose_A=True, clear_accum=True)

                    # --- per-direction tangent: one device loop, single scratch reused;
                    # every op reuses the primal QK/WQK/out_f/S_s/v_s just built ---
                    dQK = T.alloc_fragment([cs, cs], acc)
                    dWQK = T.alloc_fragment([cs, cs], acc)
                    dout_f = T.alloc_fragment([cs, P], acc)
                    dqsin = T.alloc_fragment([cs, P], acc)
                    qdsin = T.alloc_fragment([cs, P], acc)
                    dSc = T.alloc_fragment([N, P], acc)
                    for b in T.serial(rb):
                        T.copy(DL[b, i_b, i_h, c0:c0 + cs], dL_s)
                        T.copy(DSCALE[b, i_b, i_h, c0:c0 + cs], dsc_s)
                        T.copy(DQKDOT[b, i_b, i_h, c0:c0 + cs], dqk_s)
                        T.copy(DV[b, i_b, c0:c0 + cs, i_h, :], dv_s)
                        T.copy(DZ[b, i_b, c0:c0 + cs, i_h, :], dz_s)
                        T.copy(DQRAW[b, i_b, c0:c0 + cs, i_h // HG, :], dqr_s)
                        T.copy(DKRAW[b, i_b, c0:c0 + cs, i_h // HG, :], dksc_s)
                        T.copy(DTHETA[b, i_b, i_h, c0:c0 + cs, :], dth_s)
                        for ii, pp in T.Parallel(cs, N // 2):
                            c = T.if_then_else(pp < Da, cos_s[ii, pp % Da], 1.0)
                            sn = T.if_then_else(pp < Da, sin_s[ii, pp % Da], 0.0)
                            dt = T.if_then_else(pp < Da, dth_s[ii, pp % Da], 0.0)
                            d0 = dqr_s[ii, 2 * pp]; d1 = dqr_s[ii, 2 * pp + 1]
                            q0 = qr_s[ii, 2 * pp]; q1 = qr_s[ii, 2 * pp + 1]
                            dqr_s[ii, 2 * pp] = d0 * c - d1 * sn - dt * q1
                            dqr_s[ii, 2 * pp + 1] = d0 * sn + d1 * c + dt * q0
                        for ii, pp in T.Parallel(cs, N // 2):
                            c = T.if_then_else(pp < Da, cos_s[ii, pp % Da], 1.0)
                            sn = T.if_then_else(pp < Da, sin_s[ii, pp % Da], 0.0)
                            dt = T.if_then_else(pp < Da, dth_s[ii, pp % Da], 0.0)
                            d0 = dksc_s[ii, 2 * pp]; d1 = dksc_s[ii, 2 * pp + 1]
                            k0 = kr_s[ii, 2 * pp]; k1 = kr_s[ii, 2 * pp + 1]
                            dksc_s[ii, 2 * pp] = d0 * c - d1 * sn - dt * k1
                            dksc_s[ii, 2 * pp + 1] = d0 * sn + d1 * c + dt * k0
                        for ii, nn in T.Parallel(cs, N):
                            dksc_s[ii, nn] = dksc_s[ii, nn] * sc_s[ii] + kr_s[ii, nn] * dsc_s[ii]

                        T.copy(dS_f[b, :, :], dS_s)
                        T.gemm(dqr_s, ksc_s, dQK, transpose_B=True, clear_accum=True)
                        T.gemm(qr_s, dksc_s, dQK, transpose_B=True, clear_accum=False)
                        for ii, jj in T.Parallel(cs, cs):
                            w = T.if_then_else(ii >= jj, T.exp(L_s[ii] - L_s[jj]), 0.0)
                            dWQK[ii, jj] = w * (dL_s[ii] - dL_s[jj]) * QK[ii, jj] + w * dQK[ii, jj]
                        T.copy(dWQK, dWQK_s)
                        T.gemm(dWQK_s, v_s, dout_f, clear_accum=True)
                        T.gemm(WQK_s, dv_s, dout_f, clear_accum=False)
                        T.gemm(dqr_s, S_s, dqsin, clear_accum=True)
                        T.gemm(qr_s, dS_s, qdsin, clear_accum=True)
                        for ii, pp in T.Parallel(cs, P):
                            e = T.exp(L_s[ii])
                            dout_f[ii, pp] += e * (dL_s[ii] * qsin[ii, pp] + dqsin[ii, pp] + qdsin[ii, pp])
                        for ii, pp in T.Parallel(cs, P):
                            o = out_f[ii, pp] + DSKIP[i_h] * v_s[ii, pp] - v_s[ii, pp] * qk_s[ii]
                            do = dout_f[ii, pp] + DSKIP[i_h] * dv_s[ii, pp] \
                                - (dv_s[ii, pp] * qk_s[ii] + v_s[ii, pp] * dqk_s[ii])
                            sg = 1.0 / (1.0 + T.exp(-z_s[ii, pp]))
                            gate = z_s[ii, pp] * sg
                            dgate = sg * (1.0 + z_s[ii, pp] * (1.0 - sg)) * dz_s[ii, pp]
                            accm_f[b, ii, pp] += T.if_then_else(
                                c0 + ii < s_true, do * gate + o * dgate, 0.0)
                        for jj, nn in T.Parallel(cs, N):
                            wl = T.exp(L_s[cs - 1] - L_s[jj])
                            scr_s[jj, nn] = wl * (dL_s[cs - 1] - dL_s[jj]) * ksc_s[jj, nn]
                        T.gemm(scr_s, v_s, dSc, transpose_A=True, clear_accum=True)
                        for jj, nn in T.Parallel(cs, N):
                            wl = T.exp(L_s[cs - 1] - L_s[jj])
                            scr_s[jj, nn] = wl * dksc_s[jj, nn]
                        T.gemm(scr_s, v_s, dSc, transpose_A=True, clear_accum=False)
                        T.gemm(wksc_s, dv_s, dSc, transpose_A=True, clear_accum=False)
                        for nn, pp in T.Parallel(N, P):
                            cd = T.exp(L_s[cs - 1])
                            dcd = cd * dL_s[cs - 1]
                            dS_f[b, nn, pp] = cd * dS_f[b, nn, pp] + dcd * S_f[nn, pp] + dSc[nn, pp]

                    # --- shared primal S_f update (AFTER directions read S_f) ---
                    for nn, pp in T.Parallel(N, P):
                        S_f[nn, pp] = T.exp(L_s[cs - 1]) * S_f[nn, pp] + Sc[nn, pp]

                for b in T.serial(rb):
                    T.copy(accm_f[b, :, :], ACC[b, i_b, :, i_h, :])

        return kernel


def _get_kernel(B, S, H, N, P, chunk_size, dtype='bfloat16'):
    key = (B, S, H, N, P, chunk_size, dtype)
    if key not in _KERNEL_CACHE:
        _KERNEL_CACHE[key] = _chunked_scan_kernel(B, S, H, N, P, chunk_size, dtype=dtype)
    return _KERNEL_CACHE[key]


def _get_lbi_kernel(B, S, H, N, P, chunk_size, rb, dtype='bfloat16'):
    key = ("lbi", B, S, H, N, P, chunk_size, rb, dtype)
    if key not in _KERNEL_CACHE:
        _KERNEL_CACHE[key] = _chunked_scan_lbi_kernel(B, S, H, N, P, chunk_size, rb, dtype=dtype)
    return _KERNEL_CACHE[key]


def _get_gen_kernel(B, S, H, G, N, Da, P, chunk_size, dtype='bfloat16'):
    key = ("gen", B, S, H, G, N, Da, P, chunk_size, dtype)
    if key not in _KERNEL_CACHE:
        _KERNEL_CACHE[key] = _chunked_scan_gen_kernel(B, S, H, G, N, Da, P, chunk_size, dtype=dtype)
    return _KERNEL_CACHE[key]


def _get_gen_mean_kernel(B, S, H, G, N, Da, P, chunk_size, s_true, dtype='bfloat16'):
    key = ("gen_mean", B, S, H, G, N, Da, P, chunk_size, s_true, dtype)
    if key not in _KERNEL_CACHE:
        _KERNEL_CACHE[key] = _chunked_scan_gen_mean_kernel(
            B, S, H, G, N, Da, P, chunk_size, s_true, dtype=dtype)
    return _KERNEL_CACHE[key]


def _get_p1_kernels(B, S, H, G, N, Da, P, chunk_size, R, s_true, dtype='bfloat16'):
    """Chunked direction-grid gen kernels: (passA_primal, passA_tan, passC_full,
    passC_mean)."""
    key = ("p1", B, S, H, G, N, Da, P, chunk_size, R, s_true, dtype)
    if key not in _KERNEL_CACHE:
        _KERNEL_CACHE[key] = (
            _gen_passA_primal_kernel(B, S, H, N, P, chunk_size, dtype=dtype),
            _gen_passA_tan_kernel(B, S, H, G, N, Da, P, chunk_size, R, dtype=dtype),
            _gen_passC_kernel(B, S, H, G, N, Da, P, chunk_size, R, dtype=dtype),
            _gen_passC_mean_kernel(B, S, H, G, N, Da, P, chunk_size, R, s_true, dtype=dtype),
        )
    return _KERNEL_CACHE[key]


def _pass_b_tri_directions(SC, DSC, L_bhs, dL_bhs_lanes, cs):
    """Lane-batched pass B: primal S_in once, per-direction dS_in batched over the
    leading direction dim (same tri-matmul algebra as `_pass_b_tri`; per-direction dcd
    from each direction's dL)."""
    B, H, nc, N, P = SC.shape
    R = DSC.shape[0]
    Lr = L_bhs.reshape(B, H, nc, cs)
    llast = Lr[..., cs - 1]
    LE = torch.cumsum(llast, dim=-1)
    LEc = F.pad(LE[:, :, :-1], (1, 0), value=0.0)
    diff = LEc.unsqueeze(-1) - LE.unsqueeze(-2)
    mask = torch.ones(nc, nc, dtype=torch.bool, device=SC.device).tril(-1)
    tri = torch.where(mask, torch.exp(diff), torch.zeros_like(diff))
    # broadcasted batched GEMMs (cublas) on [.., nc, nc] @ [.., nc, N*P] --
    # no SRC intermediate, no expand copies (einsum was ~5x slower here).
    SCf = SC.reshape(B, H, nc, N * P)
    S_INf = torch.matmul(tri, SCf)                              # [B,H,nc,NP]
    dllast = dL_bhs_lanes.reshape(R, B, H, nc, cs)[..., cs - 1]
    dcd = torch.exp(llast).unsqueeze(0) * dllast                # [R, B, H, nc]
    # fold dcd into a per-direction tri (tiny) : DS_IN = tri@DSC + (tri*dcd_j)@S_IN.
    tri_d = tri.unsqueeze(0) * dcd.unsqueeze(-2)                # [R,B,H,c,j]
    DS_INf = (torch.matmul(tri, DSC.reshape(R, B, H, nc, N * P))
              + torch.matmul(tri_d, S_INf.unsqueeze(0)))
    return (S_INf.reshape(B, H, nc, N, P).contiguous(),
            DS_INf.reshape(R, B, H, nc, N, P).contiguous())


def _get_cp_kernels(B, S, H, N, P, chunk_size, dtype='bfloat16'):
    key = ("cp", B, S, H, N, P, chunk_size, dtype)
    if key not in _KERNEL_CACHE:
        _KERNEL_CACHE[key] = (
            _chunked_scan_passA_kernel(B, S, H, N, P, chunk_size, dtype=dtype),
            _chunked_scan_passC_kernel(B, S, H, N, P, chunk_size, dtype=dtype),
        )
    return _KERNEL_CACHE[key]


def _tangent_fields_kernel_layout(pf, directions, Sp: int, cs: int, bf16_out: bool):
    """Tangent-field math and kernel-layout emission (permute + pad + cast +
    per-chunk dL cumsum) in one compiled unit, so the layout pass fuses into
    the producers."""
    tf = _prepare_tangent_fields_batched(pf, directions)
    r = tf["r"]
    S, B, H = pf["S"], pf["B"], pf["H"]
    out_dt = torch.bfloat16 if bf16_out else torch.float32

    def lay(f):
        X = f.shape[-1]
        t = f.reshape(r, S, B, H, X).permute(0, 2, 1, 3, 4)
        if Sp > S:
            t = F.pad(t, (0, 0, 0, 0, 0, Sp - S))
        return t.to(out_dt).contiguous()

    dQr = lay(tf["dQ_r"])
    dKr = lay(tf["dK_sc"])
    dVr = lay(tf["dVf"])
    dADTf_p = F.pad(tf["dADTf"], (0, 0, 0, Sp - S)) if Sp > S else tf["dADTf"]
    dLr = torch.cumsum(dADTf_p.reshape(r, Sp // cs, cs, B * H), dim=2)
    dLr = dLr.reshape(r, Sp, B, H).permute(0, 2, 3, 1).contiguous()
    return tf, dQr, dKr, dVr, dLr


def _pass_b_tri(SC, DSC, L_bhs, dL_bhs, cs):
    """Pass B: the inter-chunk (S_in, dS_in) scan as TWO batched tri-matmuls
    with no python loop. Both are first-order linear recurrences, so
    S_in = tri @ SC and dS_in = tri @ (DSC + dchunk_decay * S_in), with
    tri[c, j] = exp(LE[c-1] - LE[j]) masked j <= c-1 (LE = cumulative chunk
    log-decay). Decays are <= 0 so exp <= 1 (no overflow)."""
    B, H, nc, N, P = SC.shape
    Lr = L_bhs.reshape(B, H, nc, cs)
    llast = Lr[..., cs - 1]                                     # per-chunk log-decay
    dllast = dL_bhs.reshape(B, H, nc, cs)[..., cs - 1]
    LE = torch.cumsum(llast, dim=-1)                            # [B, H, nc]
    # tri[c, j] = exp(LE[c-1] - LE[j]), j <= c-1
    LEc = F.pad(LE[:, :, :-1], (1, 0), value=0.0)               # LE[c-1] with LE[-1]=0
    diff = LEc.unsqueeze(-1) - LE.unsqueeze(-2)                 # [B, H, c, j]
    mask = torch.ones(nc, nc, dtype=torch.bool, device=SC.device).tril(-1)
    tri = torch.where(mask, torch.exp(diff), torch.zeros_like(diff))
    S_IN = torch.einsum("bhcj,bhjnp->bhcnp", tri, SC)
    dcd = torch.exp(llast) * dllast                             # d(chunk_decay)
    SRC = DSC + dcd.unsqueeze(-1).unsqueeze(-1) * S_IN
    DS_IN = torch.einsum("bhcj,bhjnp->bhcnp", tri, SRC)
    return S_IN, DS_IN


def _to_bshx(fld: torch.Tensor, S: int, B: int, H: int) -> torch.Tensor:
    return fld.reshape(S, B, H, fld.shape[-1]).permute(1, 0, 2, 3).contiguous()


_TORCH_DT = {'bfloat16': torch.bfloat16, 'float32': torch.float32}


def mamba3_siso_chunked_scan_tilelang(
    primals: tuple[torch.Tensor, ...],
    tangents: tuple[torch.Tensor, ...],
    *,
    chunk_size: int = 64,
    compute_dtype: torch.dtype = torch.float32,
    operand_dtype: str = 'bfloat16',
) -> tuple[torch.Tensor, torch.Tensor]:
    """(Out, dOut) through the fused tilelang chunked scan, a drop-in for
        `mamba3_siso_chunked_scan_ref_blocked`: field preparation and finalize stay
        in torch, the kernel runs the (S, dS) scan on tensor cores. `operand_dtype`
        is 'bfloat16' (default) or 'float32' (TF32 tensor cores); accumulators are
        fp32 either way."""
    if not _HAS_TILELANG:
        raise RuntimeError("tilelang is not available in this environment.")
    if operand_dtype not in _TORCH_DT:
        raise ValueError("operand_dtype must be 'bfloat16' or 'float32'")
    f = _prepare_dual_fields(primals, tangents, compute_dtype)
    S, B, H, N, P = f["S"], f["B"], f["H"], f["Dqk"], f["Dv"]
    cs = int(chunk_size)
    Sp = ((S + cs - 1) // cs) * cs
    op_dt = _TORCH_DT[operand_dtype]

    def cumsum_chunk(x):  # [S, BH] -> [Sp, BH] per-chunk inclusive cumsum
        xp = F.pad(x, (0, 0, 0, Sp - S)) if Sp > S else x
        return torch.cumsum(xp.reshape(Sp // cs, cs, -1), dim=1).reshape(Sp, -1)

    L = cumsum_chunk(f["ADTf"])
    dL = cumsum_chunk(f["dADTf"])

    def pad_bshx(fld):
        t = _to_bshx(fld, S, B, H)
        if Sp > S:
            t = F.pad(t, (0, 0, 0, 0, 0, Sp - S))
        # operand tiles feed the tensor cores; L/dL stay fp32 (exp precision).
        return t.to(op_dt).contiguous()

    QR = pad_bshx(f["Q_r"]); DQR = pad_bshx(f["dQ_r"])
    KSC = pad_bshx(f["K_sc"]); DKSC = pad_bshx(f["dK_sc"])
    Vv = pad_bshx(f["Vf"]); DVv = pad_bshx(f["dVf"])
    L_bhs = L.reshape(Sp, B, H).permute(1, 2, 0).contiguous()
    dL_bhs = dL.reshape(Sp, B, H).permute(1, 2, 0).contiguous()

    kern = _get_kernel(B, Sp, H, N, P, cs, dtype=operand_dtype)
    OUT, DOUT = kern(QR, DQR, KSC, DKSC, Vv, DVv, L_bhs, dL_bhs)
    out_quad = OUT[:, :S].permute(1, 0, 2, 3).reshape(S, B * H, P)
    dout_quad = DOUT[:, :S].permute(1, 0, 2, 3).reshape(S, B * H, P)
    return _finalize(out_quad, dout_quad, f)


def _pad_bshx_op(fld, S, B, H, Sp, op_dt):
    t = _to_bshx(fld, S, B, H)
    if Sp > S:
        t = F.pad(t, (0, 0, 0, 0, 0, Sp - S))
    return t.to(op_dt).contiguous()


def _cumsum_chunk(x, S, cs, Sp):  # [S, BH] -> [Sp, BH] per-chunk inclusive cumsum
    xp = F.pad(x, (0, 0, 0, Sp - S)) if Sp > S else x
    return torch.cumsum(xp.reshape(Sp // cs, cs, -1), dim=1).reshape(Sp, -1)


def mamba3_siso_chunked_scan_tilelang_rwide(
    primals: tuple[torch.Tensor, ...],
    tangent_directions: "list[tuple[torch.Tensor, ...]]",
    *,
    chunk_size: int = 64,
    compute_dtype: torch.dtype = torch.float32,
    operand_dtype: str = 'bfloat16',
    chunk_parallel: "bool | None" = None,
    return_quad: bool = False,
) -> "tuple[torch.Tensor, list[torch.Tensor]]":
    """A_k over r tangent directions with the forward half computed once: primal
    field prep and forward kernel inputs (QR/KSC/V/L) are built once and
    reused across directions; only the tangent inputs (DQR/DKSC/DV/dL) are
    per-direction. Returns (out, [dout_0, ..., dout_{r-1}]) with out the
    direction-independent forward scan output."""
    if not _HAS_TILELANG:
        raise RuntimeError("tilelang is not available in this environment.")
    if operand_dtype not in _TORCH_DT:
        raise ValueError("operand_dtype must be 'bfloat16' or 'float32'")
    op_dt = _TORCH_DT[operand_dtype]

    pf = _maybe_compile(_prepare_primal_fields, "primal")(primals, compute_dtype)
    S, B, H, N, P = pf["S"], pf["B"], pf["H"], pf["Dqk"], pf["Dv"]
    cs = int(chunk_size)
    Sp = ((S + cs - 1) // cs) * cs
    r = len(tangent_directions)

    # Forward-half kernel inputs, built once and shared across directions.
    QR = _pad_bshx_op(pf["Q_r"], S, B, H, Sp, op_dt)
    KSC = _pad_bshx_op(pf["K_sc"], S, B, H, Sp, op_dt)
    Vv = _pad_bshx_op(pf["Vf"], S, B, H, Sp, op_dt)
    L = _cumsum_chunk(pf["ADTf"], S, cs, Sp)
    L_bhs = L.reshape(Sp, B, H).permute(1, 2, 0).contiguous()
    kern = _get_kernel(B, Sp, H, N, P, cs, dtype=operand_dtype)

    # tangent-field math and kernel-layout emission (permute/pad/cast/
    # dL-cumsum) fused into one compiled pass.
    tf, dQ_r_r, dK_sc_r, dV_r, dL_r = _maybe_compile(
        _tangent_fields_kernel_layout, "tf_kernel_layout"
    )(pf, tangent_directions, Sp, cs, op_dt == torch.bfloat16)

    if chunk_parallel is None:
        # regime split: the three-pass form wins only when the serial (H, B)
        # grid underfills the GPU; at saturation the serial form is faster.
        chunk_parallel = (B * H) <= 16

    out_quad = None
    dout_quads = []
    if chunk_parallel:
        # Three-pass chunk-parallel form (grid H*B*nc per pass) for the
        # underfilled-serial-grid regime (small B*H).
        kA, kC = _get_cp_kernels(B, Sp, H, N, P, cs, dtype=operand_dtype)
        for l in range(r):
            SCc, DSCc = kA(KSC, dK_sc_r[l], Vv, dV_r[l], L_bhs, dL_r[l])
            S_IN, DS_IN = _pass_b_tri(SCc, DSCc, L_bhs, dL_r[l], cs)
            OUT, DOUT = kC(QR, dQ_r_r[l], KSC, dK_sc_r[l], Vv, dV_r[l], L_bhs, dL_r[l], S_IN, DS_IN)
            if out_quad is None:
                out_quad = OUT[:, :S].permute(1, 0, 2, 3).reshape(S, B * H, P)
            dout_quads.append(DOUT[:, :S].permute(1, 0, 2, 3).reshape(S, B * H, P))
    else:
        for l in range(r):
            OUT, DOUT = kern(QR, dQ_r_r[l], KSC, dK_sc_r[l], Vv, dV_r[l], L_bhs, dL_r[l])
            if out_quad is None:
                out_quad = OUT[:, :S].permute(1, 0, 2, 3).reshape(S, B * H, P)
            dout_quads.append(DOUT[:, :S].permute(1, 0, 2, 3).reshape(S, B * H, P))
    dout_quad = torch.stack(dout_quads, dim=0)                  # [r, S, BH, P]
    if return_quad:
        # CB epilogue fusion: caller fuses finalize + its own postprocess.
        return out_quad, dout_quad, tf
    out, douts = _maybe_compile(_finalize_batched, "finalize_batched")(out_quad, dout_quad, tf)
    return out, [douts[l] for l in range(r)]


def _to_bshx_lane(fld, S, B, H, Sp, op_dt, r):
    """[r, S, BH, X] -> [r, B, Sp, H, X] padded/cast (per-direction kernel inputs)."""
    X = fld.shape[-1]
    t = fld.reshape(r, S, B, H, X).permute(0, 2, 1, 3, 4)       # [r, B, S, H, X]
    if Sp > S:
        t = F.pad(t, (0, 0, 0, 0, 0, Sp - S))
    return t.to(op_dt).contiguous()


def mamba3_siso_chunked_scan_tilelang_rwide_encoded(
    primals: tuple[torch.Tensor, ...],
    tangent_directions: "list[tuple[torch.Tensor, ...]]",
    reduce_weight: torch.Tensor,
    *,
    chunk_size: int = 64,
    compute_dtype: torch.dtype = torch.float32,
    operand_dtype: str = 'bfloat16',
) -> torch.Tensor:
    """Encode-fused variant: the interface meanpool and linear encode fold into
        the kernel's output boundary, so the [r, B, S, H, Dv] region-output tangent
        never materializes and each direction reduces to [B, r_out] at once.
        `reduce_weight` [r_out, H*Dv] is the encode operator applied after the
        sequence mean (`Wenc @ out_proj_weight`); returns the A_k contribution rows
        `delta` [r_lanes, B, r_out], to which the caller adds the skip and update
        scale. Directions are processed one at a time, so peak memory is O(B*L*D)
        rather than O(r*B*L*D)."""
    if not _HAS_TILELANG:
        raise RuntimeError("tilelang is not available in this environment.")
    if operand_dtype not in _TORCH_DT:
        raise ValueError("operand_dtype must be 'bfloat16' or 'float32'")
    op_dt = _TORCH_DT[operand_dtype]
    pf = _prepare_primal_fields(primals, compute_dtype)
    S, B, H, N, P = pf["S"], pf["B"], pf["H"], pf["Dqk"], pf["Dv"]
    cs = int(chunk_size)
    Sp = ((S + cs - 1) // cs) * cs
    r = len(tangent_directions)
    Dv = P
    Wr = reduce_weight.to(torch.float32)                        # [r_out, H*Dv]

    # Forward-half kernel inputs, built once and shared across directions.
    QR = _pad_bshx_op(pf["Q_r"], S, B, H, Sp, op_dt)
    KSC = _pad_bshx_op(pf["K_sc"], S, B, H, Sp, op_dt)
    Vv = _pad_bshx_op(pf["Vf"], S, B, H, Sp, op_dt)
    L_bhs = _cumsum_chunk(pf["ADTf"], S, cs, Sp).reshape(Sp, B, H).permute(1, 2, 0).contiguous()
    kern = _get_kernel(B, Sp, H, N, P, cs, dtype=operand_dtype)

    out_quad = None
    deltas = []
    for l in range(r):
        # decode side: this direction's tangent fields only (transient).
        f = _prepare_tangent_fields(pf, tangent_directions[l])
        DQR = _pad_bshx_op(f["dQ_r"], S, B, H, Sp, op_dt)
        DKSC = _pad_bshx_op(f["dK_sc"], S, B, H, Sp, op_dt)
        DVv = _pad_bshx_op(f["dVf"], S, B, H, Sp, op_dt)
        dL = _cumsum_chunk(f["dADTf"], S, cs, Sp).reshape(Sp, B, H).permute(1, 2, 0).contiguous()
        OUT, DOUT = kern(QR, DQR, KSC, DKSC, Vv, DVv, L_bhs, dL)
        if out_quad is None:
            out_quad = OUT[:, :S].permute(1, 0, 2, 3).reshape(S, B * H, P)
        dout_quad = DOUT[:, :S].permute(1, 0, 2, 3).reshape(S, B * H, P)
        _, do = _finalize(out_quad, dout_quad, f)               # [B, S, H, Dv]
        # Encode side: meanpool over S plus fused encode. The [B,S,H,Dv]
        # tangent reduces to [B, r_out] here and is dropped, never stacked.
        do_mean = do.reshape(B, S, H * Dv).float().mean(dim=1)  # [B, H*Dv]
        deltas.append(do_mean @ Wr.t())                         # [B, r_out]
    return torch.stack(deltas, dim=0)                           # [r_lanes, B, r_out]
