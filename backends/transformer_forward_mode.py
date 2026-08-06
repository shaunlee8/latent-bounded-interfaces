"""Forward-mode region JVP for the transformer backend.

`transformer_region_output_jvp` is the torch.func reference: a differentiable
region replay (chunked math attention; flash has no forward-AD rule) with the
P tangent lanes riding a vmap over torch.func.jvp.
`transformer_region_output_jvp_kernel` runs the lanes lane-major [r, B, L, D]
through the CUDA/triton kernels, with tangent GEMMs flattened to single
[r*B*L, d] calls. For an L-constant tangent basis the first block's norm +
in_proj tangent collapses exactly to s*U_j - a_j*qkv (`LBI_FWDMODE_BCAST=0`
disables the collapse for general bases).
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from backbones.transformer.rope import _rope_tables, apply_rope
from backends.transformer import softmax_scale


def _chunked_causal_attention_math(q, k, v, scale, qchunk=256):
    """Causal attention in chunked math form; differentiable and forward-AD
    safe (the flash kernels have no jvp rule). q/k/v [B, H, L, hd]."""
    Lq = q.shape[-2]
    outs = []
    for s0 in range(0, Lq, qchunk):
        e = min(s0 + qchunk, Lq)
        S = (q[..., s0:e, :].float() @ k[..., :e, :].float().transpose(-2, -1)) * scale
        mask = (torch.arange(e, device=q.device)[None, :]
                > torch.arange(s0, e, device=q.device)[:, None])
        S = S.masked_fill(mask, float("-inf"))
        P = torch.softmax(S, dim=-1).to(q.dtype)
        outs.append(P @ v[..., :e, :])
    return torch.cat(outs, dim=-2)


def _attention_forward_math(attn, x, qchunk=256):
    """The attention module's forward with math attention substituted."""
    bsz, seqlen, _ = x.shape
    qkv = attn.in_proj(x)
    if attn.d_conv > 0:
        qkv = attn.conv1d(qkv.transpose(1, 2))[..., :seqlen].transpose(1, 2)
    q_dim = attn.n_heads * attn.head_dim
    kv_dim = attn.n_kv_heads * attn.head_dim
    q, k, v = qkv.split((q_dim, kv_dim, kv_dim), dim=-1)
    q = q.view(bsz, seqlen, attn.n_heads, attn.head_dim).transpose(1, 2)
    k = k.view(bsz, seqlen, attn.n_kv_heads, attn.head_dim).transpose(1, 2)
    v = v.view(bsz, seqlen, attn.n_kv_heads, attn.head_dim).transpose(1, 2)
    q = apply_rope(q, base=attn.rope_base, interleaved=attn.rope_interleaved)
    k = apply_rope(k, base=attn.rope_base, interleaved=attn.rope_interleaved)
    k = attn._expand_kv(k)
    v = attn._expand_kv(v)
    heads = _chunked_causal_attention_math(q, k, v, softmax_scale(attn), qchunk)
    out = heads.transpose(1, 2).contiguous().view(bsz, seqlen, attn.out_dim)
    return attn.out_proj(out)


def transformer_region_forward_math(backend, region_input, start, end, qchunk=256):
    """Differentiable replay of the region map on the math attention path."""
    h, res = region_input, None
    for layer_index in range(start, end):
        blk = backend.backbone.blocks[layer_index]
        norm_out, res = blk._prenorm(h, res, blk.norm)
        h = _attention_forward_math(blk.attn, norm_out, qchunk)
        norm_out, res = blk._prenorm(h, res, blk.norm2)
        h = blk.mlp(norm_out)
    return h + res if res is not None else h


def transformer_region_output_jvp(backend, *, cache, region_input_tangent_basis,
                                  compute_dtype=None):
    """Reference tangent map via vmap over torch.func.jvp of the region
    replay. Basis [B, P, L, D] -> [B, P, L, D], returned in the basis dtype.
    The norm and softmax internals run fp32 regardless of `compute_dtype`;
    the module weights stay in their own dtype."""
    del compute_dtype
    start, end = cache.layer_range
    u0 = cache.region_input.detach()
    tangents = region_input_tangent_basis.movedim(1, 0).to(u0.dtype)

    def region(u):
        return transformer_region_forward_math(backend, u, start, end)

    out = torch.vmap(lambda dv: torch.func.jvp(region, (u0,), (dv,))[1])(tangents)
    return out.movedim(0, 1).contiguous().to(region_input_tangent_basis.dtype)


def _linear_jvp(lin, x, t):
    """Primal + tangent through a linear layer; the tangent GEMM is flattened
    to one [r*B*L, d] call (bias drops from the tangent)."""
    r, B, L, d = t.shape
    dy = (t.reshape(-1, d) @ lin.weight.t()).reshape(r, B, L, -1)
    return lin(x), dy


def _linear_tangent(lin, t):
    r, B, L, d = t.shape
    return (t.reshape(-1, d) @ lin.weight.t()).reshape(r, B, L, -1)


def _swiglu_tangent(gate, up, activation, dup, dgate, dtype):
    """Tangent of up * silu(gate) at the operating point (activation is the
    cached silu(gate))."""
    g = gate.float()
    sg = torch.sigmoid(g)
    dsilu = (sg * (1 + g * (1 - sg))).to(dtype)
    return dup * activation.to(dtype) + up.to(dtype).unsqueeze(0) * dsilu * dgate


def _mlp_jvp(mlp, x, t):
    h, dh = _linear_jvp(mlp.fc1, x, t)
    up, gate = h.chunk(2, dim=-1)
    dup, dgate = dh.chunk(2, dim=-1)
    silu = (gate.float() * torch.sigmoid(gate.float())).to(x.dtype)
    dmid = _swiglu_tangent(gate, up, silu, dup, dgate, x.dtype)
    return _linear_jvp(mlp.fc2, up * silu, dmid)


def _norm_collapse_terms(norm, lin, x, t):
    """Scalars for the in-kernel first-block tangent synthesis (L-constant
    basis): dx = s*U_j - a_j*x with s = rsqrt(ms) [B,L], a = s^2*mean(x.*t_c)
    [r,B,L], U = (w.*t_c)@W^T [r,B,3D]. Requires a bias-free in_proj."""
    xf = x.float()
    s = torch.rsqrt(xf.pow(2).mean(dim=-1) + norm.eps)
    w = norm.weight.to(x.dtype)
    tc = t[:, :, 0, :]
    U = ((w * tc).reshape(-1, tc.shape[-1]) @ lin.weight.t()).reshape(
        t.shape[0], t.shape[1], -1)
    a = s.pow(2).unsqueeze(0) * (torch.einsum("bld,rbd->rbl", xf, tc.float())
                                 / x.shape[-1])
    return U, s, a


# Compiled cache-fed chains, keyed by (region range, tangent shape, pooled);
# LBI_FWDMODE_COMPILE opts in (first call pays the compile).
_COMPILED_CHAINS: dict = {}

_CUDA_ATTN_JVP = None  # wrapper once resolved; False when unavailable


def _resolve_cuda_attn_jvp():
    """The CUDA flash-JVP wrapper, or None on non-Hopper devices, failed
    builds, or LBI_FWDMODE_CUDA_ATTN=0 (the triton kernel runs instead)."""
    global _CUDA_ATTN_JVP
    if os.environ.get("LBI_FWDMODE_CUDA_ATTN", "1") == "0":
        return None
    if _CUDA_ATTN_JVP is None:
        try:
            from cuda.transformer import (
                _module, flash_attention_jvp_cuda, flash_jvp_available)
            if flash_jvp_available():
                _module()  # build here so failures downgrade, not raise
                _CUDA_ATTN_JVP = flash_attention_jvp_cuda
            else:
                _CUDA_ATTN_JVP = False
        except Exception:
            _CUDA_ATTN_JVP = False
    return _CUDA_ATTN_JVP or None


def _cache_fed_tangent_chain(backend, layer_caches, start, end, t,
                             collapse_first, compute_dtype, pooled,
                             cuda_attn_fn=None, token_start=0):
    """The cache-fed tangent thread as one compilable function: every primal
    operating point comes from the caches; the custom kernels trace opaque.

    `token_start` > 0 runs the whole tangent thread on the token suffix only
    (`t` arrives suffix-shaped [r, B, L - s, D]). Exact when the true tangent
    is zero before `token_start`: every per-token op slices, the conv's
    zero left-pad IS the zero tangent at the boundary, the rope tables shift
    to absolute positions, and the flash JVP runs suffix queries against the
    full-length cached keys/values."""
    from backbones.transformer.ops.triton.region_jvp import (
        flash_attention_jvp, fused_qkv_prep, rmsnorm_jvp)

    s = int(token_start)
    dres = None
    dh = t
    for i, layer_index in enumerate(range(start, end)):
        blk = backend.backbone.blocks[layer_index]
        attn = blk.attn
        hd = attn.head_dim
        lc = layer_caches[i]
        L = lc.attention.norm_input.shape[-2]
        dres = dh + dres if dres is not None else dh
        x_attn = lc.attention.norm_input[:, s:].to(compute_dtype)
        if i == 0 and collapse_first and s == 0 and attn.in_proj.bias is None:
            qkv_pre = lc.attention.qkv_preconv
            if qkv_pre is None:
                qkv_pre = attn.in_proj(blk.norm(x_attn))
            synth = _norm_collapse_terms(blk.norm, attn.in_proj, x_attn, dres)
            dqkv = None
            shape_src = qkv_pre.contiguous()
        else:
            _, dnh = rmsnorm_jvp(x_attn.contiguous(), dres.contiguous(),
                                 blk.norm.weight, blk.norm.eps, emit_y=False)
            dqkv = _linear_tangent(attn.in_proj, dnh).contiguous()
            synth = None
            shape_src = dqkv[0]
        cos, sin = _rope_tables(L, hd // 2, attn.rope_base, x_attn.device)
        conv_w = attn.conv1d.weight.view(-1, attn.d_conv)
        _, _, _, dq, dk, dv = fused_qkv_prep(
            shape_src, dqkv, synth, conv_w, attn.conv1d.bias,
            cos[s:].contiguous(), sin[s:].contiguous(),
            attn.n_heads, attn.n_kv_heads, hd, emit_primal=False)
        q = lc.attention.q_rope.contiguous()
        k = lc.attention.k_rope.contiguous()
        v = lc.attention.v.contiguous()
        scale = softmax_scale(attn)
        if cuda_attn_fn is not None and s == 0 and hd == 64 and L % 64 == 0:
            # Flat-layout epilogue: do arrives [r, B, L, H*hd], out_proj-ready.
            _, do = cuda_attn_fn(q, k, v, dq, dk, dv, scale)
        else:
            _, do = flash_attention_jvp(q, k, v, dq, dk, dv, scale, query_start=s)
            do = do.transpose(-3, -2).flatten(-2)
        dh = _linear_tangent(attn.out_proj, do)
        dres = dh + dres
        x_mlp = lc.mlp.norm_input[:, s:].to(compute_dtype)
        _, dnh = rmsnorm_jvp(x_mlp.contiguous(), dres.contiguous(),
                             blk.norm2.weight, blk.norm2.eps, emit_y=False)
        dh1 = _linear_tangent(blk.mlp.fc1, dnh)
        dup, dgate = dh1.chunk(2, dim=-1)
        dmid = _swiglu_tangent(lc.mlp.gate[:, s:], lc.mlp.up[:, s:],
                               lc.mlp.activation[:, s:], dup, dgate, compute_dtype)
        dh = _linear_tangent(blk.mlp.fc2, dmid)
    dout = dh + dres
    if pooled:
        # The suffix sum over the true (zero-prefix) tangent divided by the
        # FULL length is the exact full-sequence mean.
        dout = dout.sum(dim=2, keepdim=True) / float(L) if s else dout.mean(dim=2, keepdim=True)
    return dout


def _kernel_path_supported(backend, start, end):
    dim = backend.backbone.blocks[start].norm.weight.shape[0]
    if dim & (dim - 1):
        return False
    for layer_index in range(start, end):
        attn = backend.backbone.blocks[layer_index].attn
        hd = attn.head_dim
        if (attn.d_conv <= 0 or attn.n_heads != attn.n_kv_heads
                or attn.rope_interleaved or hd & (hd - 1) or hd < 16):
            return False
    return True


def transformer_region_output_jvp_kernel(backend, *, cache,
                                         region_input_tangent_basis,
                                         compute_dtype=torch.bfloat16,
                                         pooled=False,
                                         tangent_token_start=0):
    """Kernel tangent map, lane-major [r, B, L, D] internally. Falls back to
    the reference path for module configurations outside the kernel contract
    (grouped KV heads, interleaved rope, no conv, non-power-of-two widths).

    `tangent_token_start` > 0: the caller certifies the tangent basis is zero
    before that token; the cache-fed chain then computes only the suffix and
    the result is zero-filled back to full length. Exact; ignored (full
    compute, still exact) on the fallback and recompute paths."""
    start, end = cache.layer_range
    if not _kernel_path_supported(backend, start, end):
        out = transformer_region_output_jvp(
            backend, cache=cache,
            region_input_tangent_basis=region_input_tangent_basis,
            compute_dtype=compute_dtype)
        return out.mean(dim=2, keepdim=True) if pooled else out

    from backbones.transformer.ops.triton.region_jvp import (
        flash_attention_jvp, fused_qkv_prep, rmsnorm_jvp)

    x = cache.region_input
    # The first-block collapse is exact only for an L-constant tangent basis
    # (broadcast decodes); token-varying bases (tokenwise decodes) must take
    # the general norm-JVP path. Detect before contiguous() expands stride-0.
    l_const = (region_input_tangent_basis.shape[2] == 1
               or region_input_tangent_basis.stride(2) == 0)
    collapse_first = (os.environ.get("LBI_FWDMODE_BCAST", "1") != "0") and l_const

    # Cache-fed mode reads every primal operating point from the region
    # forward's caches; the recompute path below rebuilds the primal chain.
    layer_caches = getattr(cache, "layer_caches", None)
    cache_fed = (layer_caches is not None
                 and len(layer_caches) == end - start
                 and os.environ.get("LBI_FWDMODE_CACHEPRE", "1") != "0")

    s = int(tangent_token_start) if cache_fed else 0
    full_len = region_input_tangent_basis.shape[2]
    if s > 0 and (l_const or s >= full_len):
        s = 0
    basis = region_input_tangent_basis[:, :, s:] if s else region_input_tangent_basis
    t = basis.movedim(1, 0).to(compute_dtype).contiguous()

    if cache_fed:
        cuda_attn_fn = _resolve_cuda_attn_jvp()
        chain = _cache_fed_tangent_chain
        if os.environ.get("LBI_FWDMODE_COMPILE", "0") != "0":
            key = (start, end, tuple(t.shape), bool(pooled), s,
                   cuda_attn_fn is not None)
            chain = _COMPILED_CHAINS.get(key)
            if chain is None:
                chain = torch.compile(_cache_fed_tangent_chain, dynamic=False)
                _COMPILED_CHAINS[key] = chain
        dout = chain(backend, layer_caches, start, end, t, collapse_first,
                     compute_dtype, pooled, cuda_attn_fn, s)
        dout = dout.movedim(0, 1)
        if s and not pooled:
            full = torch.zeros(
                dout.shape[0], dout.shape[1], full_len, dout.shape[3],
                device=dout.device, dtype=dout.dtype)
            full[:, :, s:] = dout
            return full
        return dout

    h, dh = x.to(compute_dtype), t
    res = dres = None
    for i, layer_index in enumerate(range(start, end)):
        blk = backend.backbone.blocks[layer_index]
        attn = blk.attn
        hd = attn.head_dim
        L = h.shape[-2]
        res = h + res if res is not None else h
        dres = dh + dres if dres is not None else dh
        if i == 0 and collapse_first and attn.in_proj.bias is None:
            qkv = attn.in_proj(blk.norm(res))
            synth = _norm_collapse_terms(blk.norm, attn.in_proj, res, dres)
            dqkv = None
        else:
            nh, dnh = rmsnorm_jvp(res.contiguous(), dres.contiguous(),
                                  blk.norm.weight, blk.norm.eps)
            qkv, dqkv = _linear_jvp(attn.in_proj, nh, dnh)
            synth = None
        cos, sin = _rope_tables(L, hd // 2, attn.rope_base, qkv.device)
        conv_w = attn.conv1d.weight.view(-1, attn.d_conv)
        q, k, v, dq, dk, dv = fused_qkv_prep(
            qkv.contiguous(), None if dqkv is None else dqkv.contiguous(),
            synth, conv_w, attn.conv1d.bias, cos, sin,
            attn.n_heads, attn.n_kv_heads, hd)
        scale = softmax_scale(attn)
        o, do = flash_attention_jvp(q, k, v, dq, dk, dv, scale)
        o = o.transpose(-3, -2).flatten(-2)
        do = do.transpose(-3, -2).flatten(-2)
        h, dh = _linear_jvp(attn.out_proj, o, do)
        res, dres = h + res, dh + dres
        nh, dnh = rmsnorm_jvp(res.contiguous(), dres.contiguous(),
                              blk.norm2.weight, blk.norm2.eps)
        h, dh = _mlp_jvp(blk.mlp, nh, dnh)
    dout = dh + dres
    if pooled:
        dout = dout.mean(dim=2, keepdim=True)
    return dout.movedim(0, 1)
