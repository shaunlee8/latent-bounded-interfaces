"""Forward-mode region JVP for the transformer backend.

`transformer_region_output_jvp` is the torch.func reference: a differentiable
region replay with chunked math attention and the P tangent directions under
vmap over torch.func.jvp. `transformer_region_output_jvp_kernel` runs the
directions direction-major [r, B, L, D] through the CUDA and Triton kernels
with the tangent GEMMs flattened to single [r*B*L, d] calls; for an
L-constant tangent basis the first block's norm and in_proj tangent collapses
exactly to s*U_j - a_j*qkv."""
from __future__ import annotations

import torch

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


def _linear_tangent(lin, t):
    r, B, L, d = t.shape
    w = lin.weight
    if w.dtype != t.dtype:
        # harmonize to the weight dtype (mixed-precision training contexts)
        return (t.reshape(-1, d).to(w.dtype) @ w.t()).reshape(r, B, L, -1).to(t.dtype)
    return (t.reshape(-1, d) @ w.t()).reshape(r, B, L, -1)


def _swiglu_tangent(gate, up, activation, dup, dgate, dtype):
    """Tangent of up * silu(gate) at the operating point (activation is the
    cached silu(gate))."""
    g = gate.float()
    sg = torch.sigmoid(g)
    dsilu = (sg * (1 + g * (1 - sg))).to(dtype)
    return dup * activation.to(dtype) + up.to(dtype).unsqueeze(0) * dsilu * dgate


def _norm_collapse_terms(norm, lin, x, t):
    """Scalars for the in-kernel first-block tangent synthesis (L-constant
    basis): dx = s*U_j - a_j*x with s = rsqrt(ms) [B,L], a = s^2*mean(x.*t_c)
    [r,B,L], U = (w.*t_c)@W^T [r,B,3D]. Requires a bias-free in_proj."""
    xf = x.float()
    s = torch.rsqrt(xf.pow(2).mean(dim=-1) + norm.eps)
    w = norm.weight.to(x.dtype)
    tc = t[:, :, 0, :]
    # matmul operands harmonized to the projection weight's dtype (training
    # contexts can hold in_proj at a different precision than the tangents)
    wd = lin.weight.dtype
    U = ((w * tc).to(wd).reshape(-1, tc.shape[-1]) @ lin.weight.t()).reshape(
        t.shape[0], t.shape[1], -1).to(t.dtype)
    a = s.pow(2).unsqueeze(0) * (torch.einsum("bld,rbd->rbl", xf, tc.float())
                                 / x.shape[-1])
    return U, s, a


_CUDA_ATTN_JVP = None  # wrapper once resolved; False when unavailable


def _resolve_cuda_attn_jvp():
    """The CUDA flash-JVP wrapper, or None on non-Hopper devices or failed
    builds (the Triton kernel runs instead)."""
    global _CUDA_ATTN_JVP
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


def _linear_tangent_add(lin, t, res):
    """res + t @ W^T in one GEMM epilogue (cuBLAS beta=1): the residual
    add of the tangent chain folded into the projection, one pass over
    the tangent stack fewer than the unfused (GEMM, then add) pair."""
    r, B, L, d = t.shape
    w = lin.weight
    if w.dtype != t.dtype or res.dtype != t.dtype:
        return _linear_tangent(lin, t) + res
    return torch.addmm(res.reshape(-1, res.shape[-1]), t.reshape(-1, d), w.t()).reshape(res.shape)


def _cache_fed_tangent_chain_fused(backend, layer_caches, start, end, t,
                                   collapse_first, compute_dtype, pooled,
                                   cuda_attn_fn=None, token_start=0):
    """The cache-fed tangent chain: every primal operating point comes from
    the region caches, both residual adds per block fold into the out_proj /
    fc2 GEMM epilogues (`_linear_tangent_add`), and the SwiGLU tangent runs as
    one Triton pass. With `token_start` > 0 the chain runs on the token suffix
    only, exact when the tangent is zero before it."""
    from backbones.transformer.ops.triton.region_jvp import (
        flash_attention_jvp, fused_qkv_prep, rmsnorm_jvp, swiglu_jvp)

    s = int(token_start)
    dres = t
    for i, layer_index in enumerate(range(start, end)):
        blk = backend.backbone.blocks[layer_index]
        attn = blk.attn
        hd = attn.head_dim
        lc = layer_caches[i]
        L = lc.attention.norm_input.shape[-2]
        x_attn = lc.attention.norm_input[:, s:].to(compute_dtype)
        if i == 0 and collapse_first and s == 0 and attn.in_proj.bias is None:
            qkv_pre = lc.attention.qkv_preconv
            if qkv_pre is None:
                qkv_pre = attn.in_proj(blk.norm(x_attn))
            synth = _norm_collapse_terms(blk.norm, attn.in_proj, x_attn, dres)
            dqkv = None
            shape_src = qkv_pre.contiguous()
        else:
            _, dnh = rmsnorm_jvp(x_attn.contiguous(), dres, blk.norm.weight, blk.norm.eps,
                                 emit_y=False)
            dqkv = _linear_tangent(attn.in_proj, dnh)
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
        rep = attn.n_heads // attn.n_kv_heads
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
            dk = dk.repeat_interleave(rep, dim=2)
            dv = dv.repeat_interleave(rep, dim=2)
        scale = softmax_scale(attn)
        if cuda_attn_fn is not None and s == 0 and hd == 64 and L % 64 == 0:
            _, do = cuda_attn_fn(q, k, v, dq, dk, dv, scale)
        else:
            _, do = flash_attention_jvp(q, k, v, dq, dk, dv, scale, query_start=s)
            do = do.transpose(-3, -2).flatten(-2)
        dres = _linear_tangent_add(attn.out_proj, do, dres)
        x_mlp = lc.mlp.norm_input[:, s:].to(compute_dtype)
        _, dnh = rmsnorm_jvp(x_mlp.contiguous(), dres, blk.norm2.weight, blk.norm2.eps,
                             emit_y=False)
        dh1 = _linear_tangent(blk.mlp.fc1, dnh)
        dmid = swiglu_jvp(lc.mlp.gate[:, s:], lc.mlp.up[:, s:], lc.mlp.activation[:, s:],
                          dh1 if dh1.is_contiguous() else dh1.contiguous())
        dres = _linear_tangent_add(blk.mlp.fc2, dmid, dres)
    dout = dres
    if pooled:
        dout = dout.sum(dim=2, keepdim=True) / float(L) if s else dout.mean(dim=2, keepdim=True)
    return dout


def _kernel_path_supported(backend, start, end):
    dim = backend.backbone.blocks[start].norm.weight.shape[0]
    if dim & (dim - 1):
        return False
    for layer_index in range(start, end):
        attn = backend.backbone.blocks[layer_index].attn
        hd = attn.head_dim
        if (attn.d_conv <= 0 or attn.n_heads % attn.n_kv_heads
                or attn.rope_interleaved or hd & (hd - 1) or hd < 16):
            return False
    return True


def transformer_region_output_jvp_kernel(backend, *, cache,
                                         region_input_tangent_basis,
                                         compute_dtype=torch.bfloat16,
                                         pooled=False,
                                         tangent_token_start=0):
    """Kernel tangent map, direction-major [r, B, L, D] internally, reading every
    primal operating point from the region forward's layer caches; falls back to
    the reference path for module configurations outside the kernel contract.
    With `tangent_token_start` > 0 the chain computes only the suffix the caller
    certifies as nonzero and zero-fills the prefix."""
    start, end = cache.layer_range
    if region_input_tangent_basis.dtype is not torch.bfloat16:
        # The fused attention JVP computes in bf16 and its fp32 inputs are not
        # promoted consistently (the result is a non-finite Jacobian rather
        # than an error); fail like the Mamba-3 backend does.
        raise ValueError(
            "transformer forward-mode construction requires dtype=bfloat16; "
            "use interface_jacobian_mode='graph' for float32")
    if not _kernel_path_supported(backend, start, end):
        if not getattr(backend, "_fwdmode_fallback_warned", False):
            backend._fwdmode_fallback_warned = True
            import warnings

            warnings.warn(
                "transformer forward-mode kernel path unsupported for this "
                "module configuration (needs power-of-two dim/head_dim >= 16, "
                "n_heads divisible by n_kv_heads, conv > 0, non-interleaved rope); "
                "falling back to the reference JVP at ~5-10x the cost.",
                stacklevel=2,
            )
        out = transformer_region_output_jvp(
            backend, cache=cache,
            region_input_tangent_basis=region_input_tangent_basis,
            compute_dtype=compute_dtype)
        return out.mean(dim=2, keepdim=True) if pooled else out

    # First-block collapse is exact only for an L-constant tangent basis;
    # token-varying bases take the general norm-JVP path. Detect before contiguous().
    l_const = (region_input_tangent_basis.shape[2] == 1
               or region_input_tangent_basis.stride(2) == 0)
    collapse_first = l_const

    layer_caches = cache.layer_caches
    if len(layer_caches) != end - start:
        raise ValueError("transformer forward-mode kernel path needs the region forward's layer caches")

    s = int(tangent_token_start)
    full_len = region_input_tangent_basis.shape[2]
    if s > 0 and (l_const or s >= full_len):
        s = 0
    basis = region_input_tangent_basis[:, :, s:] if s else region_input_tangent_basis
    t = basis.movedim(1, 0).to(compute_dtype).contiguous()

    cuda_attn_fn = _resolve_cuda_attn_jvp()
    dout = _cache_fed_tangent_chain_fused(backend, layer_caches, start, end, t,
                                          collapse_first, compute_dtype, pooled,
                                          cuda_attn_fn, s)
    dout = dout.movedim(0, 1)
    if s and not pooled:
        full = torch.zeros(
            dout.shape[0], dout.shape[1], full_len, dout.shape[3],
            device=dout.device, dtype=dout.dtype)
        full[:, :, s:] = dout
        return full
    return dout
