"""Fused block-sparse attention via Triton.

Forward + backward kernel for block-sparse attention with:
  - Online-softmax in SRAM (no gathered K/V materialization)
  - Optional causal masking (own block gets triu)
  - Gate bias (additive log-sigmoid from selector)
  - Custom autograd Function for training

Layouts (matching ssa_model.py):
  Q, K, V: (B, H, L, dh)
  sel:     (B, nblk, kk) int32 — sel[:, :, 0] must be the own block.
  gate:    (B, nblk, kk) float32 — optional additive bias per selected block
  O:       (B, H, L, dh)
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _bsattn_fwd(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    sel_ptr, gate_ptr,
    USE_GATE: tl.constexpr, CAUSAL: tl.constexpr,
    q_stride_b, q_stride_h, q_stride_l,
    k_stride_b, k_stride_h, k_stride_l,
    sel_stride_b, sel_stride_n,
    nblk, kk,
    BS: tl.constexpr, DH: tl.constexpr, KK: tl.constexpr, SCALE: tl.constexpr,
):
    pid_qb = tl.program_id(0).to(tl.int64)
    pid_h = tl.program_id(1).to(tl.int64)
    pid_b = tl.program_id(2).to(tl.int64)

    q_row = tl.arange(0, BS)
    q_col = tl.arange(0, DH)
    q_base = pid_b * q_stride_b + pid_h * q_stride_h + pid_qb * BS * q_stride_l
    q_off = q_base + q_row[:, None] * q_stride_l + q_col[None, :]
    q = tl.load(Q_ptr + q_off).to(tl.float32)

    m_i = tl.full((BS,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BS,), dtype=tl.float32)
    acc = tl.zeros((BS, DH), dtype=tl.float32)

    k_row = tl.arange(0, BS)
    k_col = tl.arange(0, DH)
    if CAUSAL:
        causal_valid = q_row[:, None] >= k_row[None, :]

    for j in tl.static_range(KK):
        kblk = tl.load(sel_ptr + pid_b * sel_stride_b + pid_qb * sel_stride_n + j).to(tl.int64)
        k_base = pid_b * k_stride_b + pid_h * k_stride_h + kblk * BS * k_stride_l
        k_off = k_base + k_row[:, None] * k_stride_l + k_col[None, :]
        k = tl.load(K_ptr + k_off).to(tl.float32)
        v = tl.load(V_ptr + k_off).to(tl.float32)
        scores = tl.dot(q, tl.trans(k)) * SCALE
        if CAUSAL:
            if j == 0:
                scores = tl.where(causal_valid, scores, float("-inf"))
            elif kblk >= pid_qb:
                scores = tl.full((BS, BS), float("-inf"), dtype=tl.float32)
        if USE_GATE:
            g = tl.load(gate_ptr + pid_b * sel_stride_b + pid_qb * sel_stride_n + j)
            scores = scores + g
        m_j = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_j)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    o = acc / l_i[:, None]
    tl.store(O_ptr + q_off, o.to(Q_ptr.dtype.element_ty))


@triton.jit
def _bsattn_bwd(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr,
    dQ_ptr, dK_ptr, dV_ptr,
    sel_ptr, gate_ptr,
    USE_GATE: tl.constexpr, CAUSAL: tl.constexpr,
    q_stride_b, q_stride_h, q_stride_l,
    k_stride_b, k_stride_h, k_stride_l,
    sel_stride_b, sel_stride_n,
    nblk, kk,
    BS: tl.constexpr, DH: tl.constexpr, KK: tl.constexpr, SCALE: tl.constexpr,
):
    pid_qb = tl.program_id(0).to(tl.int64)
    pid_h = tl.program_id(1).to(tl.int64)
    pid_b = tl.program_id(2).to(tl.int64)

    q_row = tl.arange(0, BS)
    q_col = tl.arange(0, DH)
    k_row = tl.arange(0, BS)
    k_col = tl.arange(0, DH)

    q_base = pid_b * q_stride_b + pid_h * q_stride_h + pid_qb * BS * q_stride_l
    q_off = q_base + q_row[:, None] * q_stride_l + q_col[None, :]
    q = tl.load(Q_ptr + q_off).to(tl.float32)
    dO = tl.load(dO_ptr + q_off).to(tl.float32)

    if CAUSAL:
        causal_valid = q_row[:, None] >= k_row[None, :]

    # ---- pass 1: recompute forward (m_i, l_i) ----
    m_i = tl.full((BS,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BS,), dtype=tl.float32)

    for j in tl.static_range(KK):
        kblk = tl.load(sel_ptr + pid_b * sel_stride_b + pid_qb * sel_stride_n + j).to(tl.int64)
        k_base = pid_b * k_stride_b + pid_h * k_stride_h + kblk * BS * k_stride_l
        k_off = k_base + k_row[:, None] * k_stride_l + k_col[None, :]
        k = tl.load(K_ptr + k_off).to(tl.float32)
        scores = tl.dot(q, tl.trans(k)) * SCALE
        if CAUSAL:
            if j == 0:
                scores = tl.where(causal_valid, scores, float("-inf"))
            elif kblk >= pid_qb:
                scores = tl.full((BS, BS), float("-inf"), dtype=tl.float32)
        if USE_GATE:
            g = tl.load(gate_ptr + pid_b * sel_stride_b + pid_qb * sel_stride_n + j)
            scores = scores + g
        m_j = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_j)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    # ---- pass 2: compute D = sum_j rowsum(P_j * dP_j) ----
    D_acc = tl.zeros((BS,), dtype=tl.float32)
    for j in tl.static_range(KK):
        kblk = tl.load(sel_ptr + pid_b * sel_stride_b + pid_qb * sel_stride_n + j).to(tl.int64)
        k_base = pid_b * k_stride_b + pid_h * k_stride_h + kblk * BS * k_stride_l
        k_off = k_base + k_row[:, None] * k_stride_l + k_col[None, :]
        k = tl.load(K_ptr + k_off).to(tl.float32)
        v = tl.load(V_ptr + k_off).to(tl.float32)
        scores = tl.dot(q, tl.trans(k)) * SCALE
        if CAUSAL:
            if j == 0:
                scores = tl.where(causal_valid, scores, float("-inf"))
            elif kblk >= pid_qb:
                scores = tl.full((BS, BS), float("-inf"), dtype=tl.float32)
        if USE_GATE:
            g = tl.load(gate_ptr + pid_b * sel_stride_b + pid_qb * sel_stride_n + j)
            scores = scores + g
        p = tl.exp(scores - m_i[:, None]) / l_i[:, None]
        dP = tl.dot(dO, tl.trans(v))
        D_acc += tl.sum(p * dP, axis=1)

    # ---- pass 3: compute dQ, dK, dV ----
    dQ_acc = tl.zeros((BS, DH), dtype=tl.float32)
    for j in tl.static_range(KK):
        kblk = tl.load(sel_ptr + pid_b * sel_stride_b + pid_qb * sel_stride_n + j).to(tl.int64)
        k_base = pid_b * k_stride_b + pid_h * k_stride_h + kblk * BS * k_stride_l
        k_off = k_base + k_row[:, None] * k_stride_l + k_col[None, :]
        k = tl.load(K_ptr + k_off).to(tl.float32)
        v = tl.load(V_ptr + k_off).to(tl.float32)
        scores = tl.dot(q, tl.trans(k)) * SCALE
        if CAUSAL:
            if j == 0:
                scores = tl.where(causal_valid, scores, float("-inf"))
            elif kblk >= pid_qb:
                scores = tl.full((BS, BS), float("-inf"), dtype=tl.float32)
        if USE_GATE:
            g = tl.load(gate_ptr + pid_b * sel_stride_b + pid_qb * sel_stride_n + j)
            scores = scores + g
        p = tl.exp(scores - m_i[:, None]) / l_i[:, None]

        dV = tl.dot(tl.trans(p).to(dO.dtype), dO)
        dP = tl.dot(dO, tl.trans(v))
        ds = p * (dP - D_acc[:, None]) * SCALE

        dQ_acc += tl.dot(ds.to(k.dtype), k)
        dK = tl.dot(tl.trans(ds).to(q.dtype), q)

        tl.atomic_add(dV_ptr + k_off, dV.to(dV_ptr.dtype.element_ty))
        tl.atomic_add(dK_ptr + k_off, dK.to(dK_ptr.dtype.element_ty))

    tl.store(dQ_ptr + q_off, dQ_acc.to(dQ_ptr.dtype.element_ty))


def bsattn_forward(q, k, v, sel, gate_bias=None, causal=False):
    B, H, L, dh = q.shape
    nblk, kk = sel.shape[1], sel.shape[2]
    Bs = L // nblk
    assert L % nblk == 0
    scale = float(dh ** -0.5)

    q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
    o = torch.empty_like(q)
    sel_c = sel.contiguous().to(torch.int32)
    gate_c = gate_bias.contiguous() if gate_bias is not None else None

    grid = (nblk, H, B)
    _bsattn_fwd[grid](
        q, k, v, o, sel_c, gate_c,
        USE_GATE=(gate_c is not None), CAUSAL=causal,
        q_stride_b=q.stride(0), q_stride_h=q.stride(1), q_stride_l=q.stride(2),
        k_stride_b=k.stride(0), k_stride_h=k.stride(1), k_stride_l=k.stride(2),
        sel_stride_b=sel_c.stride(0), sel_stride_n=sel_c.stride(1),
        nblk=nblk, kk=kk,
        BS=Bs, DH=dh, KK=kk, SCALE=scale,
        num_warps=4 if dh <= 64 else 8, num_stages=2,
    )
    return o


def bsattn_backward(q, k, v, sel, o, do, gate_bias=None, causal=False):
    B, H, L, dh = q.shape
    nblk, kk = sel.shape[1], sel.shape[2]
    Bs = L // nblk
    scale = float(dh ** -0.5)

    q = q.contiguous(); k = k.contiguous(); v = v.contiguous(); do = do.contiguous()
    sel_c = sel.contiguous().to(torch.int32)
    gate_c = gate_bias.contiguous() if gate_bias is not None else None
    dQ = torch.empty_like(q)
    dK = torch.zeros_like(k)
    dV = torch.zeros_like(v)

    grid = (nblk, H, B)
    _bsattn_bwd[grid](
        q, k, v, o, do, dQ, dK, dV, sel_c, gate_c,
        USE_GATE=(gate_c is not None), CAUSAL=causal,
        q_stride_b=q.stride(0), q_stride_h=q.stride(1), q_stride_l=q.stride(2),
        k_stride_b=k.stride(0), k_stride_h=k.stride(1), k_stride_l=k.stride(2),
        sel_stride_b=sel_c.stride(0), sel_stride_n=sel_c.stride(1),
        nblk=nblk, kk=kk,
        BS=Bs, DH=dh, KK=kk, SCALE=scale,
        num_warps=4 if dh <= 64 else 8, num_stages=2,
    )
    return dQ, dK, dV


class BSAttnFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sel, gate_bias, causal):
        # Save contiguous copies so backward recomputes from the same data
        qc, kc, vc = q.contiguous(), k.contiguous(), v.contiguous()
        o = bsattn_forward(qc, kc, vc, sel, gate_bias, causal=causal)
        ctx.save_for_backward(qc, kc, vc, o, sel.contiguous().to(torch.int32),
                              gate_bias.contiguous() if gate_bias is not None else torch.empty(0, device=q.device))
        ctx.has_gate = gate_bias is not None
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, sel, gate_bias = ctx.saved_tensors
        gate = gate_bias if ctx.has_gate else None
        dQ, dK, dV = bsattn_backward(q, k, v, sel, o, do, gate, causal=ctx.causal)
        return dQ, dK, dV, None, None, None


def bsattn(q, k, v, sel, gate_bias=None, causal=False):
    """Autograd-aware block-sparse attention. Use this for training."""
    return BSAttnFunction.apply(q, k, v, sel, gate_bias, causal)
