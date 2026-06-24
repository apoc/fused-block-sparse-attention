"""GQA-aware, chunked, PyTorch block-sparse attention for the Qwen3.6 PoC.

Consumes post-RoPE q/k/v (q: (B,Hq,L,d), k/v: (B,Hkv,L,d), GQA Hq % Hkv == 0) and
replaces "softmax over all keys" with "select top-k key blocks + SDPA over them".
No Triton; memory is held flat by chunking the query-block loop.

Selection is per KV head (shared across its grp query heads). Selectors:
  * SelectMax   - training-free, RoPE-tolerant Quest-style block importance:
                  score(i,j) = sum_d max(qpool_i . kmin_j, qpool_i . kmax_j),
                  the cheap O(nblk^2 d) exact estimate of max_{t,s} q_t . k_s.
  * SelectOracle - top-k by a supplied per-block attention mass (upper bound).
  * SelectRandom - random causal blocks (lower bound).
Every selector appends the own block and (optionally) the sink block 0; the gather
path masks future, padded, and duplicate key positions, so over-selection is safe.
"""
import torch
import torch.nn.functional as F


def _pad_blocks(x, bs):
    L = x.shape[-2]
    nblk = (L + bs - 1) // bs
    pad = nblk * bs - L
    if pad:
        x = F.pad(x, (0, 0, 0, pad))
    return x, nblk, pad


def blocksparse_forward(q, k, v, *, selector, topk, bs, causal=True, scale=None, q_chunk=4):
    B, Hq, L, d = q.shape
    Hkv = k.shape[1]
    grp = Hq // Hkv
    if scale is None:
        scale = d ** -0.5
    nblk = (L + bs - 1) // bs

    # ---- all-blocks fast path (used by the correctness gate) ----
    if selector is None and topk >= nblk:
        kr = k.repeat_interleave(grp, dim=1)
        vr = v.repeat_interleave(grp, dim=1)
        return F.scaled_dot_product_attention(q, kr, vr, is_causal=causal, scale=scale)

    qp, nblk, pad = _pad_blocks(q, bs)
    kp, _, _ = _pad_blocks(k, bs)
    vp, _, _ = _pad_blocks(v, bs)
    Lp = nblk * bs
    dev = q.device

    idx = selector.select(q, k)                      # (B, Hkv, nblk, kk) block ids
    kk = idx.shape[-1]
    kblk = kp.view(B, Hkv, nblk, bs, d)
    vblk = vp.view(B, Hkv, nblk, bs, d)
    out = torch.zeros(B, Hq, Lp, d, dtype=q.dtype, device=dev)
    ar_bs = torch.arange(bs, device=dev)
    tri_kk = torch.tril(torch.ones(kk, kk, dtype=torch.bool, device=dev), -1)

    for i0 in range(0, nblk, q_chunk):
        i1 = min(i0 + q_chunk, nblk)
        qc = i1 - i0
        sel = idx[:, :, i0:i1, :]                    # (B,Hkv,qc,kk)
        sel_e = sel.view(B, Hkv, qc, kk, 1, 1).expand(B, Hkv, qc, kk, bs, d)
        kblk_e = kblk.view(B, Hkv, 1, nblk, bs, d).expand(B, Hkv, qc, nblk, bs, d)
        vblk_e = vblk.view(B, Hkv, 1, nblk, bs, d).expand(B, Hkv, qc, nblk, bs, d)
        kb = torch.gather(kblk_e, 3, sel_e).reshape(B, Hkv, qc, kk * bs, d)
        vb = torch.gather(vblk_e, 3, sel_e).reshape(B, Hkv, qc, kk * bs, d)

        keypos = (sel.unsqueeze(-1) * bs + ar_bs).reshape(B, Hkv, qc, kk * bs)
        qpos = (torch.arange(i0, i1, device=dev) * bs).view(1, 1, qc, 1, 1) + ar_bs.view(1, 1, 1, bs, 1)
        kp_ = keypos.view(B, Hkv, qc, 1, kk * bs)
        dup = (sel.unsqueeze(-1) == sel.unsqueeze(-2)) & tri_kk
        dup = dup.any(-1).repeat_interleave(bs, dim=-1).view(B, Hkv, qc, 1, kk * bs)
        valid = (kp_ <= qpos) & (kp_ < L) & (~dup)            # (B,Hkv,qc,bs,kk*bs)

        qchunk = qp[:, :, i0 * bs:i1 * bs, :].reshape(B, Hkv, grp, qc, bs, d)
        scores = torch.einsum('bhgqtd,bhqsd->bhgqts', qchunk, kb) * scale
        scores = scores.masked_fill(~valid.view(B, Hkv, 1, qc, bs, kk * bs), float('-inf'))
        probs = torch.nan_to_num(scores.softmax(-1), nan=0.0)
        ov = torch.einsum('bhgqts,bhqsd->bhgqtd', probs, vb)
        out[:, :, i0 * bs:i1 * bs, :] = ov.reshape(B, Hq, qc * bs, d)

    return out[..., :L, :]


def _finish(score, topk, sink, B, Hkv, nblk, dev):
    diag = torch.arange(nblk, device=dev)
    score = score.masked_fill(diag.view(1, 1, nblk, 1) < diag.view(1, 1, 1, nblk), float('-inf'))
    kk = min(topk, nblk)
    topv, topi = score.topk(kk, dim=-1)
    own = diag.view(1, 1, nblk, 1).expand(B, Hkv, nblk, 1)
    # early query blocks have < kk causal blocks: topk returns -inf (future) picks;
    # clamp those to the own block (gather dedups the duplicates) so idx stays causal.
    topi = torch.where(torch.isfinite(topv), topi, own.expand_as(topi))
    parts = [topi, own]
    if sink:
        parts.append(torch.zeros(B, Hkv, nblk, 1, dtype=torch.long, device=dev))
    return torch.cat(parts, dim=-1).sort(dim=-1).values


class SelectMax:
    """Training-free RoPE-tolerant Quest-style selector."""
    def __init__(self, topk, bs, sink=True):
        self.topk, self.bs, self.sink = topk, bs, sink

    @torch.no_grad()
    def select(self, q, k):
        B, Hq, L, d = q.shape
        Hkv = k.shape[1]; grp = Hq // Hkv; bs = self.bs
        qp, nblk, _ = _pad_blocks(q, bs)
        kp, _, _ = _pad_blocks(k, bs)
        qpool = qp.view(B, Hkv, grp, nblk, bs, d).mean(2).mean(3)       # (B,Hkv,nblk,d)
        kb = kp.view(B, Hkv, nblk, bs, d)
        kmin = kb.min(3).values
        kmax = kb.max(3).values
        qpos = qpool.clamp(min=0)
        qneg = qpool.clamp(max=0)
        score = (torch.einsum('bhid,bhjd->bhij', qpos, kmax)
                 + torch.einsum('bhid,bhjd->bhij', qneg, kmin))          # (B,Hkv,nblk,nblk)
        return _finish(score, self.topk, self.sink, B, Hkv, nblk, q.device)


class SelectOracle:
    """Top-k key blocks by a supplied per-block attention mass (B,Hkv,nblk,nblk)."""
    def __init__(self, topk, bs, sink=True, mass=None):
        self.topk, self.bs, self.sink, self.mass = topk, bs, sink, mass

    @torch.no_grad()
    def select_from_mass(self, mass):
        B, Hkv, nblk, _ = mass.shape
        return _finish(mass.clone(), self.topk, self.sink, B, Hkv, nblk, mass.device)

    @torch.no_grad()
    def select(self, q, k):
        assert self.mass is not None, "SelectOracle needs mass set (per-layer teacher)"
        return self.select_from_mass(self.mass)


class SelectRandom:
    """Random causal blocks (lower-bound baseline)."""
    def __init__(self, topk, bs, sink=True, seed=0):
        self.topk, self.bs, self.sink, self.seed = topk, bs, sink, seed

    @torch.no_grad()
    def select(self, q, k):
        B, Hq, L, d = q.shape
        Hkv = k.shape[1]; bs = self.bs
        nblk = (L + bs - 1) // bs
        g = torch.Generator(device=q.device).manual_seed(self.seed)
        score = torch.rand(B, Hkv, nblk, nblk, generator=g, device=q.device)
        return _finish(score, self.topk, self.sink, B, Hkv, nblk, q.device)


class SelectAll:
    """All blocks (causal-masked in the gather path); for the gather correctness test."""
    def __init__(self, bs):
        self.bs = bs

    @torch.no_grad()
    def select(self, q, k):
        B, Hq, L, d = q.shape
        Hkv = k.shape[1]
        nblk = (L + self.bs - 1) // self.bs
        return torch.arange(nblk, device=q.device).view(1, 1, 1, nblk).expand(B, Hkv, nblk, nblk).contiguous()
