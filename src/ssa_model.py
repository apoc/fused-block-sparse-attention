"""Core attention modules for the MiniSSA experiment.

  * DenseAttention   — exact O(N^2) attention via PyTorch SDPA (flash kernel).
  * BlockSparseSSA   — content-dependent block selection + EXACT attention over
                       the selected blocks (gather; no N x N matrix).
  * CentroidSSA      — sub-quadratic selector: route query/key blocks through C
                       learned centroids (O(N*C)) instead of scoring all nblk^2
                       block pairs. (Sub-quadratic-selection goal.)

Selector training hooks (both sparse modules):
  * `last_block_logits` — raw block scores for the LAST query block, for the
    auxiliary routing loss (L_route, supervised by the known needle block).
  * `gate=True` — MoE-style score->logit coupling: each selected block's keys get
    an additive log-sigmoid(score) bias in SDPA, so the selector also gets task
    gradient. Off by default => forward unchanged, Exp 0 exactness preserved.

Refinements vs the original:
  * MLP selector (2-layer) with mean AND max block pooling (a strong single-token
    match isn't washed out by averaging).
  * The forced local block is added ON TOP of the topk content budget
    (`kk = topk content blocks + the own block`), so `topk` counts content picks
    and `topk=1` is meaningful (own + 1 needle).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _sdpa(q, k, v, attn_mask=None):
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)


def _mlp_sel(d, sel_dim):
    return nn.Sequential(nn.Linear(d, sel_dim), nn.GELU(), nn.Linear(sel_dim, sel_dim))


class DenseAttention(nn.Module):
    def __init__(self, d, h, causal=False):
        super().__init__()
        self.h, self.dh = h, d // h
        self.causal = causal
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)

    def forward(self, x, needle_pos=None):
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        q = q.view(B, L, self.h, self.dh).transpose(1, 2)
        k = k.view(B, L, self.h, self.dh).transpose(1, 2)
        v = v.view(B, L, self.h, self.dh).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.o(out)


def _gather_and_attend(q, k, v, sel, B, H, nblk, Bs, dh, raw, gate, extra=None):
    """Shared gather + per-block exact SDPA. sel: (B,nblk,kk) key-block indices.
    extra: optional additive mask (B,nblk,Bs_q,kk*Bs) broadcast over heads."""
    kk = sel.shape[-1]
    kb = k.view(B, H, nblk, Bs, dh)
    vb = v.view(B, H, nblk, Bs, dh)
    bi = torch.arange(B, device=q.device).view(B, 1, 1, 1).expand(B, H, nblk, kk)
    hi = torch.arange(H, device=q.device).view(1, H, 1, 1).expand(B, H, nblk, kk)
    si = sel.view(B, 1, nblk, kk).expand(B, H, nblk, kk)
    k_sel = kb[bi, hi, si].reshape(B, H, nblk, kk * Bs, dh)
    v_sel = vb[bi, hi, si].reshape(B, H, nblk, kk * Bs, dh)
    qB = q.view(B, H, nblk, Bs, dh)

    mask = None
    if gate:
        raw_sel = torch.gather(raw, -1, sel)                 # (B,nblk,kk)
        gbias = F.logsigmoid(raw_sel).unsqueeze(1).expand(B, H, nblk, kk)
        mask = gbias.repeat_interleave(Bs, dim=-1).reshape(B, H, nblk, 1, kk * Bs).to(qB.dtype)
    if extra is not None:
        ex = extra.unsqueeze(1).to(qB.dtype)                 # (B,1,nblk,Bs_q,kk*Bs)
        mask = ex if mask is None else (mask + ex)
    if mask is None:
        attn_mask = None
    else:
        Sq = mask.shape[-2]
        attn_mask = mask.expand(B, H, nblk, Sq, kk * Bs).reshape(B * H * nblk, Sq, kk * Bs)

    out = _sdpa(
        qB.reshape(B * H * nblk, Bs, dh),
        k_sel.reshape(B * H * nblk, kk * Bs, dh),
        v_sel.reshape(B * H * nblk, kk * Bs, dh),
        attn_mask,
    ).reshape(B, H, nblk, Bs, dh)
    return out


class BlockSparseSSA(nn.Module):
    """Content-dependent block-sparse attention (O(nblk^2) selection)."""
    def __init__(self, d, h, block=64, topk=8, sel_dim=32, gate=False, causal=False, use_triton=False):
        super().__init__()
        self.h, self.dh = h, d // h
        self.block, self.topk = block, topk
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.sel_q = _mlp_sel(d, sel_dim)
        self.sel_k = _mlp_sel(d, sel_dim)
        self.sel_dim = sel_dim
        self.gate = gate
        self.causal = causal
        self.use_triton = use_triton
        self.distill = False        # set True during LM training for self-distillation
        self.last_hit = None
        self.last_block_logits = None
        self.distill_logits = None  # (B,nblk,nblk) causal selector scores
        self.distill_target = None  # (B,nblk,nblk) dense per-block attention mass (detached)

    def _scores(self, x, B, nblk, Bs):
        sq = self.sel_q(x).view(B, nblk, Bs, self.sel_dim).mean(2)     # (B,nblk,s)
        skv = self.sel_k(x).view(B, nblk, Bs, self.sel_dim)
        sk_mean = skv.mean(2)
        sk_max = skv.max(2).values
        return torch.maximum(sq @ sk_mean.transpose(-1, -2),
                             sq @ sk_max.transpose(-1, -2))           # (B,nblk,nblk)

    def forward(self, x, needle_pos=None):
        B, L, D = x.shape
        Bs = self.block
        pad = (Bs - L % Bs) % Bs
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        Lp = L + pad
        nblk = Lp // Bs
        H, dh = self.h, self.dh
        dev = x.device

        q, k, v = self.qkv(x).chunk(3, -1)
        q = q.view(B, Lp, H, dh).transpose(1, 2)
        k = k.view(B, Lp, H, dh).transpose(1, 2)
        v = v.view(B, Lp, H, dh).transpose(1, 2)

        raw = self._scores(x, B, nblk, Bs)                            # (B,nblk,nblk)
        self.last_block_logits = raw[:, nblk - 1, :]

        if self.distill:
            # self-distillation target: dense per-block attention mass from main q,k
            scale = dh ** -0.5
            att = (q @ k.transpose(-1, -2)) * scale                   # (B,H,Lp,Lp)
            cm = torch.triu(torch.ones(Lp, Lp, dtype=torch.bool, device=dev), 1)
            att = att.masked_fill(cm, float("-inf")).softmax(-1)
            am = att.view(B, H, nblk, Bs, nblk, Bs).sum(-1).mean(3)    # (B,H,nblk,nblk)
            self.distill_target = am.mean(1).detach()                 # (B,nblk,nblk)
            futd = torch.arange(nblk, device=dev).view(nblk, 1) < torch.arange(nblk, device=dev).view(1, nblk)
            self.distill_logits = raw.masked_fill(futd.unsqueeze(0), -1e9)  # finite mask (stable CE)

        # ---- decoupled selection: own block + topk CONTENT blocks ----
        diag = torch.arange(nblk, device=dev)
        content = raw.clone()
        content[:, diag, diag] = float("-inf")                        # exclude own
        if self.causal:
            fut = diag.view(nblk, 1) < diag.view(1, nblk)             # (nblk,nblk) key j>i
            content = content.masked_fill(fut.unsqueeze(0), float("-inf"))
        kk_c = min(self.topk, nblk - 1)
        own = diag.view(1, nblk, 1).expand(B, nblk, 1)
        if kk_c > 0:
            sel_c = content.topk(kk_c, dim=-1).indices                # (B,nblk,kk_c)
            sel = torch.cat([own, sel_c], dim=-1)
        else:
            sel = own.contiguous()
        kk = sel.shape[-1]

        extra = None
        if self.causal:
            NEG = torch.finfo(torch.float32).min
            qidx = diag.view(1, nblk, 1)                              # (1,nblk,1) = i
            blk_valid = torch.ones(B, nblk, kk, dtype=torch.bool, device=dev)
            if kk > 1:
                blk_valid[..., 1:] = sel[..., 1:] < qidx             # content: strictly past
            # Build the additive mask via broadcast — avoid the O(B·nblk·Bs·kk·Bs)
            # 5D intermediate that the old code allocated and masked_fill'd.
            # Block validity: (B,nblk,kk) → (B,nblk,1,kk*Bs)
            blk_add = torch.zeros(B, nblk, kk, device=dev, dtype=torch.float32)
            blk_add.masked_fill_(~blk_valid, NEG)
            blk_add = blk_add.repeat_interleave(Bs, dim=-1).unsqueeze(2)  # (B,nblk,1,kk*Bs)
            # Own-block causal (kk=0 only): (Bs,kk*Bs) with triu in the first Bs cols
            tri_add = torch.zeros(Bs, kk * Bs, device=dev, dtype=torch.float32)
            tri_add[:, :Bs] = torch.triu(torch.full((Bs, Bs), NEG, device=dev, dtype=torch.float32), 1)
            extra = blk_add + tri_add  # (B,nblk,Bs,kk*Bs) via broadcast

        if self.use_triton:
            from triton_kernel import bsattn as _bsattn
            sel_i32 = sel.to(torch.int32)
            gate_bias = None
            if self.gate:
                raw_sel = torch.gather(raw, -1, sel)                 # (B,nblk,kk)
                gate_bias = F.logsigmoid(raw_sel)
            out = _bsattn(q, k, v, sel_i32, gate_bias, causal=self.causal)  # (B,H,Lp,dh)
        else:
            out = _gather_and_attend(q, k, v, sel, B, H, nblk, Bs, dh, raw, self.gate, extra)
        out = out.reshape(B, H, Lp, dh).transpose(1, 2).reshape(B, Lp, D)[:, :L]

        if needle_pos is not None:
            needle_blk = needle_pos // Bs
            chosen_last = sel[:, nblk - 1, :]
            self.last_hit = (chosen_last == needle_blk.unsqueeze(-1)).any(-1).float().mean().item()
        return self.o(out)


class CentroidSSA(nn.Module):
    """Sub-quadratic selector: route blocks through C learned centroids.

    Selection is O(nblk * C) (+ bucketed gather), NOT O(nblk^2): no nblk x nblk
    tensor is ever formed. Each key-block is assigned to its nearest centroid and
    dropped into a fixed-capacity bucket; each query-block routes to its top-`cpq`
    centroids and attends to those buckets (cpq*cap blocks) plus its own block.
    Per-query budget is FIXED => attention stays linear in N too.
    Exactness check: n_centroids=1, cpq=1, cap>=nblk => all blocks => == dense.
    """
    def __init__(self, d, h, block=64, topk=8, sel_dim=32, n_centroids=16,
                 cpq=2, cap=4, gate=False, refine_topk=None):
        super().__init__()
        self.h, self.dh = h, d // h
        self.block, self.topk = block, topk
        self.cpq, self.cap = cpq, cap
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.sel_q = _mlp_sel(d, sel_dim)
        self.sel_k = _mlp_sel(d, sel_dim)
        self.sel_dim = sel_dim
        self.cent = nn.Parameter(torch.randn(n_centroids, sel_dim) * (sel_dim ** -0.5))
        self.refine_topk = refine_topk
        self.gate = gate
        self.last_hit = None
        self.last_block_logits = None
        self.align_q = None        # (B,C) last query-block routing logits
        self.align_k = None        # (B,C) needle key-block routing logits

    def forward(self, x, needle_pos=None):
        B, L, D = x.shape
        Bs = self.block
        pad = (Bs - L % Bs) % Bs
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        Lp = L + pad
        nblk = Lp // Bs
        H, dh = self.h, self.dh
        C = self.cent.shape[0]
        cpq = min(self.cpq, C)
        cap = min(self.cap, nblk)
        dev = x.device

        q, k, v = self.qkv(x).chunk(3, -1)
        q = q.view(B, Lp, H, dh).transpose(1, 2)
        k = k.view(B, Lp, H, dh).transpose(1, 2)
        v = v.view(B, Lp, H, dh).transpose(1, 2)

        sq = self.sel_q(x).view(B, nblk, Bs, self.sel_dim).mean(2)     # (B,nblk,s)
        skv = self.sel_k(x).view(B, nblk, Bs, self.sel_dim)
        sk = torch.maximum(skv.mean(2), skv.max(2).values)            # (B,nblk,s)

        # ---- O(nblk*C) routing (no nblk^2) ----
        kc = sk @ self.cent.t()                                        # (B,nblk,C)
        qc = sq @ self.cent.t()                                        # (B,nblk,C)
        key_cent = kc.argmax(-1)                                       # (B,nblk)
        top_c = qc.topk(cpq, dim=-1).indices                          # (B,nblk,cpq)

        # route-loss logits for the LAST query block (gather, O(nblk))
        self.last_block_logits = qc[:, nblk - 1, :].gather(1, key_cent)  # (B,nblk)

        # ---- fixed-capacity buckets (B,C,cap) of key-block ids via cumcount ----
        oh = F.one_hot(key_cent, C)                                    # (B,nblk,C)
        within = (oh.cumsum(1) - oh).gather(2, key_cent.unsqueeze(-1)).squeeze(-1)
        keep = within < cap
        buckets = torch.full((B, C, cap), -1, dtype=torch.long, device=dev)
        bI = torch.arange(B, device=dev).view(B, 1).expand(B, nblk)
        jI = torch.arange(nblk, device=dev).view(1, nblk).expand(B, nblk)
        buckets[bI[keep], key_cent[keep], within[keep]] = jI[keep]

        # gather each query block's top-cpq buckets -> (B,nblk,cpq*cap), + own block
        bexp = buckets.unsqueeze(1).expand(B, nblk, C, cap)
        selb = torch.gather(bexp, 2, top_c.unsqueeze(-1).expand(B, nblk, cpq, cap))
        selb = selb.reshape(B, nblk, cpq * cap)
        own = torch.arange(nblk, device=dev).view(1, nblk, 1).expand(B, nblk, 1)
        selb = selb.masked_fill(selb == own, -1)                       # dedup own (no double-count)
        # ---- within-bucket refinement: score candidates with MAIN attention q/k ----
        # Uses the task-trained q/k (not sel_q/sel_k which are coarse-routing only).
        # O(cpq*cap) per query block — constant, independent of nblk. Total stays O(nblk*C).
        if self.refine_topk is not None and self.refine_topk < cpq * cap:
            cand_valid = selb >= 0                                   # (B,nblk,cpq*cap)
            cand_idx = selb.clamp(min=0)
            # block-level mean of main q/k (already computed above, per-head averaged)
            q_blk = q.view(B, H, nblk, Bs, dh).mean(3).mean(1)     # (B,nblk,dh)
            k_blk = k.view(B, H, nblk, Bs, dh).mean(3).mean(1)     # (B,nblk,dh)
            bI2 = torch.arange(B, device=dev).view(B, 1, 1)
            k_cand = k_blk[bI2, cand_idx]                          # (B,nblk,cpq*cap,dh)
            cand_score = (q_blk.unsqueeze(2) * k_cand).sum(-1)     # (B,nblk,cpq*cap)
            cand_score = cand_score.masked_fill(~cand_valid, float("-inf"))
            rk = min(self.refine_topk, cpq * cap)
            refine_idx = cand_score.topk(rk, dim=-1).indices       # (B,nblk,rk)
            selb = selb.gather(-1, refine_idx)                     # (B,nblk,rk)
        sel_raw = torch.cat([own, selb], dim=-1)                       # (B,nblk,1+rk) pads=-1
        valid = sel_raw >= 0
        sel = sel_raw.clamp(min=0)
        kk = sel.shape[-1]

        # ---- gather K/V and attend, masking padded (invalid) blocks ----
        kb = k.view(B, H, nblk, Bs, dh)
        vb = v.view(B, H, nblk, Bs, dh)
        bi = torch.arange(B, device=dev).view(B, 1, 1, 1).expand(B, H, nblk, kk)
        hi = torch.arange(H, device=dev).view(1, H, 1, 1).expand(B, H, nblk, kk)
        si = sel.view(B, 1, nblk, kk).expand(B, H, nblk, kk)
        k_sel = kb[bi, hi, si].reshape(B, H, nblk, kk * Bs, dh)
        v_sel = vb[bi, hi, si].reshape(B, H, nblk, kk * Bs, dh)
        qB = q.view(B, H, nblk, Bs, dh)

        add = torch.zeros(B, H, nblk, kk, device=dev, dtype=qB.dtype)
        add = add.masked_fill(~valid.view(B, 1, nblk, kk), float("-inf"))
        if self.gate:
            sel_cent = key_cent.gather(1, sel.reshape(B, -1)).reshape(B, nblk, kk)
            gscore = torch.gather(qc, 2, sel_cent)                     # (B,nblk,kk)
            gbias = F.logsigmoid(gscore).masked_fill(~valid, 0.0)
            add = add + gbias.view(B, 1, nblk, kk).to(qB.dtype)
        attn_mask = add.repeat_interleave(Bs, dim=-1).reshape(B * H * nblk, 1, kk * Bs)

        out = _sdpa(qB.reshape(B * H * nblk, Bs, dh),
                    k_sel.reshape(B * H * nblk, kk * Bs, dh),
                    v_sel.reshape(B * H * nblk, kk * Bs, dh),
                    attn_mask).reshape(B, H, nblk, Bs, dh)
        out = out.reshape(B, H, Lp, dh).transpose(1, 2).reshape(B, Lp, D)[:, :L]

        if needle_pos is not None:
            needle_blk = needle_pos // Bs
            chosen_last = sel_raw[:, nblk - 1, :]
            self.last_hit = (chosen_last == needle_blk.unsqueeze(-1)).any(-1).float().mean().item()
            # InfoNCE alignment hooks: stash last query block centroid logits
            # and ALL key-block centroid logits so the training loop can compute
            # a contrastive loss (positive = needle block, negatives = rest).
            self.align_q = qc[:, nblk - 1, :]                              # (B,C)
            self.align_k_all = kc                                          # (B,nblk,C)
        return self.o(out)


class LSHBucketSSA(nn.Module):
    """Linear-cost selection via LSH bucketing + full-fidelity refine.

    Selection is O(nblk * n_rounds * (n_buckets + cap)) -- no nblk^2 matrix is
    ever formed. Each key block is hashed (argmax of rep @ R, R fixed random)
    into one bucket per round; each query block reads its own bucket across
    rounds, then RE-SCORES the candidate blocks with the real sel_q.sel_k dot
    (full fidelity) and keeps the top-`topk`, matching BlockSparseSSA's budget so
    the attention read sees an identical-size, lossless candidate set ("prune the
    search, not the representation").

    Learning: R is a fixed buffer (not trained). The selector MLPs learn
    hash-friendly reps via (a) the auxiliary routing loss on the dense last-block
    logits (sq_last . sk_all, O(nblk)) and (b) gate coupling on selected blocks.
    Higher sq.sk similarity -> higher LSH collision probability -> better recall.
    """
    def __init__(self, d, h, block=32, topk=4, sel_dim=32, n_rounds=4,
                 n_buckets=8, cap=8, gate=False, causal=False, scale=None):
        super().__init__()
        self.h, self.dh = h, d // h
        self.block, self.topk = block, topk
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.sel_q = _mlp_sel(d, sel_dim)
        self.sel_k = _mlp_sel(d, sel_dim)
        self.sel_dim = sel_dim
        self.scale = float(sel_dim ** 0.5) if scale is None else float(scale)  # gate/route cosine temperature
        self.n_rounds, self.n_buckets, self.cap = n_rounds, n_buckets, cap
        self.gate, self.causal = gate, causal
        self.dense_select = False  # train: dense all-pairs top-k (allowed); eval: LSH bucketing
        self.register_buffer("R", torch.randn(sel_dim, n_rounds, n_buckets))
        self.last_hit = None
        self.last_block_logits = None

    def forward(self, x, needle_pos=None):
        B, L, D = x.shape
        Bs = self.block
        pad = (Bs - L % Bs) % Bs
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        Lp = L + pad
        nblk = Lp // Bs
        H, dh = self.h, self.dh
        dev = x.device
        R_, NB, cap = self.n_rounds, self.n_buckets, min(self.cap, nblk)

        q, k, v = self.qkv(x).chunk(3, -1)
        q = q.view(B, Lp, H, dh).transpose(1, 2)
        k = k.view(B, Lp, H, dh).transpose(1, 2)
        v = v.view(B, Lp, H, dh).transpose(1, 2)

        sq = F.normalize(self.sel_q(x).view(B, nblk, Bs, self.sel_dim).mean(2), dim=-1)  # (B,nblk,s)
        skv = self.sel_k(x).view(B, nblk, Bs, self.sel_dim)
        sk = F.normalize(torch.maximum(skv.mean(2), skv.max(2).values), dim=-1)          # (B,nblk,s)
        diag = torch.arange(nblk, device=dev)

        # route-loss hook: dense scores for the LAST query block only, O(nblk)
        self.last_block_logits = self.scale * torch.bmm(sk, sq[:, nblk - 1:nblk].transpose(-1, -2)).squeeze(-1)

        if self.dense_select:
            # train-time dense all-pairs selection (O(nblk^2), allowed at train time).
            # Same normalized-cosine geometry + gate as the LSH path below.
            raw = self.scale * (sq @ sk.transpose(-1, -2))                # (B,nblk,nblk)
            raw = raw.masked_fill(diag.view(1, nblk, 1) == diag.view(1, 1, nblk), float("-inf"))
            if self.causal:
                raw = raw.masked_fill(diag.view(1, 1, nblk) > diag.view(1, nblk, 1), float("-inf"))
            kk_c = min(self.topk, max(1, nblk - 1))
            topv, topi = raw.topk(kk_c, dim=-1)
            sel_content = topi                                           # top-k indices ARE block ids
            cont_valid = torch.isfinite(topv)
        else:
            # ---- LSH bucketing (fixed random R), O(nblk * R_ * NB) ----
            kbucket = torch.einsum("bns,src->bnrc", sk, self.R).argmax(-1)  # (B,nblk,R_)
            qbucket = torch.einsum("bns,src->bnrc", sq, self.R).argmax(-1)  # (B,nblk,R_)
            bI = torch.arange(B, device=dev).view(B, 1).expand(B, nblk)
            jI = diag.view(1, nblk).expand(B, nblk)
            cands = []
            for r in range(R_):
                kb_r, qb_r = kbucket[..., r], qbucket[..., r]
                oh = F.one_hot(kb_r, NB)
                within = (oh.cumsum(1) - oh).gather(2, kb_r.unsqueeze(-1)).squeeze(-1)
                keep = within < cap
                buckets = torch.full((B, NB, cap), -1, dtype=torch.long, device=dev)
                buckets[bI[keep], kb_r[keep], within[keep]] = jI[keep]
                cands.append(torch.gather(buckets, 1, qb_r.unsqueeze(-1).expand(B, nblk, cap)))
            cand = torch.cat(cands, dim=-1)                               # (B,nblk,P) ids, -1=empty
            P = cand.shape[-1]
            valid = cand >= 0
            cand_c = cand.clamp(min=0)
            # dedup across rounds (P is constant, so this stays linear in nblk)
            eq = cand_c.unsqueeze(-1) == cand_c.unsqueeze(-2)
            lower = torch.tril(torch.ones(P, P, dtype=torch.bool, device=dev), -1)
            valid = valid & ~(eq & lower).any(-1)
            bb = torch.arange(B, device=dev).view(B, 1, 1)
            sk_cand = sk[bb, cand_c]                                       # (B,nblk,P,s)
            score = self.scale * (sq.unsqueeze(2) * sk_cand).sum(-1)       # (B,nblk,P) cosine*scale
            score = score.masked_fill(~valid, float("-inf"))
            score = score.masked_fill(cand_c == diag.view(1, nblk, 1), float("-inf"))  # own added below
            if self.causal:
                score = score.masked_fill(cand_c > diag.view(1, nblk, 1), float("-inf"))
            kk_c = min(self.topk, max(1, nblk - 1))
            rk = min(kk_c, P)
            topv, topi = score.topk(rk, dim=-1)
            sel_content = torch.gather(cand_c, -1, topi)                  # (B,nblk,rk)
            cont_valid = torch.isfinite(topv)

        own = diag.view(1, nblk, 1).expand(B, nblk, 1)
        own_score = self.scale * (sq * sk).sum(-1, keepdim=True)       # (B,nblk,1)
        sel = torch.cat([own, sel_content], dim=-1)                   # (B,nblk,kk)
        sval = torch.cat([torch.ones(B, nblk, 1, dtype=torch.bool, device=dev), cont_valid], dim=-1)
        gscore = torch.cat([own_score, topv.masked_fill(~cont_valid, 0.0)], dim=-1)
        kk = sel.shape[-1]

        # ---- gather selected K/V and attend (non-causal for MQAR) ----
        kb = k.view(B, H, nblk, Bs, dh)
        vb = v.view(B, H, nblk, Bs, dh)
        bi = torch.arange(B, device=dev).view(B, 1, 1, 1).expand(B, H, nblk, kk)
        hi = torch.arange(H, device=dev).view(1, H, 1, 1).expand(B, H, nblk, kk)
        si = sel.view(B, 1, nblk, kk).expand(B, H, nblk, kk)
        k_sel = kb[bi, hi, si].reshape(B, H, nblk, kk * Bs, dh)
        v_sel = vb[bi, hi, si].reshape(B, H, nblk, kk * Bs, dh)
        qB = q.view(B, H, nblk, Bs, dh)

        add = torch.zeros(B, H, nblk, kk, device=dev, dtype=qB.dtype)
        add = add.masked_fill(~sval.view(B, 1, nblk, kk), float("-inf"))
        if self.gate:
            gbias = F.logsigmoid(gscore).masked_fill(~sval, 0.0)
            add = add + gbias.view(B, 1, nblk, kk).to(qB.dtype)
        if self.causal:
            NEG = float("-inf")   # bf16-safe (finfo(float32).min overflows bf16 under autocast)
            tri = torch.zeros(Bs, kk * Bs, device=dev, dtype=qB.dtype)
            tri[:, :Bs] = torch.triu(torch.full((Bs, Bs), NEG, device=dev, dtype=qB.dtype), 1)
            attn_mask = (add.repeat_interleave(Bs, dim=-1).reshape(B, H, nblk, 1, kk * Bs)
                         + tri).reshape(B * H * nblk, Bs, kk * Bs)
        else:
            attn_mask = add.repeat_interleave(Bs, dim=-1).reshape(B * H * nblk, 1, kk * Bs)

        out = _sdpa(qB.reshape(B * H * nblk, Bs, dh),
                    k_sel.reshape(B * H * nblk, kk * Bs, dh),
                    v_sel.reshape(B * H * nblk, kk * Bs, dh),
                    attn_mask).reshape(B, H, nblk, Bs, dh)
        out = out.reshape(B, H, Lp, dh).transpose(1, 2).reshape(B, Lp, D)[:, :L]

        if needle_pos is not None:
            needle_blk = needle_pos // Bs
            chosen_last = sel[:, nblk - 1, :]
            self.last_hit = (chosen_last == needle_blk.unsqueeze(-1)).any(-1).float().mean().item()
        return self.o(out)


def _make_attn(d, h, attn, kw):
    if attn == "dense":
        return DenseAttention(d, h, causal=kw.get("causal", False))
    if attn == "centroid":
        return CentroidSSA(d, h, **kw)
    if attn == "lsh":
        return LSHBucketSSA(d, h, **kw)
    return BlockSparseSSA(d, h, **kw)


class TinyTransformer(nn.Module):
    def __init__(self, vocab, n_val, d=128, h=4, layers=2, max_len=4096,
                 attn="dense", **kw):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(1, max_len, d) * 0.02)
        self.blocks = nn.ModuleList(nn.ModuleDict({
            "ln1": nn.LayerNorm(d), "attn": _make_attn(d, h, attn, kw),
            "ln2": nn.LayerNorm(d),
            "mlp": nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)),
        }) for _ in range(layers))
        self.head = nn.Linear(d, n_val)

    def forward(self, toks, needle_pos=None):
        x = self.emb(toks) + self.pos[:, : toks.size(1)]
        for b in self.blocks:
            x = x + b["attn"](b["ln1"](x), needle_pos=needle_pos)
            x = x + b["mlp"](b["ln2"](x))
        return self.head(x[:, -1])

    def set_dense_select(self, flag):
        for b in self.blocks:
            if hasattr(b["attn"], "dense_select"):
                b["attn"].dense_select = flag

    def selection_hit(self):
        hs = [b["attn"].last_hit for b in self.blocks
              if hasattr(b["attn"], "last_hit") and b["attn"].last_hit is not None]
        return sum(hs) / len(hs) if hs else None

    def route_logits(self):
        return [b["attn"].last_block_logits for b in self.blocks
                if getattr(b["attn"], "last_block_logits", None) is not None]

    def centroid_align(self):
        return [(b["attn"].align_q, b["attn"].align_k_all) for b in self.blocks
                if getattr(b["attn"], "align_q", None) is not None]
