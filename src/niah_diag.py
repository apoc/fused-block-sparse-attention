"""Needle-miss diagnostic: where does each selector rank the needle's KV block for
the answer-generating query, among all blocks (lower is better)?

Three query reductions, all scored with the Quest min/max key bound:
  * pool : block-mean of the last query block          (what SelectMax uses)
  * exact: the lone answer-trigger query token         (decode-time signal)
  * qmax : max over the last query block's tokens       (what SelectMaxQMax uses)

The pool-vs-exact gap shows query pooling dilutes the retrieval signal. The qmax rank
is the decisive one for interpreting the SelectMaxQMax fix: if at 64k the needle's
qmax rank is within budget yet retrieval still fails (qwen_niah_fix.json), the
bottleneck is propagation, not selection; if qmax also ranks it out of budget, the
fix simply does not select it (qmax's per-block max inflates competitor scores too).

Samples mirror run_followups.run_niah exactly. Dense forward only; cheap.
"""
import argparse, json, random, statistics, types
import torch
import torch.nn.functional as F
import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as _M
from patch_qwen import load_model, find_attn_modules
from run_followups import KEYS, CORPUS, BS

apply_rope = _M.apply_rotary_pos_emb


def build(tok, filler, ctx, depth, key, value):
    needle = tok(f"\nThe special magic number for {key} is {value}.\n", add_special_tokens=False).input_ids
    quest = tok(f"\n\nWhat is the special magic number for {key}? The special magic number for {key} is",
                add_special_tokens=False).input_ids
    ans = tok(f" {value}", add_special_tokens=False).input_ids
    budget = ctx - len(needle) - len(quest) - len(ans)
    assert budget > 0
    hay = filler[:budget]
    pos = int(budget * depth)
    ids = hay[:pos] + needle + hay[pos:] + quest + ans
    L = len(ids)
    needle_blocks = list(range(pos // BS, (pos + len(needle) - 1) // BS + 1))
    off = (L - len(ans) - 1) - (L - BS)            # answer-trigger offset within the last block
    return torch.tensor(ids, dtype=torch.long)[None], needle_blocks, off


def make_diag(store):
    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kw):
        ishape = hidden_states.shape[:-1]; hshape = (*ishape, -1, self.head_dim)
        qy, gate = torch.chunk(self.q_proj(hidden_states).view(*ishape, -1, self.head_dim * 2), 2, -1)
        gate = gate.reshape(*ishape, -1)
        q = self.q_norm(qy.view(hshape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states).view(hshape)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hshape).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rope(q, k, cos, sin)
        B, Hq, L, d = q.shape
        Hkv = k.shape[1]; grp = Hq // Hkv; nblk = (L + BS - 1) // BS
        kb = k.view(B, Hkv, nblk, BS, d)
        kmin = kb.min(3).values[0]; kmax = kb.max(3).values[0]                  # (Hkv,nblk,d)
        qlast = q[:, :, -BS:, :].view(B, Hkv, grp, BS, d).mean(2)[0]            # (Hkv,BS,d) last block, grp-pooled
        store[self.layer_idx] = (kmin.float().cpu(), kmax.float().cpu(), qlast.float().cpu())
        kr, vr = k.repeat_interleave(grp, 1), v.repeat_interleave(grp, 1)
        out = F.scaled_dot_product_attention(q, kr, vr, is_causal=True, scale=self.scaling)
        out = out.transpose(1, 2).reshape(*ishape, -1).contiguous() * torch.sigmoid(gate)
        return self.o_proj(out), None
    return forward


def ranks_from_score(sc, needle_blocks, nblk):
    """sc (Hkv,nblk) -> 1-indexed rank of the best needle block, per head."""
    out = []
    for h in range(sc.shape[0]):
        order = sc[h].argsort(descending=True).tolist()
        pos_of = {b: r for r, b in enumerate(order)}
        out.append(min(pos_of[b] for b in needle_blocks if b < nblk) + 1)
    return out


def score_single(qvec, kmin, kmax):
    qpos = qvec.clamp(min=0); qneg = qvec.clamp(max=0)               # (Hkv,d)
    return torch.einsum("hd,hjd->hj", qpos, kmax) + torch.einsum("hd,hjd->hj", qneg, kmin)


def score_qmax(qlast, kmin, kmax):
    qpos = qlast.clamp(min=0); qneg = qlast.clamp(max=0)            # (Hkv,BS,d)
    s = torch.einsum("htd,hjd->htj", qpos, kmax) + torch.einsum("htd,hjd->htj", qneg, kmin)  # (Hkv,BS,nblk)
    return s.max(1).values                                          # (Hkv,nblk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctxs", default="16384,32768,65536")
    ap.add_argument("--depths", default="0.1,0.5,0.9")
    ap.add_argument("--needles_per_depth", type=int, default=2)
    ap.add_argument("--budgets", default="8,16,32")
    a = ap.parse_args()
    ctxs = [int(x) for x in a.ctxs.split(",")]
    depths = [float(x) for x in a.depths.split(",")]
    budgets = [int(x) for x in a.budgets.split(",")]

    tok, model = load_model()
    mods = find_attn_modules(model)
    assert len(mods) == 10
    filler = tok(open(CORPUS).read()[5000:], add_special_tokens=False).input_ids

    rng = random.Random(0)
    samples = []
    for d in depths:
        for _ in range(a.needles_per_depth):
            samples.append((d, rng.choice(KEYS), rng.randint(1000000, 9999999)))

    store = {}
    for m in mods:
        m.forward = types.MethodType(make_diag(store), m)

    out = {}
    for ctx in ctxs:
        nblk = ctx // BS
        agg = {"pool": [], "exact": [], "qmax": []}
        for (depth, key, value) in samples:
            ids, needle_blocks, off = build(tok, filler, ctx, depth, key, value)
            store.clear()
            with torch.no_grad():
                model.model(input_ids=ids.cuda(), use_cache=False)
            for li, (kmin, kmax, qlast) in store.items():
                agg["pool"] += ranks_from_score(score_single(qlast.mean(1), kmin, kmax), needle_blocks, nblk)
                agg["exact"] += ranks_from_score(score_single(qlast[:, off, :], kmin, kmax), needle_blocks, nblk)
                agg["qmax"] += ranks_from_score(score_qmax(qlast, kmin, kmax), needle_blocks, nblk)

        def summ(xs):
            return {"med": int(statistics.median(xs)), "min": min(xs),
                    "in_topk": {k: round(sum(1 for x in xs if x <= k) / len(xs), 3) for k in budgets}}
        cell = {"nblk": nblk, **{name: summ(agg[name]) for name in agg}}
        out[str(ctx)] = cell
        print(f"\n##### ctx={ctx} nblk={nblk} (needle rank, lower=better) #####", flush=True)
        for name in ("pool", "exact", "qmax"):
            s = cell[name]
            print(f"  {name:5s}: median {s['med']:>4}  best {s['min']:>3}  in top-k {s['in_topk']}", flush=True)
        json.dump(out, open("qwen_niah_diag.json", "w"), indent=2)
        print(f"  wrote qwen_niah_diag.json", flush=True)

    for m in mods:
        if "forward" in m.__dict__:
            del m.__dict__["forward"]


if __name__ == "__main__":
    main()
