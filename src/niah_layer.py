"""Per-layer needle-rank diagnostic: is retrieval concentrated in specific
full-attention layers, or uniform across all 10?

For each NIAH sample, records the needle block's rank under the SelectMax score
(pool) and under the single answer-trigger query token (exact), for EACH
full-attention layer separately (not aggregated across layers). Output: per-layer
median/min rank over (6 samples x 2 KV heads = 12 points), at each context.

If retrieval is layer-concentrated (a few layers rank the needle ~2-8 while most
rank it ~30+), layer-adaptive budget allocation is worth testing.
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
    off = (L - len(ans) - 1) - (L - BS)
    return torch.tensor(ids, dtype=torch.long)[None], list(range(pos // BS, (pos + len(needle) - 1) // BS + 1)), off


def make_store(store):
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
        kmin = kb.min(3).values[0]; kmax = kb.max(3).values[0]
        qlast = q[:, :, -BS:, :].view(B, Hkv, grp, BS, d).mean(2)[0]
        store[self.layer_idx] = (kmin.float().cpu(), kmax.float().cpu(), qlast.float().cpu())
        kr, vr = k.repeat_interleave(grp, 1), v.repeat_interleave(grp, 1)
        out = F.scaled_dot_product_attention(q, kr, vr, is_causal=True, scale=self.scaling)
        out = out.transpose(1, 2).reshape(*ishape, -1).contiguous() * torch.sigmoid(gate)
        return self.o_proj(out), None
    return forward


def rank_of(qvec, kmin, kmax, needle_blocks, nblk):
    out = []
    for h in range(qvec.shape[0]):
        qpos = qvec[h].clamp(min=0); qneg = qvec[h].clamp(max=0)
        sc = (torch.einsum("d,jd->j", qpos, kmax[h]) + torch.einsum("d,jd->j", qneg, kmin[h]))
        order = sc.argsort(descending=True).tolist()
        pos_of = {b: r for r, b in enumerate(order)}
        out.append(min(pos_of[b] for b in needle_blocks if b < nblk) + 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctxs", default="16384,32768,65536")
    ap.add_argument("--depths", default="0.1,0.5,0.9")
    ap.add_argument("--needles_per_depth", type=int, default=2)
    a = ap.parse_args()
    ctxs = [int(x) for x in a.ctxs.split(",")]
    depths = [float(x) for x in a.depths.split(",")]

    tok, model = load_model()
    mods = find_attn_modules(model)
    assert len(mods) == 10
    filler = tok(open(CORPUS).read()[5000:], add_special_tokens=False).input_ids

    rng = random.Random(0)
    samples = [(d, rng.choice(KEYS), rng.randint(1000000, 9999999))
               for d in depths for _ in range(a.needles_per_depth)]

    store = {}
    for m in mods:
        m.forward = types.MethodType(make_store(store), m)

    out = {}
    for ctx in ctxs:
        nblk = ctx // BS
        pl = {li: [] for li in [m.layer_idx for m in mods]}
        el = {li: [] for li in pl}
        for (depth, key, value) in samples:
            ids, needle_blocks, off = build(tok, filler, ctx, depth, key, value)
            store.clear()
            with torch.no_grad():
                model.model(input_ids=ids.cuda(), use_cache=False)
            for li, (kmin, kmax, qlast) in store.items():
                pl[li] += rank_of(qlast.mean(1), kmin, kmax, needle_blocks, nblk)
                el[li] += rank_of(qlast[:, off, :], kmin, kmax, needle_blocks, nblk)

        cell = {}
        print(f"\n##### ctx={ctx} nblk={nblk} (per-layer needle rank, lower=better) #####", flush=True)
        print(f"  layer | pool med | pool min | exact med | exact min", flush=True)
        for li in sorted(pl):
            s = {"pool_med": int(statistics.median(pl[li])), "pool_min": min(pl[li]),
                 "exact_med": int(statistics.median(el[li])), "exact_min": min(el[li])}
            cell[li] = s
            print(f"  {li:5d} | {s['pool_med']:7d} | {s['pool_min']:8d} | "
                  f"{s['exact_med']:9d} | {s['exact_min']:9d}", flush=True)
        out[str(ctx)] = cell
        json.dump(out, open("qwen_niah_perlayer.json", "w"), indent=2)
        print(f"  wrote qwen_niah_perlayer.json", flush=True)

    for m in mods:
        if "forward" in m.__dict__:
            del m.__dict__["forward"]


if __name__ == "__main__":
    main()
