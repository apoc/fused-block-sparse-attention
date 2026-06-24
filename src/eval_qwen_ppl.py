"""Task 6: held-out perplexity for the block-sparse swap on Qwen3.6-35B-A3B.

One model load, looped over contexts. Per ctx: stock, fastpath (integration gate),
all-blocks-gather (within-path ceiling; only at <=16K, it is O(N^2)), SelectMax
top-k sweep, random, and SelectOracle top-k (the ceiling: top-k blocks by the true
dense attention mass). Primary delta vs all-blocks-gather when present (cancels bf16
MoE amplification) else vs stock. Loss via hidden-states + chunked lm_head, so full
logits are never materialized (fits even at 64K).

Oracle mass is recaptured fresh per eval window per layer with a dense capture
forward (numerically the stock attention; mirrors distill_qwen_sel.make_capture),
then bound per layer into SelectOracle. Stock itself stays native (factory=None) so
the integration gate keeps comparing our reconstruction against HF's own attention.
"""
import argparse, math, json, time, statistics, types
import torch
import torch.nn.functional as F
import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as _M
from patch_qwen import load_model, patch_attention, unpatch, find_attn_modules
from qwen_blocksparse import SelectAll, SelectMax, SelectRandom, SelectOracle

CORPUS = "eval_corpus_raw.txt"
BS = 128
apply_rope = _M.apply_rotary_pos_emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctxs", default="16384,32768,65536")
    ap.add_argument("--nseq", type=int, default=4)
    ap.add_argument("--topks", default="8,16,32")
    ap.add_argument("--rand", type=int, default=16)
    ap.add_argument("--allgather_max_ctx", type=int, default=16384)
    ap.add_argument("--oracle_max_ctx", type=int, default=65536)
    ap.add_argument("--loss_chunk", type=int, default=2048)
    a = ap.parse_args()
    ctxs = [int(x) for x in a.ctxs.split(",")]
    topks = [int(x) for x in a.topks.split(",")]

    tok, model = load_model()
    assert len(find_attn_modules(model)) == 10
    toks = tok(open(CORPUS).read()[5000:], return_tensors="pt").input_ids[0]
    print(f"corpus {len(toks)} tokens", flush=True)

    @torch.no_grad()
    def seq_loss(ids):
        L = ids.shape[1]
        hidden = model.model(input_ids=ids, use_cache=False)[0]    # (1,L,D) bf16
        tot, ntok = 0.0, 0
        for c in range(0, L - 1, a.loss_chunk):
            e = min(c + a.loss_chunk, L - 1)
            lg = model.lm_head(hidden[:, c:e]).float()
            tg = ids[:, c + 1:e + 1]
            tot += F.cross_entropy(lg.reshape(-1, lg.size(-1)), tg.reshape(-1),
                                   reduction="sum").item()
            ntok += tg.numel()
        return tot / ntok

    @torch.no_grad()
    def ppl(ids_all):
        seq_losses = [seq_loss(ids_all[i:i + 1]) for i in range(ids_all.shape[0])]
        return math.exp(statistics.mean(seq_losses))

    @torch.no_grad()
    def capture_mass(ids):
        """Dense capture forward -> {layer_idx: true block-mass (1,Hkv,nblk,nblk)}.
        Chunked over query blocks so no L x L matrix is held (memory-flat at 64K)."""
        store = {}

        def cap(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_values=None, **kw):
            ishape = hidden_states.shape[:-1]
            hshape = (*ishape, -1, self.head_dim)
            qy, gate = torch.chunk(
                self.q_proj(hidden_states).view(*ishape, -1, self.head_dim * 2), 2, -1)
            gate = gate.reshape(*ishape, -1)
            q = self.q_norm(qy.view(hshape)).transpose(1, 2)
            k = self.k_norm(self.k_proj(hidden_states).view(hshape)).transpose(1, 2)
            v = self.v_proj(hidden_states).view(hshape).transpose(1, 2)
            cos, sin = position_embeddings
            q, k = apply_rope(q, k, cos, sin)
            B, Hq, L, d = q.shape
            Hkv = k.shape[1]; grp = Hq // Hkv; nblk = (L + BS - 1) // BS
            kr, vr = k.repeat_interleave(grp, 1), v.repeat_interleave(grp, 1)
            kpos = torch.arange(L, device=q.device)
            ablk = torch.zeros(B, Hq, nblk, nblk, device=q.device)
            for i in range(nblk):
                qi = q[:, :, i * BS:(i + 1) * BS, :]
                s = (qi @ kr.transpose(-1, -2)) * self.scaling
                qpos = i * BS + torch.arange(qi.shape[2], device=q.device)
                s = s.float().masked_fill(
                    kpos[None, None, None, :] > qpos[None, None, :, None], float('-inf'))
                aw = s.softmax(-1)
                ablk[:, :, i, :] = aw.view(B, Hq, qi.shape[2], nblk, BS).sum(-1).mean(2)
            store[self.layer_idx] = ablk.view(B, Hkv, grp, nblk, nblk).mean(2)
            out = F.scaled_dot_product_attention(q, kr, vr, is_causal=True, scale=self.scaling)
            out = out.transpose(1, 2).reshape(*ishape, -1).contiguous() * torch.sigmoid(gate)
            return self.o_proj(out), None

        mods = find_attn_modules(model)
        for m in mods:
            m.forward = types.MethodType(cap, m)
        model.model(input_ids=ids, use_cache=False)
        for m in mods:
            if "forward" in m.__dict__:
                del m.__dict__["forward"]
        torch.cuda.empty_cache()
        return store

    @torch.no_grad()
    def oracle_ppls(ids_all, oks):
        """Per sequence: capture true block-mass once, then eval every oracle top-k."""
        losses = {k: [] for k in oks}
        for i in range(ids_all.shape[0]):
            ids = ids_all[i:i + 1]
            mass = capture_mass(ids)
            for k in oks:
                sels = {li: SelectOracle(k, BS, mass=mass[li]) for li in mass}
                patch_attention(model, lambda li, s=sels, kk=k: (s[li], kk, BS))
                losses[k].append(seq_loss(ids))
                unpatch(model)
            del mass
            torch.cuda.empty_cache()
        return {k: math.exp(statistics.mean(losses[k])) for k in oks}

    def eval_ctx(ctx):
        wins = [toks[i * ctx:(i + 1) * ctx] for i in range(a.nseq)]
        wins = [w for w in wins if len(w) == ctx]
        if not wins:
            print(f"ctx={ctx}: not enough tokens, skip", flush=True)
            return
        ids_all = torch.stack(wins).cuda()
        print(f"\n##### ctx={ctx} nseq={len(wins)} nblk={ctx//BS} #####", flush=True)

        def run(name, factory):
            if factory is not None:
                patch_attention(model, factory)
            t = time.time()
            p = ppl(ids_all)
            if factory is not None:
                unpatch(model)
            print(f"{name:12s} ppl={p:.4f}  ({time.time()-t:.0f}s)", flush=True)
            return p

        res = {"ctx": ctx, "nseq": len(wins), "bs": BS, "nblk": ctx // BS}
        res["stock"] = run("stock", None)
        res["fastpath"] = run("fastpath", lambda li: (None, 10 ** 9, BS))
        rel = abs(res["fastpath"] - res["stock"]) / res["stock"]
        res["integration_rel"] = rel
        print(f"INTEGRATION GATE: {'PASS' if rel < 0.01 else 'FAIL'} (rel {rel:.4%})", flush=True)
        if ctx <= a.allgather_max_ctx:
            res["allgather"] = run("allgather", lambda li: (SelectAll(BS), 0, BS))
        for k in topks:
            sm = SelectMax(k, BS)
            res[f"max{k}"] = run(f"max{k}", (lambda li, s=sm, kk=k: (s, kk, BS)))
        sr = SelectRandom(a.rand, BS)
        res[f"rand{a.rand}"] = run(f"rand{a.rand}", lambda li: (sr, a.rand, BS))
        if ctx <= a.oracle_max_ctx:
            t = time.time()
            ops = oracle_ppls(ids_all, topks)
            for k in topks:
                res[f"oracle{k}"] = ops[k]
            print(f"oracle{topks}  {[round(ops[k],4) for k in topks]}  ({time.time()-t:.0f}s)",
                  flush=True)

        base = res.get("allgather", res["stock"])
        bname = "gather" if "allgather" in res else "stock"
        stock = res["stock"]
        pct = lambda x: 100 * (res[x] - base) / base
        names = [f"max{x}" for x in topks] + [f"rand{a.rand}"]
        names += [f"oracle{x}" for x in topks if f"oracle{x}" in res]
        print(f"--- deltas (vs {bname}, vs stock) ---", flush=True)
        for k in names:
            print(f"{k:10s} {res[k]:.4f}   {pct(k):+.2f}% vs {bname}   "
                  f"{100*(res[k]-stock)/stock:+.2f}% vs stock", flush=True)
        json.dump(res, open(f"qwen_poc_ppl_{ctx}.json", "w"), indent=2)
        print(f"wrote qwen_poc_ppl_{ctx}.json", flush=True)

    for ctx in ctxs:
        eval_ctx(ctx)


if __name__ == "__main__":
    main()
