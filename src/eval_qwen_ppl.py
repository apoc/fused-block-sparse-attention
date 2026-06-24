"""Task 6: held-out perplexity for the block-sparse swap on Qwen3.6-35B-A3B.

One model load, looped over contexts. Per ctx: stock, fastpath (integration gate),
all-blocks-gather (within-path ceiling; only at <=16K, it is O(N^2)), SelectMax
top-k sweep, random. Primary delta vs all-blocks-gather when present (cancels bf16
MoE amplification) else vs stock. Loss via hidden-states + chunked lm_head, so full
logits are never materialized (fits even at 64K).
"""
import argparse, math, json, time, statistics
import torch
import torch.nn.functional as F
from patch_qwen import load_model, patch_attention, unpatch, find_attn_modules
from qwen_blocksparse import SelectAll, SelectMax, SelectRandom

CORPUS = "eval_corpus_raw.txt"
BS = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctxs", default="16384,32768,65536")
    ap.add_argument("--nseq", type=int, default=4)
    ap.add_argument("--topks", default="8,16,32")
    ap.add_argument("--rand", type=int, default=16)
    ap.add_argument("--allgather_max_ctx", type=int, default=16384)
    ap.add_argument("--loss_chunk", type=int, default=2048)
    a = ap.parse_args()
    ctxs = [int(x) for x in a.ctxs.split(",")]
    topks = [int(x) for x in a.topks.split(",")]

    tok, model = load_model()
    assert len(find_attn_modules(model)) == 10
    toks = tok(open(CORPUS).read()[5000:], return_tensors="pt").input_ids[0]
    print(f"corpus {len(toks)} tokens", flush=True)

    @torch.no_grad()
    def ppl(ids_all):
        seq_losses = []
        for i in range(ids_all.shape[0]):
            ids = ids_all[i:i + 1]
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
            seq_losses.append(tot / ntok)
        return math.exp(statistics.mean(seq_losses))

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

        base = res.get("allgather", res["stock"])
        bname = "gather" if "allgather" in res else "stock"
        stock = res["stock"]
        pct = lambda x: 100 * (res[x] - base) / base
        print(f"--- deltas (vs {bname}, vs stock) ---", flush=True)
        for k in [f"max{x}" for x in topks] + [f"rand{a.rand}"]:
            print(f"{k:8s} {res[k]:.4f}   {pct(k):+.2f}% vs {bname}   "
                  f"{100*(res[k]-stock)/stock:+.2f}% vs stock", flush=True)
        json.dump(res, open(f"qwen_poc_ppl_{ctx}.json", "w"), indent=2)
        print(f"wrote qwen_poc_ppl_{ctx}.json", flush=True)

    for ctx in ctxs:
        eval_ctx(ctx)


if __name__ == "__main__":
    main()
