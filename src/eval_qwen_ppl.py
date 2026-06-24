"""Task 6: held-out perplexity for the block-sparse swap on Qwen3.6-35B-A3B.

One model load. Reports ppl over N windows for:
  stock, fastpath (integration gate vs stock), all-blocks-gather (within-path
  ceiling), SelectMax top-k in {4,8,16}, random top-8.
Primary delta = vs all-blocks-gather (isolates selection; bf16 MoE amplification
is common-mode and cancels). Secondary delta = vs stock.
"""
import torch, math, json, time, statistics
from patch_qwen import load_model, patch_attention, unpatch, find_attn_modules
from qwen_blocksparse import SelectAll, SelectMax, SelectRandom

CORPUS = "eval_corpus_raw.txt"
BS = 128
CTX = 8192
NSEQ = 6
OUT = "qwen_poc_ppl.json"


def main():
    tok, model = load_model()
    n = len(find_attn_modules(model))
    assert n == 10, f"expected 10 full-attention modules, got {n}"
    toks = tok(open(CORPUS).read()[5000:], return_tensors="pt").input_ids[0]
    wins = [toks[i * CTX:(i + 1) * CTX] for i in range(NSEQ)]
    wins = [w for w in wins if len(w) == CTX]
    ids_all = torch.stack(wins).cuda()
    print(f"ctx={CTX} nseq={len(wins)} (corpus {len(toks)} tokens)", flush=True)

    @torch.no_grad()
    def ppl():
        losses = []
        for i in range(ids_all.shape[0]):
            ids = ids_all[i:i + 1]
            logits = model(ids, use_cache=False).logits[:, :-1].float()
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1))
            losses.append(loss.item())
        return math.exp(statistics.mean(losses))

    def run(name, factory):
        if factory is not None:
            patch_attention(model, factory)
        t = time.time()
        p = ppl()
        if factory is not None:
            unpatch(model)
        print(f"{name:12s} ppl={p:.4f}  ({time.time()-t:.0f}s)", flush=True)
        return p

    res = {"ctx": CTX, "nseq": len(wins), "bs": BS}
    res["stock"] = run("stock", None)
    res["fastpath"] = run("fastpath", lambda li: (None, 10 ** 9, BS))
    rel = abs(res["fastpath"] - res["stock"]) / res["stock"]
    res["integration_rel"] = rel
    print(f"INTEGRATION GATE: {'PASS' if rel < 0.01 else 'FAIL'} (fastpath rel {rel:.4%})", flush=True)
    res["allgather"] = run("allgather", lambda li: (SelectAll(BS), 0, BS))
    for k in (4, 8, 16):
        sm = SelectMax(k, BS)
        res[f"max{k}"] = run(f"max{k}", (lambda li, s=sm, kk=k: (s, kk, BS)))
    sr = SelectRandom(8, BS)
    res["rand8"] = run("rand8", lambda li: (sr, 8, BS))

    base, stock = res["allgather"], res["stock"]
    print("\n=== deltas (ppl, vs all-blocks-gather, vs stock) ===", flush=True)
    for k in ("max4", "max8", "max16", "rand8"):
        dv = 100 * (res[k] - base) / base
        ds = 100 * (res[k] - stock) / stock
        print(f"{k:8s} {res[k]:.4f}   {dv:+.2f}% vs gather   {ds:+.2f}% vs stock", flush=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
