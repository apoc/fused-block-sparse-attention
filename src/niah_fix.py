"""Causal test for the needle-miss diagnostic: does replacing SelectMax's query
block-mean with a max-over-query-block score actually restore NIAH retrieval, or do
ranks improve while retrieval does not?

The diagnostic showed the mean-pooled query ranks the needle ~13% of blocks deep
while the lone retrieval token ranks it ~8. But a selected needle (rank 2 in some
head) still failed to retrieve at 32k, so rank does not cleanly predict retrieval.
This runs the intervention head-to-head on the SAME needles as run_followups: stock,
SelectMax (mean query), SelectMaxQMax (max-over-query-block) at matched top-k.

  * qmax >> max on NIAH  -> query pooling was causally limiting retrieval (fixable).
  * qmax == max on NIAH  -> the bottleneck is propagation, not selection; the rank
                            diagnostic, while real, does not explain the floor.
"""
import argparse, json, random, statistics
import torch
from patch_qwen import load_model, patch_attention, unpatch, find_attn_modules
from qwen_blocksparse import SelectMax, SelectMaxQMax
from run_followups import build_niah, niah_hit, KEYS, CORPUS, BS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctxs", default="16384,32768,65536")
    ap.add_argument("--topks", default="8,16,32")
    ap.add_argument("--depths", default="0.1,0.5,0.9")
    ap.add_argument("--needles_per_depth", type=int, default=2)
    a = ap.parse_args()
    ctxs = [int(x) for x in a.ctxs.split(",")]
    topks = [int(x) for x in a.topks.split(",")]
    depths = [float(x) for x in a.depths.split(",")]

    tok, model = load_model()
    assert len(find_attn_modules(model)) == 10
    filler = tok(open(CORPUS).read()[5000:], add_special_tokens=False).input_ids

    rng = random.Random(0)
    samples = []
    for d in depths:
        for _ in range(a.needles_per_depth):
            samples.append((d, rng.choice(KEYS), rng.randint(1000000, 9999999)))

    out = {}
    for ctx in ctxs:
        built = [build_niah(tok, filler, ctx, d, k, v) for (d, k, v) in samples]
        print(f"\n##### NIAH-fix ctx={ctx} ({len(built)} needles) #####", flush=True)

        def ev(name, factory):
            if factory is not None:
                patch_attention(model, factory)
            hits = [niah_hit(model, ids, nans)[0] for ids, nans in built]
            if factory is not None:
                unpatch(model)
            acc = sum(hits) / len(hits)
            print(f"  {name:10s} {acc:.2f} ({sum(hits)}/{len(hits)})", flush=True)
            return acc

        cell = {"stock": ev("stock", None)}
        for k in topks:
            cell[f"max{k}"] = ev(f"max{k}", (lambda li, kk=k: (SelectMax(kk, BS), kk, BS)))
            cell[f"qmax{k}"] = ev(f"qmax{k}", (lambda li, kk=k: (SelectMaxQMax(kk, BS), kk, BS)))
        out[str(ctx)] = cell
        json.dump(out, open("qwen_niah_fix.json", "w"), indent=2)
        print(f"  --- max vs qmax (exact-match) ---", flush=True)
        for k in topks:
            print(f"  top{k}: max {cell[f'max{k}']:.2f}  qmax {cell[f'qmax{k}']:.2f}", flush=True)
        print(f"  wrote qwen_niah_fix.json", flush=True)


if __name__ == "__main__":
    main()
