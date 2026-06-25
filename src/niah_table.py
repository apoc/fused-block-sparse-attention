"""Authoritative NIAH table for paper Section 8: mean-query (SelectMax) vs last-token
(SelectLastTok) at matched budget, plus random, at n=20 needles/cell (4/depth x 5
depths) on a single seed. Replaces the noise-prone n=6 numbers in Table 10.
"""
import argparse, json, random
import torch
from patch_qwen import load_model, patch_attention, unpatch, find_attn_modules
from qwen_blocksparse import SelectMax, SelectLastTok, SelectRandom
from run_followups import build_niah, niah_hit, KEYS, CORPUS, BS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctxs", default="16384,32768,65536")
    ap.add_argument("--topks", default="8,16,32")
    ap.add_argument("--rand", type=int, default=16)
    ap.add_argument("--depths", default="0.1,0.3,0.5,0.7,0.9")
    ap.add_argument("--needles_per_depth", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    ctxs = [int(x) for x in a.ctxs.split(",")]
    topks = [int(x) for x in a.topks.split(",")]
    depths = [float(x) for x in a.depths.split(",")]

    tok, model = load_model()
    assert len(find_attn_modules(model)) == 10
    filler = tok(open(CORPUS).read()[5000:], add_special_tokens=False).input_ids

    rng = random.Random(a.seed)
    samples = [(d, rng.choice(KEYS), rng.randint(1000000, 9999999))
               for d in depths for _ in range(a.needles_per_depth)]

    out = {"seed": a.seed, "n_needles": len(samples)}
    for ctx in ctxs:
        built = [build_niah(tok, filler, ctx, d, k, v) for (d, k, v) in samples]
        print(f"\n##### ctx={ctx} ({len(built)} needles, seed={a.seed}) #####", flush=True)

        def ev(name, factory):
            if factory is not None:
                patch_attention(model, factory)
            hits = [niah_hit(model, ids, nans)[0] for ids, nans in built]
            if factory is not None:
                unpatch(model)
            acc = sum(hits) / len(hits)
            print(f"  {name:10s} {acc:.3f} ({sum(hits)}/{len(hits)})", flush=True)
            return acc

        cell = {"stock": ev("stock", None)}
        for k in topks:
            cell[f"max{k}"] = ev(f"max{k}", (lambda li, kk=k: (SelectMax(kk, BS), kk, BS)))
            cell[f"lt{k}"] = ev(f"lt{k}", (lambda li, kk=k: (SelectLastTok(kk, BS), kk, BS)))
        cell[f"rand{a.rand}"] = ev(f"rand{a.rand}", lambda li: (SelectRandom(a.rand, BS), a.rand, BS))
        out[ctx] = cell
        json.dump(out, open("qwen_niah_table.json", "w"), indent=2)
        print(f"  --- max / last-tok ---", flush=True)
        for k in topks:
            print(f"  top{k}: {cell[f'max{k}']:.2f} / {cell[f'lt{k}']:.2f}", flush=True)
        print(f"  wrote qwen_niah_table.json", flush=True)


if __name__ == "__main__":
    main()
