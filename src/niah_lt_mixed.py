"""Quick A/B: does last-token selection improve NIAH retrieval over mean-query
SelectMax, on the SAME needles, at matched budget? Plus a layer-adaptive bonus.

Paired uniform comparison at top-{8,16,32}: SelectMax (mean-query) vs SelectLastTok
(last token of each query block), identical needles, at 32k and 64k. Then two
iso-compute layer-adaptive last-token configs (critical layers by per-layer exact
rank get more budget) to check if concentration helps on top of last-token.

Eval needles: seed 1 (disjoint from the seed-0 per-layer diagnostic), 3/depth x 5.
"""
import argparse, json, random
import torch
from patch_qwen import load_model, patch_attention, unpatch, find_attn_modules
from qwen_blocksparse import SelectMax, SelectLastTok
from run_followups import build_niah, niah_hit, KEYS, CORPUS, BS


def get_critical(path, ctx_ref=32768, n=3, key="exact_med"):
    d = json.load(open(path))
    layers = d[str(ctx_ref)]
    return [int(li) for li in sorted(layers, key=lambda li: layers[li][key])[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctxs", default="32768,65536")
    ap.add_argument("--depths", default="0.1,0.3,0.5,0.7,0.9")
    ap.add_argument("--needles_per_depth", type=int, default=3)
    ap.add_argument("--eval_seed", type=int, default=1)
    ap.add_argument("--perlayer", default="qwen_niah_perlayer.json")
    ap.add_argument("--ncrit", type=int, default=3)
    a = ap.parse_args()
    ctxs = [int(x) for x in a.ctxs.split(",")]
    depths = [float(x) for x in a.depths.split(",")]

    crit = get_critical(a.perlayer, 32768, a.ncrit, "exact_med")
    print(f"critical layers (lowest exact-med at 32k): {crit}", flush=True)

    tok, model = load_model()
    assert len(find_attn_modules(model)) == 10
    filler = tok(open(CORPUS).read()[5000:], add_special_tokens=False).input_ids

    rng = random.Random(a.eval_seed)
    samples = [(d, rng.choice(KEYS), rng.randint(1000000, 9999999))
               for d in depths for _ in range(a.needles_per_depth)]

    uniform_ks = [8, 16, 32]
    mixed_cfgs = {"lt_m16": (32, 9), "lt_m32": (64, 18)}   # iso ~ top-16 / top-32

    out = {"critical_layers": crit, "eval_seed": a.eval_seed, "n_needles": len(samples)}
    for ctx in ctxs:
        built = [build_niah(tok, filler, ctx, d, k, v) for (d, k, v) in samples]
        print(f"\n##### ctx={ctx} ({len(built)} needles) #####", flush=True)

        def ev(name, factory):
            if factory is not None:
                patch_attention(model, factory)
            hits = [niah_hit(model, ids, nans)[0] for ids, nans in built]
            if factory is not None:
                unpatch(model)
            acc = sum(hits) / len(hits)
            print(f"  {name:14s} {acc:.2f} ({sum(hits)}/{len(hits)})", flush=True)
            return acc

        cell = {"stock": ev("stock", None)}
        for k in uniform_ks:                       # paired mean-query vs last-token
            cell[f"max{k}"] = ev(f"max{k}", (lambda li, kk=k: (SelectMax(kk, BS), kk, BS)))
            cell[f"lt{k}"] = ev(f"lt{k}", (lambda li, kk=k: (SelectLastTok(kk, BS), kk, BS)))
        for mname, (hi, lo) in mixed_cfgs.items():  # last-token + layer-adaptive
            cell[mname] = ev(
                mname,
                (lambda li, c=set(crit), H=hi, L=lo:
                 (SelectLastTok(H if li in c else L, BS), H if li in c else L, BS)))
        out[ctx] = cell
        json.dump(out, open("qwen_niah_lt_mixed.json", "w"), indent=2)
        print(f"  --- max vs last-tok (uniform) ---", flush=True)
        for k in uniform_ks:
            print(f"  top{k}: max {cell[f'max{k}']:.2f}  lt {cell[f'lt{k}']:.2f}", flush=True)
        print(f"  wrote qwen_niah_lt_mixed.json", flush=True)


if __name__ == "__main__":
    main()
