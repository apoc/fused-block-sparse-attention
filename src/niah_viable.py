"""Clean iso-compute layer-adaptive test: viable floor + surplus to critical layers,
with the uniform floor sweep to locate the viability knee.

Measures uniform top-{9,10,11,12,13,16} alongside the iso-compute mixed configs
(mixed_17: 3*17+7*10=121, mixed_20: 3*20+7*9=123) on disjoint seed-1 needles.

Key question: is uniform top-12 the viability knee (any reduction below ~12
collapses)? If so, no iso-compute concentration can keep all layers viable, and
"uniform sits at the knee, no headroom to redistribute" is the conclusion.
If the knee is lower (e.g. top-10 viable), mixed_17 can cleanly test whether
rank-guided surplus allocation helps.
"""
import argparse, json, random
import torch
from patch_qwen import load_model, patch_attention, unpatch, find_attn_modules
from qwen_blocksparse import SelectMax
from run_followups import build_niah, niah_hit, KEYS, CORPUS, BS


def get_critical(perlayer_path, ctx_ref=32768, n=3):
    d = json.load(open(perlayer_path))
    layers = d[str(ctx_ref)]
    return [int(li) for li in sorted(layers, key=lambda li: layers[li]["pool_med"])[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctxs", default="16384,32768,65536")
    ap.add_argument("--depths", default="0.1,0.3,0.5,0.7,0.9")
    ap.add_argument("--needles_per_depth", type=int, default=2)
    ap.add_argument("--eval_seed", type=int, default=1)
    ap.add_argument("--perlayer", default="qwen_niah_perlayer.json")
    ap.add_argument("--ncrit", type=int, default=3)
    a = ap.parse_args()
    ctxs = [int(x) for x in a.ctxs.split(",")]
    depths = [float(x) for x in a.depths.split(",")]

    crit = get_critical(a.perlayer, 32768, a.ncrit)
    print(f"critical layers: {crit}", flush=True)

    tok, model = load_model()
    assert len(find_attn_modules(model)) == 10
    filler = tok(open(CORPUS).read()[5000:], add_special_tokens=False).input_ids

    rng = random.Random(a.eval_seed)
    samples = [(d, rng.choice(KEYS), rng.randint(1000000, 9999999))
               for d in depths for _ in range(a.needles_per_depth)]

    uniform_ks = [9, 10, 11, 12, 13, 16]
    # iso-compute mixed: critical get surplus, rest get viable floor
    mixed_cfgs = {
        "mixed_17": (17, 10),   # 3*17 + 7*10 = 121 ~ top-12 (120)
        "mixed_20": (20, 9),    # 3*20 + 7*9  = 123 ~ top-12 (120)
    }

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
        for k in uniform_ks:
            cell[f"top{k}"] = ev(f"top{k}", (lambda li, kk=k: (SelectMax(kk, BS), kk, BS)))
        for mname, (hi, lo) in mixed_cfgs.items():
            cell[mname] = ev(
                mname,
                (lambda li, c=set(crit), H=hi, L=lo:
                 (SelectMax(H if li in c else L, BS), H if li in c else L, BS)))
            comp = a.ncrit * hi + (10 - a.ncrit) * lo
            print(f"    ({mname}: {a.ncrit} layers top-{hi} + {10 - a.ncrit} top-{lo} = {comp})",
                  flush=True)
        out[ctx] = cell
        json.dump(out, open("qwen_niah_viable.json", "w"), indent=2)
        print(f"  wrote qwen_niah_viable.json", flush=True)


if __name__ == "__main__":
    main()
