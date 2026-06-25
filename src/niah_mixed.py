"""Layer-adaptive NIAH: does concentrating the block budget on retrieval-critical
layers preserve retrieval at lower total compute than a uniform budget?

Critical layers are identified by per-layer needle rank (qwen_niah_perlayer.json,
seed-0 needles). Evaluation uses DISJOINT seed-1 needles so the critical-layer choice
cannot overfit to the eval set. Two iso-compute mixed configs (both sum to 120 total
blocks, matching uniform top-12):
  mixed_32: critical top-32, rest top-4   -> 3*32 + 7*4 = 124 (report raw)
  mixed_31: critical top-31, rest top-4   -> 3*31 + 7*4 = 121
  mixed_26: critical top-26, rest top-6   -> 3*26 + 7*6 = 120  (exact iso-compute)
If retrieval is layer-concentrated, mixed_26 beats uniform top-12 at equal compute.
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
    ap.add_argument("--topks", default="4,8,12,16,32")
    ap.add_argument("--depths", default="0.1,0.3,0.5,0.7,0.9")
    ap.add_argument("--needles_per_depth", type=int, default=2)
    ap.add_argument("--eval_seed", type=int, default=1)
    ap.add_argument("--perlayer", default="qwen_niah_perlayer.json")
    ap.add_argument("--ncrit", type=int, default=3)
    a = ap.parse_args()
    ctxs = [int(x) for x in a.ctxs.split(",")]
    topks = [int(x) for x in a.topks.split(",")]
    depths = [float(x) for x in a.depths.split(",")]

    crit = get_critical(a.perlayer, 32768, a.ncrit)
    print(f"critical layers (lowest pool-med at 32k): {crit}", flush=True)

    tok, model = load_model()
    assert len(find_attn_modules(model)) == 10
    filler = tok(open(CORPUS).read()[5000:], add_special_tokens=False).input_ids

    # DISJOINT eval needles (seed 1, not the seed-0 diagnostic needles)
    rng = random.Random(a.eval_seed)
    samples = [(d, rng.choice(KEYS), rng.randint(1000000, 9999999))
               for d in depths for _ in range(a.needles_per_depth)]

    # iso-compute mixed configs: all sum to 120 total blocks = uniform top-12
    mixed_cfgs = {
        "mixed_32": (32, 4),   # 3*32 + 7*4 = 124 (raw, slightly over)
        "mixed_26": (26, 6),   # 3*26 + 7*6 = 120 (exact iso-compute with top-12)
        "mixed_31": (31, 4),   # 3*31 + 7*4 = 121
    }

    out = {"critical_layers": crit, "eval_seed": a.eval_seed, "n_needles": len(samples)}
    for ctx in ctxs:
        built = [build_niah(tok, filler, ctx, d, k, v) for (d, k, v) in samples]
        print(f"\n##### ctx={ctx} ({len(built)} needles, seed={a.eval_seed}) #####", flush=True)

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
        for k in topks:
            cell[f"top{k}"] = ev(f"top{k}", (lambda li, kk=k: (SelectMax(kk, BS), kk, BS)))
        for mname, (hi, lo) in mixed_cfgs.items():
            cell[mname] = ev(mname, (lambda li, c=set(crit), H=hi, L=lo:
                                     (SelectMax(H if li in c else L, BS),
                                      H if li in c else L, BS)))
            comp = a.ncrit * hi + (10 - a.ncrit) * lo
            print(f"    ({mname}: {a.ncrit} layers top-{hi} + {10 - a.ncrit} top-{lo} = {comp} blocks)",
                  flush=True)
        out[ctx] = cell
        json.dump(out, open("qwen_niah_mixed.json", "w"), indent=2)
        print(f"  wrote qwen_niah_mixed.json", flush=True)


if __name__ == "__main__":
    main()
