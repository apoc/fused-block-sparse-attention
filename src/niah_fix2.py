"""NIAH query-reduction shootout: do last-token or two-stage selection close the
retrieval floor the diagnostic blamed on query block-mean pooling?

Same needles as run_followups. Compares, at matched top-k:
  * max      : SelectMax (block-mean query)        -- baseline
  * lasttok  : SelectLastTok (block's last token)  -- idea #3
  * twostage : SelectTwoStage (mean recall -> qmax precision) -- idea #1

Success target: 16k top-16 reaching ~0.82+ (exact-query coverage) would confirm the
query-reduction story and largely close the floor.
"""
import argparse, json, random
import torch
from patch_qwen import load_model, patch_attention, unpatch, find_attn_modules
from qwen_blocksparse import SelectMax, SelectLastTok, SelectTwoStage
from run_followups import build_niah, niah_hit, KEYS, CORPUS, BS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctxs", default="16384,32768,65536")
    ap.add_argument("--topks", default="8,16,32")
    ap.add_argument("--depths", default="0.1,0.5,0.9")
    ap.add_argument("--needles_per_depth", type=int, default=2)
    ap.add_argument("--over", type=int, default=4)
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

    sels = {
        "max": lambda kk: SelectMax(kk, BS),
        "lasttok": lambda kk: SelectLastTok(kk, BS),
        "twostage": lambda kk: SelectTwoStage(kk, BS, over=a.over),
    }

    out = {}
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
            print(f"  {name:16s} {acc:.2f} ({sum(hits)}/{len(hits)})", flush=True)
            return acc

        cell = {"stock": ev("stock", None)}
        for name, mk in sels.items():
            for kk in topks:
                cell[f"{name}{kk}"] = ev(f"{name}{kk}", (lambda li, m=mk, k=kk: (m(k), k, BS)))
        out[str(ctx)] = cell
        json.dump(out, open("qwen_niah_fix2.json", "w"), indent=2)
        print(f"  --- by budget (max / lasttok / twostage) ---", flush=True)
        for kk in topks:
            print(f"  top{kk}: {cell[f'max{kk}']:.2f} / {cell[f'lasttok{kk}']:.2f} / {cell[f'twostage{kk}']:.2f}",
                  flush=True)
        print(f"  wrote qwen_niah_fix2.json", flush=True)


if __name__ == "__main__":
    main()
