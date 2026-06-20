"""Extended bench: (1) causal+gate block-sparse vs original non-causal, showing
the mask-memory fix payoff; (2) CentroidSSA end-to-end forward pass vs
BlockSparseSSA vs dense, showing whether linear selection translates to
real wall-clock wins."""
import argparse, time, csv, torch
from ssa_model import DenseAttention, BlockSparseSSA, CentroidSSA

def bench_one(mod, B, L, D, device, iters=10):
    x = torch.randn(B, L, D, device=device, dtype=torch.bfloat16 if device=="cuda" else torch.float32)
    mod = mod.to(device)
    if device == "cuda": mod = mod.to(torch.bfloat16)
    with torch.no_grad():
        for _ in range(3): mod(x)
    if device == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(iters): y = mod(x)
    if device == "cuda": torch.cuda.synchronize()
    dt = (time.time() - t0) / iters * 1000
    mem = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
    del x, y
    if device == "cuda": torch.cuda.empty_cache()
    return dt, mem

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--D", type=int, default=512)
    p.add_argument("--H", type=int, default=8)
    p.add_argument("--B", type=int, default=1)
    p.add_argument("--block", type=int, default=128)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--lengths", default="4096,8192,16384,32768,65536,131072,262144")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="bench2.csv")
    p.add_argument("--n_centroids", type=int, default=16)
    p.add_argument("--cpq", type=int, default=2)
    p.add_argument("--cap", type=int, default=4)
    a = p.parse_args()
    dev = a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu"
    Ls = [int(x) for x in a.lengths.split(",")]

    # Modules: dense-causal, sparse-noncausal (original), sparse-causal-gate (deploy),
    # centroid (linear selection)
    dn_c = DenseAttention(a.D, a.H, causal=True)
    sp_nc = BlockSparseSSA(a.D, a.H, block=a.block, topk=a.topk, sel_dim=32)
    sp_cg = BlockSparseSSA(a.D, a.H, block=a.block, topk=a.topk, sel_dim=32, gate=True, causal=True)
    ct = CentroidSSA(a.D, a.H, block=a.block, sel_dim=32,
                     n_centroids=a.n_centroids, cpq=a.cpq, cap=a.cap, gate=True)

    mods = [("dense_causal", dn_c), ("sparse_noncausal", sp_nc),
            ("sparse_causal_gate", sp_cg), ("centroid_gate", ct)]
    rows = [("L",) + tuple(name for name, _ in mods) + tuple(name+"_GB" for name, _ in mods)]
    hdr = f"{'L':>8}" + "".join(f" {name:>18}" for name, _ in mods) + "".join(f" {name+'_GB':>14}" for name, _ in mods)
    print(hdr)
    for L in Ls:
        times, mems = {}, {}
        for name, mod in mods:
            try:
                t, m = bench_one(mod, a.B, L, a.D, dev)
            except RuntimeError as e:
                t, m = float("nan"), float("nan")
                print(f"  {name} OOM/err at {L}: {str(e)[:60]}")
            times[name] = t; mems[name] = m
        parts = f"{L:>8}" + "".join(f" {times[n]:>18.2f}" for n, _ in mods) + "".join(f" {mems[n]:>14.2f}" for n, _ in mods)
        print(parts, flush=True)
        row = (L,) + tuple(round(times[n],3) for n,_ in mods) + tuple(round(mems[n],3) for n,_ in mods)
        rows.append(row)
    csv.writer(open(a.out, "w")).writerows(rows)
    print("wrote", a.out)
