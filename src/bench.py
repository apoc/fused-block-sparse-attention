"""Forward-only latency + memory scaling: dense (SDPA flash) vs block-sparse.
Shows the crossover where sparse overtakes dense as context grows."""
import argparse, time, csv, torch
from ssa_model import DenseAttention, BlockSparseSSA

def bench_one(mod, B, L, D, device, iters=10):
    x = torch.randn(B, L, D, device=device, dtype=torch.bfloat16 if device=="cuda" else torch.float32)
    mod = mod.to(device)
    if device == "cuda": mod = mod.to(torch.bfloat16)
    # warmup
    with torch.no_grad():
        for _ in range(3): mod(x)
    if device == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(iters): y = mod(x)
    if device == "cuda": torch.cuda.synchronize()
    dt = (time.time() - t0) / iters * 1000  # ms
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
    p.add_argument("--out", default="bench.csv")
    a = p.parse_args()
    dev = a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu"
    Ls = [int(x) for x in a.lengths.split(",")]
    dn = DenseAttention(a.D, a.H)
    sp = BlockSparseSSA(a.D, a.H, block=a.block, topk=a.topk, sel_dim=32)
    rows = [("L", "dense_ms", "dense_GB", "sparse_ms", "sparse_GB", "speedup")]
    print(f"{'L':>8} {'dense_ms':>10} {'sparse_ms':>10} {'speedup':>9} {'dense_GB':>9} {'sparse_GB':>9}")
    for L in Ls:
        try:
            d_ms, d_gb = bench_one(dn, a.B, L, a.D, dev)
        except RuntimeError as e:
            d_ms, d_gb = float("nan"), float("nan"); print("dense OOM/err at", L, str(e)[:60])
        try:
            s_ms, s_gb = bench_one(sp, a.B, L, a.D, dev)
        except RuntimeError as e:
            s_ms, s_gb = float("nan"), float("nan"); print("sparse OOM/err at", L, str(e)[:60])
        sp_up = d_ms / s_ms if s_ms == s_ms and s_ms > 0 else float("nan")
        print(f"{L:>8} {d_ms:>10.2f} {s_ms:>10.2f} {sp_up:>8.1f}x {d_gb:>9.2f} {s_gb:>9.2f}")
        rows.append((L, round(d_ms,3), round(d_gb,3), round(s_ms,3), round(s_gb,3), round(sp_up,3)))
    csv.writer(open(a.out, "w")).writerows(rows)
    print("wrote", a.out)
