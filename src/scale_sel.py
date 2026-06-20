"""Selection-cost scaling proof: block-pair O(nblk^2) vs centroid O(nblk*C).

Isolates the SELECTION step (not attention) and measures latency + peak memory
as the number of blocks grows. Block-pair scoring forms an (nblk x nblk) matrix
=> quadratic; centroid routing never does => linear. The gap should grow with N.
"""
import torch, time, argparse
import torch.nn.functional as F

dev = "cuda" if torch.cuda.is_available() else "cpu"
s, C, topk, cpq, cap, B = 32, 16, 8, 2, 4, 1

def bench(fn, iters=20):
    for _ in range(3): fn()
    if dev == "cuda": torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(iters): fn()
    if dev == "cuda": torch.cuda.synchronize()
    ms = (time.time() - t0) / iters * 1000
    mem = torch.cuda.max_memory_allocated() / 1e6 if dev == "cuda" else 0.0
    if dev == "cuda": torch.cuda.empty_cache()
    return ms, mem

nblks = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
print(f"{'nblk':>7} {'blkpair_ms':>11} {'blkpair_MB':>11} {'cent_ms':>9} {'cent_MB':>9} {'speedup':>8} {'mem_x':>7}")
rows = []
for nblk in nblks:
    sqb = torch.randn(B, nblk, s, device=dev)
    skb = torch.randn(B, nblk, s, device=dev)
    cent = torch.randn(C, s, device=dev)
    eye = torch.eye(nblk, device=dev).unsqueeze(0)

    def blockpair():
        raw = sqb @ skb.transpose(-1, -2)          # (B,nblk,nblk)  O(nblk^2)
        return (raw + eye * 1e4).topk(topk, dim=-1).indices

    def centroid():
        kc = skb @ cent.t()                         # (B,nblk,C)
        qc = sqb @ cent.t()
        key_cent = kc.argmax(-1)
        top_c = qc.topk(cpq, -1).indices
        oh = F.one_hot(key_cent, C)
        within = (oh.cumsum(1) - oh).gather(2, key_cent.unsqueeze(-1)).squeeze(-1)
        keep = within < cap
        buckets = torch.full((B, C, cap), -1, dtype=torch.long, device=dev)
        bI = torch.arange(B, device=dev).view(B, 1).expand(B, nblk)
        jI = torch.arange(nblk, device=dev).view(1, nblk).expand(B, nblk)
        buckets[bI[keep], key_cent[keep], within[keep]] = jI[keep]
        bexp = buckets.unsqueeze(1).expand(B, nblk, C, cap)
        return torch.gather(bexp, 2, top_c.unsqueeze(-1).expand(B, nblk, cpq, cap))

    try:
        bm, bmem = bench(blockpair)
    except RuntimeError as e:
        bm, bmem = float("nan"), float("nan"); print("blockpair OOM at", nblk, str(e)[:40])
    cm, cmem = bench(centroid)
    sp = bm / cm if cm > 0 else float("nan")
    mx = bmem / cmem if cmem > 0 else float("nan")
    print(f"{nblk:>7} {bm:>11.3f} {bmem:>11.1f} {cm:>9.3f} {cmem:>9.1f} {sp:>7.1f}x {mx:>6.1f}x")
    rows.append((nblk, round(bm, 4), round(bmem, 1), round(cm, 4), round(cmem, 1)))

import csv
csv.writer(open("scale_sel.csv", "w")).writerows(
    [("nblk", "blkpair_ms", "blkpair_MB", "cent_ms", "cent_MB")] + rows)
print("wrote scale_sel.csv")
