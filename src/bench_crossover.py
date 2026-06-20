"""Crossover benchmark: isolate SELECTION (O(nblk^2)) from ATTENTION READ (O(n)).

Our BlockSparseSSA attention READ is linear in N (each query attends to topk+1
blocks). The SELECTION step (_scores: score every query-block against every
key-block, then top-k) is O(nblk^2) = O(N^2 / B^2) -- quadratic, with a 1/B^2
constant. This script measures the two stages SEPARATELY across context lengths
and block sizes so we can see:

  1. the scaling SHAPE of each stage (selection should ~4x per doubling of N;
     attention read should ~2x), proving selection is quadratic, not linear;
  2. where selection LATENCY overtakes the attention read (the crossover);
  3. where the O(nblk^2) score matrix OOMs (memory blowup of quadratic selection).

This directly tests the claim that "looks linear at the measured N" can hide a
quadratic selector: with a large block the crossover is pushed far out; with a
small block it appears (or OOMs) within reach.
"""
import argparse, time, csv, math, torch
import torch.nn.functional as F
from ssa_model import BlockSparseSSA
from triton_kernel import bsattn_forward

dev = "cuda"
OOM_ERRORS = (torch.cuda.OutOfMemoryError, RuntimeError)


def _time(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters * 1000.0
    mem = torch.cuda.max_memory_allocated() / 1e9
    return dt, mem


def do_select(model, x, B, nblk, Bs, topk, causal):
    """The O(nblk^2) selection: produces sel (B,nblk,kk). Mirrors BlockSparseSSA.forward."""
    raw = model._scores(x, B, nblk, Bs)                      # (B,nblk,nblk) -- quadratic matmul
    diag = torch.arange(nblk, device=x.device)
    content = raw.clone()
    content[:, diag, diag] = float("-inf")                   # exclude own block
    if causal:
        fut = diag.view(nblk, 1) < diag.view(1, nblk)        # (nblk,nblk) bool -- quadratic mem
        content = content.masked_fill(fut.unsqueeze(0), float("-inf"))
    kk_c = min(topk, nblk - 1)
    own = diag.view(1, nblk, 1).expand(B, nblk, 1)
    sel_c = content.topk(kk_c, dim=-1).indices               # (B,nblk,kk_c) -- O(nblk^2) scan
    sel = torch.cat([own, sel_c], dim=-1)
    return sel, raw


def build_qkv(B, L, H, dh):
    # random q/k/v: the attention-read cost depends only on shapes, not values,
    # and skipping the qkv projection avoids its O(N) intermediate at 8M/16M.
    q = torch.randn(B, H, L, dh, device=dev, dtype=torch.bfloat16)
    k = torch.randn(B, H, L, dh, device=dev, dtype=torch.bfloat16)
    v = torch.randn(B, H, L, dh, device=dev, dtype=torch.bfloat16)
    return q, k, v


def make_sel(B, nblk, kk, device):
    """Fabricated causal selection (own block + kk-1 random past): lets the linear
    attention read be timed even where the quadratic real selection OOMs. The read
    cost depends only on the count kk, not which blocks are chosen."""
    own = torch.arange(nblk, device=device).view(1, nblk, 1).expand(B, nblk, 1)
    rnd = torch.randint(0, nblk, (B, nblk, kk - 1), device=device)
    return torch.cat([own, rnd], dim=-1).to(torch.int32)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--D", type=int, default=512)
    p.add_argument("--H", type=int, default=8)
    p.add_argument("--B", type=int, default=1)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--sel_dim", type=int, default=32)
    p.add_argument("--blocks", default="32,64,128")
    p.add_argument("--lengths", default="262144,524288,1048576,4194304")
    p.add_argument("--out", default="bench_crossover.csv")
    a = p.parse_args()

    dh = a.D // a.H
    Ls = [int(x) for x in a.lengths.split(",")]
    blocks = [int(x) for x in a.blocks.split(",")]

    hdr = (f"{'block':>6} {'L':>9} {'nblk':>8} {'sel_ms':>9} {'attn_ms':>9} "
           f"{'sel_GB':>8} {'attn_GB':>8} {'sel_PF':>9} {'attn_PF':>9}")
    print(hdr, flush=True)
    rows = [("block", "L", "nblk", "sel_ms", "attn_ms", "sel_GB", "attn_GB",
             "sel_pflop", "attn_pflop")]

    for Bs in blocks:
        for L in Ls:
            if L % Bs:
                continue
            nblk = L // Bs
            kk = a.topk + 1
            # analytical FLOPs (2 = mul+add per MAC; selection does mean & max matmuls -> x2)
            sel_pf = 2 * 2 * a.B * nblk * nblk * a.sel_dim / 1e15
            attn_pf = 2 * 2 * a.B * a.H * L * kk * Bs * dh / 1e15

            iters = 5 if L <= 1_048_576 else 3
            warmup = 2 if L <= 1_048_576 else 1
            sel_ms = attn_ms = sel_gb = attn_gb = float("nan")

            try:
                model = BlockSparseSSA(a.D, a.H, block=Bs, topk=a.topk,
                                       sel_dim=a.sel_dim, causal=True,
                                       use_triton=True).to(dev).to(torch.bfloat16).eval()
                x = torch.randn(a.B, L, a.D, device=dev, dtype=torch.bfloat16)
            except OOM_ERRORS as e:
                print(f"{Bs:>6} {L:>9}  setup OOM: {str(e)[:50]}", flush=True)
                torch.cuda.empty_cache()
                continue

            # --- selection stage ---
            try:
                with torch.no_grad():
                    sel_ms, sel_gb = _time(
                        lambda: do_select(model, x, a.B, nblk, Bs, a.topk, True),
                        iters, warmup)
            except OOM_ERRORS as e:
                print(f"{Bs:>6} {L:>9}  SELECT OOM ({str(e)[:40]})", flush=True)
                torch.cuda.empty_cache()

            # --- attention-read stage ---
            try:
                with torch.no_grad():
                    sel_i32 = make_sel(a.B, nblk, kk, dev)
                    gate_bias = F.logsigmoid(
                        torch.randn(a.B, nblk, kk, device=dev, dtype=torch.float32))
                    q, k, v = build_qkv(a.B, L, a.H, dh)
                    attn_ms, attn_gb = _time(
                        lambda: bsattn_forward(q, k, v, sel_i32, gate_bias, causal=True),
                        iters, warmup)
                    del q, k, v, sel_i32, gate_bias
            except OOM_ERRORS as e:
                print(f"{Bs:>6} {L:>9}  ATTN OOM ({str(e)[:40]})", flush=True)
                torch.cuda.empty_cache()

            print(f"{Bs:>6} {L:>9} {nblk:>8} {sel_ms:>9.3f} {attn_ms:>9.3f} "
                  f"{sel_gb:>8.2f} {attn_gb:>8.2f} {sel_pf:>9.4f} {attn_pf:>9.4f}",
                  flush=True)
            rows.append((Bs, L, nblk, round(sel_ms, 4), round(attn_ms, 4),
                         round(sel_gb, 3), round(attn_gb, 3),
                         round(sel_pf, 6), round(attn_pf, 6)))
            del model, x
            torch.cuda.empty_cache()

    csv.writer(open(a.out, "w")).writerows(rows)
    print("\nwrote", a.out)

    # --- scaling-shape summary: consecutive doubling ratios per block ---
    print("\n=== doubling ratios (x per 2x context): sel should ~4 (quadratic), attn ~2 (linear) ===")
    by_block = {}
    for r in rows[1:]:
        by_block.setdefault(r[0], []).append(r)
    for Bs, rs in by_block.items():
        rs = [r for r in rs if not math.isnan(r[3]) and not math.isnan(r[4])]
        for i in range(1, len(rs)):
            ls, sel0, sel1 = rs[i][1], rs[i-1][3], rs[i][3]
            attn0, attn1 = rs[i-1][4], rs[i][4]
            sr = sel1 / sel0 if sel0 else float("nan")
            ar = attn1 / attn0 if attn0 else float("nan")
            print(f"  block={Bs:>3} {rs[i-1][1]:>9}->{ls:>9}: sel x{sr:.2f}  attn x{ar:.2f}")
