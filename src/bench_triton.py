"""Benchmark: dense flash vs block-sparse (PyTorch gather) vs block-sparse (streaming) vs block-sparse (Triton fused).

Measures latency + peak memory at multiple context lengths. The headline target:
Triton block-sparse memory below dense flash, with latency crossover at long N.
"""
import argparse, time, csv, torch
import torch.nn.functional as F
from triton_kernel import bsattn_forward

dev = "cuda"


def make_sel(B, nblk, kk, device):
    """Causal selection: own block + (kk-1) random past blocks."""
    own = torch.arange(nblk, device=device).view(1, nblk, 1).expand(B, nblk, 1)
    rnd = torch.randint(0, nblk, (B, nblk, kk - 1), device=device)
    return torch.cat([own, rnd], dim=-1).to(torch.int32)


def bench_dense(q, k, v, iters=10):
    B, H, L, dh = q.shape
    # causal dense via SDPA
    with torch.no_grad():
        for _ in range(3): F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(iters): F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters * 1000
    mem = torch.cuda.max_memory_allocated() / 1e9
    return dt, mem


def bench_sparse_pytorch(q, k, v, sel, iters=10):
    B, H, L, dh = q.shape
    Bs = L // sel.shape[1]
    kk = sel.shape[2]
    nblk = sel.shape[1]
    with torch.no_grad():
        for _ in range(3): _sparse_pytorch(q, k, v, sel, Bs, kk, nblk)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(iters): _sparse_pytorch(q, k, v, sel, Bs, kk, nblk)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters * 1000
    mem = torch.cuda.max_memory_allocated() / 1e9
    return dt, mem


def _sparse_pytorch(q, k, v, sel, Bs, kk, nblk):
    B, H, L, dh = q.shape
    bi = torch.arange(B, device=q.device).view(B,1,1,1).expand(B,H,nblk,kk)
    hi = torch.arange(H, device=q.device).view(1,H,1,1).expand(B,H,nblk,kk)
    si = sel.view(B,1,nblk,kk).expand(B,H,nblk,kk)
    kB = k.view(B,H,nblk,Bs,dh); vB = v.view(B,H,nblk,Bs,dh)
    k_sel = kB[bi,hi,si].reshape(B,H,nblk,kk*Bs,dh)
    v_sel = vB[bi,hi,si].reshape(B,H,nblk,kk*Bs,dh)
    qB = q.view(B,H,nblk,Bs,dh)
    mask = torch.zeros(B,H,nblk,Bs,kk*Bs,device=q.device)
    tri = torch.triu(torch.ones(Bs,Bs,device=q.device,dtype=torch.bool),1)
    mask[:,:,:,:,0:Bs] = mask[:,:,:,:,0:Bs].masked_fill_(tri.view(1,1,1,Bs,Bs), float("-inf"))
    attn_mask = mask.reshape(B*H*nblk, Bs, kk*Bs)
    return F.scaled_dot_product_attention(
        qB.reshape(B*H*nblk, Bs, dh),
        k_sel.reshape(B*H*nblk, kk*Bs, dh),
        v_sel.reshape(B*H*nblk, kk*Bs, dh),
        attn_mask=attn_mask
    )


def bench_sparse_triton(q, k, v, sel, gate_bias=None, iters=10):
    with torch.no_grad():
        for _ in range(3): bsattn_forward(q, k, v, sel, gate_bias=gate_bias)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(iters): bsattn_forward(q, k, v, sel, gate_bias=gate_bias)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters * 1000
    mem = torch.cuda.max_memory_allocated() / 1e9
    return dt, mem


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--D", type=int, default=512)
    p.add_argument("--H", type=int, default=8)
    p.add_argument("--B", type=int, default=1)
    p.add_argument("--block", type=int, default=128)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--lengths", default="4096,8192,16384,32768,65536,131072,262144")
    p.add_argument("--out", default="bench_triton.csv")
    a = p.parse_args()

    dh = a.D // a.H
    Ls = [int(x) for x in a.lengths.split(",")]

    hdr = f"{'L':>8} {'dense_ms':>10} {'sparse_pt_ms':>12} {'triton_ms':>10} {'dense_GB':>9} {'sparse_pt_GB':>12} {'triton_GB':>10}"
    print(hdr)
    rows = [("L", "dense_ms", "sparse_pt_ms", "triton_ms", "dense_GB", "sparse_pt_GB", "triton_GB")]
    for L in Ls:
        Bs = a.block
        nblk = L // Bs
        kk = a.topk
        q = torch.randn(a.B, a.H, L, dh, device=dev, dtype=torch.bfloat16)
        k = torch.randn(a.B, a.H, L, dh, device=dev, dtype=torch.bfloat16)
        v = torch.randn(a.B, a.H, L, dh, device=dev, dtype=torch.bfloat16)
        sel = make_sel(a.B, nblk, kk, dev)
        gate_bias = torch.nn.functional.logsigmoid(
            torch.randn(a.B, nblk, kk, device=dev, dtype=torch.float32))

        results = {}
        for name, fn in [("dense", lambda: bench_dense(q, k, v)),
                         ("sparse_pt", lambda: bench_sparse_pytorch(q, k, v, sel)),
                         ("triton", lambda: bench_sparse_triton(q, k, v, sel, gate_bias=gate_bias))]:
            try:
                dt, mem = fn()
            except Exception as e:
                dt, mem = float("nan"), float("nan")
                print(f"  {name} err at {L}: {str(e)[:80]}")
            results[name] = (dt, mem)
            del fn
        torch.cuda.empty_cache()

        d_ms, d_gb = results["dense"]
        s_ms, s_gb = results["sparse_pt"]
        t_ms, t_gb = results["triton"]
        print(f"{L:>8} {d_ms:>10.2f} {s_ms:>12.2f} {t_ms:>10.2f} {d_gb:>9.2f} {s_gb:>12.2f} {t_gb:>10.2f}", flush=True)
        rows.append((L, round(d_ms,3), round(s_ms,3), round(t_ms,3),
                     round(d_gb,3), round(s_gb,3), round(t_gb,3)))

        del q, k, v, sel
        torch.cuda.empty_cache()

    csv.writer(open(a.out, "w")).writerows(rows)
    print("wrote", a.out)
