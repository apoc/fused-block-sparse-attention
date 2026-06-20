"""Fusion deliverable: memory-efficient block-sparse attention via online-softmax
STREAMING — gather one selected key-block at a time, accumulate with a running
max/sum, never materialize the (kk*Bs) gathered K/V or score matrix. This is the
exact dataflow a fused (FlashAttention-style) Triton kernel implements; here in
PyTorch to prove the memory win correctly.

Compares peak memory: dense SDPA vs materialized-gather sparse vs streaming sparse.
"""
import argparse, time, torch
import torch.nn.functional as F

dev = "cuda" if torch.cuda.is_available() else "cpu"


def make_sel(B, nblk, kk, device):
    # own block + (kk-1) random distinct other blocks per query block
    own = torch.arange(nblk, device=device).view(1, nblk, 1).expand(B, nblk, 1)
    rnd = torch.randint(0, nblk, (B, nblk, kk - 1), device=device)
    return torch.cat([own, rnd], dim=-1)                      # (B,nblk,kk)


def materialized(q, k, v, sel, Bs):
    B, H, Lp, dh = q.shape
    nblk, kk = sel.shape[1], sel.shape[2]
    kB = k.view(B, H, nblk, Bs, dh); vB = v.view(B, H, nblk, Bs, dh)
    bi = torch.arange(B, device=q.device).view(B, 1, 1, 1).expand(B, H, nblk, kk)
    hi = torch.arange(H, device=q.device).view(1, H, 1, 1).expand(B, H, nblk, kk)
    si = sel.view(B, 1, nblk, kk).expand(B, H, nblk, kk)
    k_sel = kB[bi, hi, si].reshape(B, H, nblk, kk * Bs, dh)
    v_sel = vB[bi, hi, si].reshape(B, H, nblk, kk * Bs, dh)
    qB = q.view(B, H, nblk, Bs, dh)
    o = F.scaled_dot_product_attention(qB.reshape(B * H * nblk, Bs, dh),
                                       k_sel.reshape(B * H * nblk, kk * Bs, dh),
                                       v_sel.reshape(B * H * nblk, kk * Bs, dh))
    return o.reshape(B, H, nblk, Bs, dh)


def streaming(q, k, v, sel, Bs):
    B, H, Lp, dh = q.shape
    nblk, kk = sel.shape[1], sel.shape[2]
    kB = k.view(B, H, nblk, Bs, dh); vB = v.view(B, H, nblk, Bs, dh)
    qB = q.view(B, H, nblk, Bs, dh)
    scale = dh ** -0.5
    neg = torch.finfo(torch.float32).min
    m = torch.full((B, H, nblk, Bs, 1), neg, device=q.device)
    l = torch.zeros((B, H, nblk, Bs, 1), device=q.device)
    acc = torch.zeros((B, H, nblk, Bs, dh), device=q.device)
    for j in range(kk):
        idx = sel[:, :, j].view(B, 1, nblk, 1, 1).expand(B, H, nblk, Bs, dh)
        kj = torch.gather(kB, 2, idx)                          # (B,H,nblk,Bs,dh)
        vj = torch.gather(vB, 2, idx)
        s = (qB.float() @ kj.float().transpose(-1, -2)) * scale   # (B,H,nblk,Bs,Bs)
        mj = s.max(-1, keepdim=True).values
        m_new = torch.maximum(m, mj)
        p = (s - m_new).exp()
        corr = (m - m_new).exp()
        l = l * corr + p.sum(-1, keepdim=True)
        acc = acc * corr + p @ vj.float()
        m = m_new
        del kj, vj, s, p
    return (acc / l).to(q.dtype)


def peak_mem(fn, *args):
    if dev == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        out = fn(*args)
    if dev == "cuda": torch.cuda.synchronize()
    mem = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0.0
    del out
    if dev == "cuda": torch.cuda.empty_cache()
    return mem


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--D", type=int, default=512); p.add_argument("--H", type=int, default=8)
    p.add_argument("--B", type=int, default=1); p.add_argument("--Bs", type=int, default=128)
    p.add_argument("--kk", type=int, default=9)
    p.add_argument("--lengths", default="16384,65536,131072,262144,524288")
    a = p.parse_args()
    dh = a.D // a.H
    dt = torch.bfloat16 if dev == "cuda" else torch.float32

    # correctness: streaming ~= materialized (small)
    torch.manual_seed(0)
    B, H, Lp = 1, 4, 512; bs = 64; nb = Lp // bs; kk = 4
    q = torch.randn(B, H, Lp, 32, device=dev, dtype=torch.float32)
    k = torch.randn_like(q); v = torch.randn_like(q)
    sel = make_sel(B, nb, kk, dev)
    d = (materialized(q, k, v, sel, bs) - streaming(q, k, v, sel, bs)).abs().max().item()
    print("streaming vs materialized max diff:", d, "->", "PASS" if d < 1e-3 else "FAIL")

    print(f"\n{'L':>8} {'dense_GB':>9} {'matsparse_GB':>13} {'stream_GB':>10} {'mat/stream':>11}")
    for L in [int(x) for x in a.lengths.split(",")]:
        nblk = L // a.Bs
        q = torch.randn(a.B, a.H, L, dh, device=dev, dtype=dt)
        k = torch.randn_like(q); v = torch.randn_like(q)
        sel = make_sel(a.B, nblk, a.kk, dev)
        try:
            dgb = peak_mem(lambda: F.scaled_dot_product_attention(q, k, v))
        except RuntimeError:
            dgb = float("nan")
        try:
            mgb = peak_mem(lambda: materialized(q, k, v, sel, a.Bs))
        except RuntimeError:
            mgb = float("nan")
        try:
            sgb = peak_mem(lambda: streaming(q, k, v, sel, a.Bs))
        except RuntimeError:
            sgb = float("nan")
        ratio = mgb / sgb if sgb == sgb and sgb > 0 else float("nan")
        print(f"{L:>8} {dgb:>9.2f} {mgb:>13.2f} {sgb:>10.2f} {ratio:>10.1f}x")
