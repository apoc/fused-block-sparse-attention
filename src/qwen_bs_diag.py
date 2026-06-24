"""Isolate fastpath vs gather (all-blocks) at the real Qwen attention shapes,
no model load. They must match (both are full causal attention)."""
import torch
from qwen_blocksparse import blocksparse_forward, SelectAll

torch.manual_seed(0)
B, Hq, Hkv, L, d, bs = 1, 16, 2, 2048, 256, 128
for dt in (torch.float32, torch.bfloat16):
    q = torch.randn(B, Hq, L, d, device="cuda", dtype=dt)
    k = torch.randn(B, Hkv, L, d, device="cuda", dtype=dt)
    v = torch.randn(B, Hkv, L, d, device="cuda", dtype=dt)
    fast = blocksparse_forward(q, k, v, selector=None, topk=10**9, bs=bs).float()
    gath = blocksparse_forward(q, k, v, selector=SelectAll(bs=bs), topk=0, bs=bs).float()
    dd = (fast - gath).abs()
    print(f"{dt}: maxabs={dd.max().item():.4e}  meanabs={dd.mean().item():.4e}  "
          f"rel={ (dd.max()/fast.abs().max()).item():.4e}")
