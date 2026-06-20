"""Correctness check for the int64/grid-reorder kernel fix.

Compares the Triton block-sparse forward against the PyTorch gather reference
(BlockSparseSSA use_triton True vs False, identical weights/input/selection):
  1. small case    -- numerics unchanged by the edit;
  2. large-nblk    -- nblk > 65535, which the old grid=(B,H,nblk) could NOT launch
                      (CUDA gridDim.z max is 65535); the fix moves nblk to the x-axis.
"""
import torch
from ssa_model import BlockSparseSSA

dev = "cuda"


def check(tag, d, h, B, L, block, topk):
    torch.manual_seed(0)
    m = BlockSparseSSA(d, h, block=block, topk=topk, causal=True,
                       gate=True).to(dev).to(torch.bfloat16).eval()
    x = torch.randn(B, L, d, device=dev, dtype=torch.bfloat16)
    nblk = L // block
    with torch.no_grad():
        m.use_triton = False
        o_ref = m(x).float()
        m.use_triton = True
        o_tri = m(x).float()
    denom = o_ref.abs().max().item() + 1e-9
    rel = (o_ref - o_tri).abs().max().item() / denom
    ok = rel < 2e-2 and torch.isfinite(o_tri).all().item()
    print(f"[{tag:11}] L={L:>7} block={block:>3} nblk={nblk:>6}  rel_max_err={rel:.2e}  {'OK' if ok else 'FAIL'}")
    del m, x, o_ref, o_tri
    torch.cuda.empty_cache()
    return ok


if __name__ == "__main__":
    a = check("small", 256, 4, 2, 1024, 64, 4)
    # nblk = 70000 > 65535: exercises the grid fix
    b = check("large-nblk", 64, 2, 1, 70000 * 16, 16, 2)
    print("ALL_PASS" if (a and b) else "SOME_FAIL")
