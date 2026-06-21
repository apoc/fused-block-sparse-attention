"""Causal-leak test for LSHBucketSSA(causal=True).

A causal module's output at positions <= p must not change when inputs at
positions > p are perturbed. We perturb a future block and a middle block and
assert earlier-position outputs are unchanged. A nonzero leak means the LM
perplexity would be invalid (the model could see the future).
"""
import torch
from ssa_model import LSHBucketSSA

dev = "cuda"


def leak(perturb_from_block, dense_select=False):
    torch.manual_seed(0)
    d, h, B, L, block = 128, 4, 2, 256, 16
    nblk = L // block
    m = LSHBucketSSA(d, h, block=block, topk=4, causal=True, gate=True).to(dev).float().eval()
    m.dense_select = dense_select
    x = torch.randn(B, L, d, device=dev)
    p = perturb_from_block * block          # first perturbed position
    with torch.no_grad():
        o1 = m(x)
        x2 = x.clone()
        x2[:, p:] += 10.0                   # perturb everything from block `perturb_from_block` on
        o2 = m(x2)
    d_before = (o1[:, :p] - o2[:, :p]).abs().max().item()
    d_after = (o1[:, p:] - o2[:, p:]).abs().max().item()
    print(f"dense_select={dense_select} perturb block {perturb_from_block:>2} (pos {p:>3}): "
          f"leak_before={d_before:.2e}  changed_after={d_after:.2e}")
    return d_before


if __name__ == "__main__":
    res = []
    for ds in (False, True):
        res.append(leak(15, ds))
        res.append(leak(8, ds))
    print("CAUSAL_OK" if all(r < 1e-5 for r in res) else "CAUSAL_LEAK")
