"""CPU unit tests for the Qwen block-sparse core (Tasks 2-4)."""
import torch
import torch.nn.functional as F
from qwen_blocksparse import (blocksparse_forward, SelectMax, SelectOracle,
                              SelectRandom, SelectAll)

torch.manual_seed(0)


def _ref_sdpa(q, k, v):
    grp = q.shape[1] // k.shape[1]
    kr = k.repeat_interleave(grp, dim=1)
    vr = v.repeat_interleave(grp, dim=1)
    return F.scaled_dot_product_attention(q, kr, vr, is_causal=True)


def test_allblocks_fastpath_matches_sdpa():
    B, Hq, Hkv, L, d = 1, 16, 2, 256, 64
    q = torch.randn(B, Hq, L, d); k = torch.randn(B, Hkv, L, d); v = torch.randn(B, Hkv, L, d)
    ref = _ref_sdpa(q, k, v)
    out = blocksparse_forward(q, k, v, selector=None, topk=9999, bs=32, causal=True)
    assert out.shape == ref.shape
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3), (out - ref).abs().max().item()


def test_gather_allcausal_matches_sdpa():
    # SelectAll routes through the gather/mask/dedup path; with causal masking it
    # must equal full causal SDPA. This is the core correctness test.
    B, Hq, Hkv, L, d = 1, 8, 2, 192, 48
    q = torch.randn(B, Hq, L, d); k = torch.randn(B, Hkv, L, d); v = torch.randn(B, Hkv, L, d)
    ref = _ref_sdpa(q, k, v)
    out = blocksparse_forward(q, k, v, selector=SelectAll(bs=32), topk=0, bs=32, causal=True, q_chunk=3)
    assert out.shape == ref.shape
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3), (out - ref).abs().max().item()


def test_selectmax_causal_and_budget():
    B, Hq, Hkv, L, d = 1, 16, 2, 256, 64
    q = torch.randn(B, Hq, L, d); k = torch.randn(B, Hkv, L, d)
    idx = SelectMax(topk=4, bs=32, sink=True).select(q, k)
    nblk = L // 32
    assert idx.shape == (B, Hkv, nblk, 4 + 2)        # topk + own + sink
    qb = torch.arange(nblk).view(1, 1, nblk, 1)
    assert (idx <= qb).all(), idx.max().item()        # never selects a future block
    assert (idx >= 0).all()


def test_selectmax_sparse_forward_finite():
    B, Hq, Hkv, L, d = 1, 8, 2, 256, 48
    q = torch.randn(B, Hq, L, d); k = torch.randn(B, Hkv, L, d); v = torch.randn(B, Hkv, L, d)
    out = blocksparse_forward(q, k, v, selector=SelectMax(topk=2, bs=32), topk=2, bs=32, q_chunk=2)
    assert out.shape == (B, Hq, L, d)
    assert torch.isfinite(out).all()


def test_selectoracle_picks_high_mass_block():
    B, Hkv, nblk = 1, 2, 8
    mass = torch.rand(B, Hkv, nblk, nblk) * 0.01
    mass[..., 0] = 1.0                                 # block 0 dominates for every query block
    idx = SelectOracle(topk=1, bs=32, sink=False).select_from_mass(mass)
    assert (idx == 0).any(-1).all(), idx               # block 0 selected in every query row


def test_selectrandom_causal_and_budget():
    B, Hq, Hkv, L, d = 1, 8, 2, 256, 48
    q = torch.randn(B, Hq, L, d); k = torch.randn(B, Hkv, L, d)
    idx = SelectRandom(topk=3, bs=32, sink=True, seed=1).select(q, k)
    nblk = L // 32
    assert idx.shape == (B, Hkv, nblk, 3 + 2)
    qb = torch.arange(nblk).view(1, 1, nblk, 1)
    assert (idx <= qb).all()


def test_selectlearned_causal_and_budget():
    from qwen_blocksparse import SelectLearned
    B, Hq, Hkv, L, d = 1, 16, 2, 256, 64
    q = torch.randn(B, Hq, L, d); k = torch.randn(B, Hkv, L, d)
    Wq = torch.randn(d, 32); Wk = torch.randn(d, 32)
    idx = SelectLearned(Wq, Wk, topk=4, bs=32, sink=True).select(q, k)
    nblk = L // 32
    assert idx.shape == (B, Hkv, nblk, 4 + 2)
    qb = torch.arange(nblk).view(1, 1, nblk, 1)
    assert (idx <= qb).all()
