"""Smoke test for the learned-hash LSH selector: parameter R, grad flow,
hash-loss finiteness, inference path, and causal non-leak."""
import torch
from ssa_model import LSHBucketSSA

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
B, L, d, h = 2, 64, 64, 4
m = LSHBucketSSA(d, h, block=4, topk=4, sel_dim=32, n_rounds=8, n_buckets=16,
                 cap=8, gate=True, causal=True, learn_hash=True).to(dev)

assert isinstance(m.R, torch.nn.Parameter), "R must be a Parameter when learn_hash"

# --- train path: dense select + hash-alignment loss, check gradients ---
m.train(); m.dense_select = True
x = torch.randn(B, L, d, device=dev)
out = m(x)
hl = m.last_hash_loss
assert hl is not None and torch.isfinite(hl), f"bad hash loss: {hl}"
(out.float().pow(2).mean() + hl).backward()
assert m.R.grad is not None and m.R.grad.abs().sum() > 0, "R got NO gradient"
assert m.sel_q[0].weight.grad is not None, "sel_q got no gradient"
print(f"TRAIN ok: hash_loss={hl.item():.4f}  R.grad_norm={m.R.grad.norm().item():.4f}")

# --- inference path: LSH bucketing with the learned R ---
m.eval(); m.dense_select = False
with torch.no_grad():
    o = m(x)
assert torch.isfinite(o).all(), "inference output not finite"
print("INFER ok: out", tuple(o.shape), "finite")

# --- causal non-leak: perturbing the future must not change the past ---
x1 = torch.randn(B, L, d, device=dev)
x2 = x1.clone(); x2[:, L // 2:] += 5.0
with torch.no_grad():
    o1, o2 = m(x1), m(x2)
leak = (o1[:, : L // 2 - 4] - o2[:, : L // 2 - 4]).abs().max().item()
print(f"causal leak (learned-hash infer): {leak:.2e}")
assert leak < 1e-4, "CAUSAL LEAK in learned-hash inference"
print("ALL SMOKE CHECKS PASSED")
