#!/usr/bin/env bash
set -euo pipefail
cd ~/devel/ssa/src
DEV=cuda

echo "=== Exp 0: correctness gate ==="
python3 - << 'PY'
import torch
from ssa_model import DenseAttention, BlockSparseSSA
torch.manual_seed(0)
B,L,D,H = 2, 512, 128, 4
x = torch.randn(B,L,D)
sp = BlockSparseSSA(D,H, block=64, topk=10_000, sel_dim=16)  # select all blocks
dn = DenseAttention(D,H)
dn.qkv.load_state_dict(sp.qkv.state_dict()); dn.o.load_state_dict(sp.o.state_dict())
with torch.no_grad():
    diff = (dn(x)-sp(x)).abs().max().item()
print("max|dense - sparse(all blocks)| =", diff)
assert diff < 1e-3, "CORRECTNESS GATE FAILED"
print("PASS")
PY

echo "=== Exp A: capability parity on MQAR (dense vs sparse) ==="
python3 train.py --attn dense  --device $DEV --steps 6000 --seq_len 512 --n_pairs 32 --out dense.json
python3 train.py --attn sparse --device $DEV --steps 6000 --seq_len 512 --n_pairs 32 \
                 --block 32 --topk 4 --sel_dim 32 --out sparse.json

echo "=== Exp B: latency / memory crossover ==="
python3 bench.py --device $DEV --D 512 --H 8 --B 1 --block 128 --topk 8 \
                 --lengths 4096,8192,16384,32768,65536,131072,262144 --out bench.csv

echo "=== Exp C: selector ablations ==="
for K in 1 2 4 8; do
  python3 train.py --attn sparse --device $DEV --steps 4000 --seq_len 512 --n_pairs 32 \
                   --block 32 --topk $K --sel_dim 32 --out sparse_topk$K.json
done
echo "DONE. Outputs: dense.json sparse.json bench.csv sparse_topk*.json"
