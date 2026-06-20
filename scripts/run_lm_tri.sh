#!/bin/bash
source .venv/bin/activate
cd ~/devel/ssa/src

echo "=== DENSE BASELINE ==="
python lm_tri.py --attn dense --steps 5000 --bs 16 --L 512 --d 512 --h 8 --layers 6 --lr 3e-4 --out lm_dense_tri.json
echo "DENSE DONE"

echo "=== SPARSE TRITON ==="
python lm_tri.py --attn sparse --use_triton --gate --steps 5000 --bs 16 --L 512 --d 512 --h 8 --layers 6 --lr 3e-4 --block 64 --topk 4 --out lm_sparse_tri.json
echo "SPARSE DONE"

echo "ALL_DONE"
