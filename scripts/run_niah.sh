#!/bin/bash
source .venv/bin/activate
cd ~/devel/ssa/src

echo "=== TRAIN DENSE (short, with checkpoint) ==="
python lm_tri.py --attn dense --steps 3000 --bs 16 --L 512 --d 512 --h 8 --layers 6 --lr 3e-4 --out lm_dense_ckpt.json
echo "DENSE DONE"

echo "=== TRAIN SPARSE TRITON (short, with checkpoint) ==="
python lm_tri.py --attn sparse --use_triton --gate --steps 3000 --bs 16 --L 512 --d 512 --h 8 --layers 6 --lr 3e-4 --block 64 --topk 4 --out lm_sparse_ckpt.json
echo "SPARSE DONE"

echo "=== NIAH DENSE ==="
python niah.py --attn dense --checkpoint lm_dense_ckpt.pt --lengths 512 --n_samples 50 --out niah_dense.json
echo "NIAH DENSE DONE"

echo "=== NIAH SPARSE ==="
python niah.py --attn sparse --use_triton --gate --checkpoint lm_sparse_ckpt.pt --block 64 --topk 4 --lengths 512 --n_samples 50 --out niah_sparse.json
echo "NIAH SPARSE DONE"

echo "ALL_DONE"
