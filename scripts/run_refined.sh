#!/usr/bin/env bash
# Exp A with (a) refined BlockSparse (MLP + mean/max + decoupled local block) and
cd ~/devel/ssa/src
# (b) the sub-quadratic centroid selector. Both gate + L_route, n_pairs=8.
set -euo pipefail
DEV=cuda; LR=1e-3

echo "=== refined BlockSparse (gate+route), 8-pair ==="
python3 train.py --attn sparse --device $DEV --curriculum --steps 6000 --seq_len 512 \
  --n_pairs 8 --d 256 --lr $LR --block 32 --topk 4 --sel_dim 32 \
  --gate --route_lambda 1.0 --out sparse_ref.json

echo "=== centroid selector O(N*C) (gate+route), 8-pair ==="
python3 train.py --attn centroid --device $DEV --curriculum --steps 6000 --seq_len 512 \
  --n_pairs 8 --d 256 --lr $LR --block 32 --topk 4 --sel_dim 32 \
  --n_centroids 16 --cpq 2 --cap 2 --gate --route_lambda 1.0 --out centroid.json
echo DONE_REFINED
