#!/usr/bin/env bash
# Properly-tuned parity pass. Runbook default lr=3e-4 under-trains this task
cd ~/devel/ssa/src
# (dense_t.json plateaued ~0.18 at n_pairs=8); the sibling zip impl reached 88%
# with lr=3e-3. Bump lr to 1e-3, keep n_pairs=8 + curriculum.
set -euo pipefail
DEV=cuda; NP=8; LR=1e-3

echo "=== Exp A (n_pairs=$NP, lr=$LR, curriculum): parity dense vs sparse ==="
python3 train.py --attn dense  --device $DEV --curriculum --steps 6000 --seq_len 512 \
                 --n_pairs $NP --d 256 --lr $LR --out dense_h.json
python3 train.py --attn sparse --device $DEV --curriculum --steps 6000 --seq_len 512 \
                 --n_pairs $NP --d 256 --lr $LR --block 32 --topk 4 --sel_dim 32 --out sparse_h.json

echo "=== Exp C (n_pairs=$NP, lr=$LR, curriculum): selector topk ablation ==="
for K in 1 2 4 8; do
  python3 train.py --attn sparse --device $DEV --curriculum --steps 4000 --seq_len 512 \
                   --n_pairs $NP --d 256 --lr $LR --block 32 --topk $K --sel_dim 32 \
                   --out sparse_h_topk$K.json
done
echo "DONE_HILR. Outputs: dense_h.json sparse_h.json sparse_h_topk*.json"
