#!/usr/bin/env bash
# Joint selector-trained runs: gate coupling + L_route (supervise needle block).
cd ~/devel/ssa/src
# Compare against the untrained-selector results (sparse_h*.json).
set -euo pipefail
DEV=cuda; LR=1e-3

echo "=== Exp A: trained selector (gate+route), n_pairs=8 ==="
python3 train.py --attn sparse --device $DEV --curriculum --steps 6000 --seq_len 512 \
  --n_pairs 8 --d 256 --lr $LR --block 32 --topk 4 --sel_dim 32 \
  --gate --route_lambda 1.0 --out sparse_sel.json

echo "=== Exp C: topk sweep, trained selector, n_pairs=8 ==="
for K in 1 2 4 8; do
  python3 train.py --attn sparse --device $DEV --curriculum --steps 4000 --seq_len 512 \
    --n_pairs 8 --d 256 --lr $LR --block 32 --topk $K --sel_dim 32 \
    --gate --route_lambda 1.0 --out sparse_sel_topk$K.json
done

echo "=== Harder: trained selector at n_pairs=16 and 32 (+dense parity) ==="
python3 train.py --attn dense  --device $DEV --curriculum --steps 6000 --seq_len 512 \
  --n_pairs 16 --d 256 --lr $LR --out dense_np16.json
python3 train.py --attn sparse --device $DEV --curriculum --steps 6000 --seq_len 512 \
  --n_pairs 16 --d 256 --lr $LR --block 32 --topk 4 --sel_dim 32 \
  --gate --route_lambda 1.0 --out sparse_sel_np16.json
python3 train.py --attn sparse --device $DEV --curriculum --steps 8000 --seq_len 512 \
  --n_pairs 32 --d 256 --lr $LR --block 32 --topk 4 --sel_dim 32 \
  --gate --route_lambda 1.0 --out sparse_sel_np32.json
echo "DONE_SELECTOR_EVAL"
