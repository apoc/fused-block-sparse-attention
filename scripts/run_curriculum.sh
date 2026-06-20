#!/usr/bin/env bash
# Curriculum re-run of Exp A + Exp C (runbook §5 stall remedy: --curriculum).
cd ~/devel/ssa/src
# Baselines (dense.json/sparse.json) were as-written and stalled at chance.
set -euo pipefail
DEV=cuda

echo "=== Exp A (curriculum): capability parity dense vs sparse ==="
python3 train.py --attn dense  --device $DEV --curriculum --steps 10000 --seq_len 512 \
                 --n_pairs 32 --d 256 --out dense_cur.json
python3 train.py --attn sparse --device $DEV --curriculum --steps 10000 --seq_len 512 \
                 --n_pairs 32 --d 256 --block 32 --topk 4 --sel_dim 32 --out sparse_cur.json

echo "=== Exp C (curriculum): selector topk ablation ==="
for K in 1 2 4 8; do
  python3 train.py --attn sparse --device $DEV --curriculum --steps 5000 --seq_len 512 \
                   --n_pairs 32 --d 256 --block 32 --topk $K --sel_dim 32 \
                   --out sparse_cur_topk$K.json
done
echo "DONE_CURRICULUM. Outputs: dense_cur.json sparse_cur.json sparse_cur_topk*.json"
