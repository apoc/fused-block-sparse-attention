#!/usr/bin/env bash
set -euo pipefail
cd ~/devel/ssa
source .venv/bin/activate
# Train DENSE all-pairs top-k (allowed quadratic at train time), infer LSH (linear).
# Reports dense oracle (final_*) vs LSH inference (lsh_*) plus an inference
# rounds/buckets sweep (R is unused during dense training, so retuning is free).
C="--steps 6000 --batch 64 --n_pairs 8 --seq_len 512 --topk 4 --gate --route_lambda 1.0 \
   --route_anneal --curriculum --log_every 3000 --cap 8"
echo "=== nblk=16 (block 32) ==="
python train.py --attn lsh --block 32 $C --out lsh_td_b32.json
echo "=== nblk=64 (block 8) ==="
python train.py --attn lsh --block 8 $C --out lsh_td_b8.json
echo "=== nblk=128 (block 4) ==="
python train.py --attn lsh --block 4 $C --out lsh_td_b4.json
echo "DONE_TDSWEEP"
