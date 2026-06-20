#!/usr/bin/env bash
set -euo pipefail
cd ~/devel/ssa
source .venv/bin/activate
# Confound check: LSH gate-only (route off) at nblk=64 with LARGE gate temperature,
# so logsigmoid(scale*cosine) has dynamic range comparable to block-sparse's raw dot.
# If LSH still collapses -> hard-hashing trainability wall is real (not weak gate).
# If it recovers -> the collapse was a saturated/weak cosine gate.
for S in 30 50; do
  echo "=== lsh gate-only nblk=64 scale=$S ==="
  python train.py --attn lsh --block 8 --steps 6000 --batch 64 --n_pairs 8 --seq_len 512 \
     --topk 4 --d 256 --h 4 --layers 2 --lr 3e-4 --gate --route_lambda 0.0 --curriculum \
     --log_every 3000 --n_rounds 4 --n_buckets 8 --cap 8 --lsh_scale $S \
     --out lsh_lsh_b8_r0_s${S}.json
done
echo "DONE_SCALE"
