#!/usr/bin/env bash
set -euo pipefail
cd ~/devel/ssa
source .venv/bin/activate
# All runs seq512, 6000 steps (reaches the ~0.77 regime). nblk swept via BLOCK SIZE.
# n_buckets=8 => random LSH recall ~0.41 even at nblk=16, so recall is never "free".
C="--steps 6000 --batch 64 --n_pairs 8 --seq_len 512 --topk 4 --d 256 --h 4 --layers 2 \
   --lr 3e-4 --gate --route_lambda 1.0 --route_anneal --curriculum --log_every 1500 \
   --n_rounds 4 --n_buckets 8 --cap 8"

# nblk=16 (block=32): baselines + lsh
for ATTN in dense sparse centroid lsh; do
  echo "=== $ATTN block=32 (nblk=16) ==="
  python train.py --attn $ATTN --block 32 $C --out lsh_${ATTN}_b32.json
done

# nblk=64 (block=8): harder recall (more blocks). Some padding -> ablation disambiguates.
for ATTN in sparse lsh; do
  echo "=== $ATTN block=8 (nblk=64) ==="
  python train.py --attn $ATTN --block 8 $C --out lsh_${ATTN}_b8.json
done

# ablation: route OFF at nblk=64 -> if recall still high, it is structure/padding not learning.
echo "=== lsh block=8 route0 (ablation) ==="
python train.py --attn lsh --block 8 --steps 6000 --batch 64 --n_pairs 8 --seq_len 512 \
   --topk 4 --d 256 --h 4 --layers 2 --lr 3e-4 --gate --route_lambda 0.0 --curriculum \
   --log_every 1500 --n_rounds 4 --n_buckets 8 --cap 8 --out lsh_lsh_b8_r0.json

# transferability: content-dense (n_pairs=64 -> all 16 blocks full, no padding crutch)
for ATTN in sparse lsh; do
  echo "=== $ATTN block=32 dense-content (n_pairs=64) ==="
  python train.py --attn $ATTN --block 32 --n_pairs 64 --steps 6000 --batch 64 --seq_len 512 \
     --topk 4 --d 256 --h 4 --layers 2 --lr 3e-4 --gate --route_lambda 1.0 --route_anneal \
     --curriculum --log_every 1500 --n_rounds 4 --n_buckets 8 --cap 8 --out lsh_${ATTN}_dense.json
done

echo "DONE_LSH_SWEEP"
