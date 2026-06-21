#!/bin/bash
# Learned-hash block selector vs random LSH (train-dense / infer-LSH), nblk=128.
# Matches the decisive baselines: block=4 topk=4 cap=8 n_buckets=64 gate, 3000 steps,
# L=512 d=512 layers=6. Compare learned LSH-inference PPL against:
#   random-LSH  8r64b=349 / 4r64b=1115   (lm_td_lsh_nblk128.json)
#   dense-select oracle 6.56             (same file)
#   block-sparse matched budget 7.87     (lm_bs_block4.json)
cd ~/devel/ssa
source .venv/bin/activate
set -x
for NR in 8 4; do
  python lm_tri.py --attn lsh --learn_hash --lambda_h 0.5 \
    --block 4 --topk 4 --cap 8 --n_buckets 64 --n_rounds $NR --gate \
    --steps 3000 --L 512 --d 512 --layers 6 --bs 16 \
    --data tinystories.bin --out lm_learn_hash_${NR}r.json
done
echo ALL_DONE
