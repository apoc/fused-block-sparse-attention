#!/bin/bash
# Recall-vs-representation probe: can a learned hash gain recall WITHOUT the
# representation penalty? Same config as the decisive learned-hash run
# (block=4 nblk=128, top-4, 8 rounds, 64 buckets, gate, 3000 steps).
#   decoupled: learn R only (reps detached) -> oracle should stay ~6.5; does R alone help?
#   anneal:    lambda_h 0.5 -> 0 -> learn hash early, let reps recover for selection.
# Targets: learned co-trained was oracle 7.8 / LSH 10.2; block-sparse 7.9; random LSH 349.
cd ~/devel/ssa
source .venv/bin/activate
set -x
python lm_tri.py --attn lsh --learn_hash --hash_detach_reps --lambda_h 0.5 \
  --block 4 --topk 4 --cap 8 --n_buckets 64 --n_rounds 8 --gate \
  --steps 3000 --L 512 --d 512 --layers 6 --bs 16 \
  --data tinystories.bin --out lm_lh_detach_8r.json
python lm_tri.py --attn lsh --learn_hash --lambda_h 0.5 --lambda_h_final 0.0 \
  --block 4 --topk 4 --cap 8 --n_buckets 64 --n_rounds 8 --gate \
  --steps 3000 --L 512 --d 512 --layers 6 --bs 16 \
  --data tinystories.bin --out lm_lh_anneal_8r.json
echo ALL_DONE
