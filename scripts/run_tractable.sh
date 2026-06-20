#!/usr/bin/env bash
# Tractable parity experiment. 32-pair MQAR stalls for a 2-layer d=256 model even
cd ~/devel/ssa/src
# with curriculum (dense_cur.json), so Exp A/C parity is run at n_pairs=8 — still
# genuine multi-pair MQAR (chance=1/64), but within this tiny model's capacity.
set -euo pipefail
DEV=cuda
NP=8

echo "=== Exp A (n_pairs=$NP, curriculum): capability parity dense vs sparse ==="
python3 train.py --attn dense  --device $DEV --curriculum --steps 6000 --seq_len 512 \
                 --n_pairs $NP --d 256 --out dense_t.json
python3 train.py --attn sparse --device $DEV --curriculum --steps 6000 --seq_len 512 \
                 --n_pairs $NP --d 256 --block 32 --topk 4 --sel_dim 32 --out sparse_t.json

echo "=== Exp C (n_pairs=$NP, curriculum): selector topk ablation ==="
for K in 1 2 4 8; do
  python3 train.py --attn sparse --device $DEV --curriculum --steps 4000 --seq_len 512 \
                   --n_pairs $NP --d 256 --block 32 --topk $K --sel_dim 32 \
                   --out sparse_t_topk$K.json
done
echo "DONE_TRACTABLE. Outputs: dense_t.json sparse_t.json sparse_t_topk*.json"
