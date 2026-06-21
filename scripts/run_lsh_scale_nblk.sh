#!/usr/bin/env bash
set -euo pipefail
cd ~/devel/ssa
source .venv/bin/activate
# Round-budget scaling: train dense, infer LSH, sweep rounds at ~constant bucket size,
# across nblk = 128/256/512 (block 4). Question: does recall hold at FIXED rounds as nblk grows?
C="--attn lsh --steps 6000 --batch 64 --n_pairs 8 --block 4 --topk 4 --gate --route_lambda 1.0 \
   --route_anneal --curriculum --log_every 3000 --cap 8"
echo "=== nblk=128 (seq 512) ==="
python train.py $C --seq_len 512 --out lsh_sc_128.json
echo "=== nblk=256 (seq 1024) ==="
python train.py $C --seq_len 1024 --out lsh_sc_256.json
echo "=== nblk=512 (seq 2048) ==="
python train.py $C --seq_len 2048 --out lsh_sc_512.json
echo "DONE_SCNBLK"
