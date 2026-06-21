#!/usr/bin/env bash
set -euo pipefail
cd ~/devel/ssa
source .venv/bin/activate
# Gate-only TinyStories LM, block=16 (nblk=32 at L=512) so LSH bucketing is non-trivial.
# All three trained identically (gate coupling, no route, no distill) for a clean
# lsh-vs-block-sparse comparison; dense is the oracle.
C="--steps 3000 --bs 16 --L 512 --d 512 --h 8 --layers 6 --block 16 --topk 4 --gate --data tinystories.bin"
echo "=== dense ==="
python lm_tri.py --attn dense $C --out lm_lsh_dense.json
echo "=== sparse (gate-only) ==="
python lm_tri.py --attn sparse $C --out lm_lsh_sparse.json
echo "=== lsh (gate-only) ==="
python lm_tri.py --attn lsh $C --n_rounds 4 --n_buckets 8 --cap 8 --out lm_lsh_lsh.json
echo "DONE_LM"
