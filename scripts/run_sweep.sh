#!/bin/bash
source .venv/bin/activate
cd ~/devel/ssa/src

echo "=== SAME SYMBOL PROBE ==="
python route_probe.py --writers none,full,linear,window --task same \
    --steps 8000 --L 128 --layers 1 --gap 1 --lr 1e-3 --out probe_same.json

echo "=== CENTROID SWEEP ==="
for C in 16 32 64; do
  for CAP in 4 8; do
    echo "--- C=${C} cap=${CAP} ---"
    python train.py --attn centroid --gate --route_lambda 1.0 --route_anneal \
        --steps 10000 --n_pairs 8 --curriculum \
        --n_centroids "${C}" --cap "${CAP}" \
        --out "centroid_C${C}_cap${CAP}.json"
  done
done

echo "ALL_DONE"
