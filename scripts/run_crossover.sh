#!/usr/bin/env bash
set -euo pipefail
cd ~/devel/ssa
source .venv/bin/activate
python bench_crossover.py \
    --blocks 32,64,128 \
    --lengths 262144,524288,1048576,4194304,8388608,16777216 \
    --out bench_crossover.csv 2>&1 | tee bench_crossover.log
echo "DONE_CROSSOVER"
