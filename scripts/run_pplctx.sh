#!/usr/bin/env bash
set -euo pipefail
cd ~/devel/ssa/src
S=3000; L=512; D=256; LAYERS=4
python llm.py --attn dense  --steps $S --L $L --d $D --layers $LAYERS --out llm_dense_ctx.json
python llm.py --attn sparse --gate --steps $S --L $L --d $D --layers $LAYERS \
  --block 32 --topk 4 --out llm_sparse_ctx.json
echo DONE_PPLCTX
