#!/usr/bin/env bash
# Causal char-LM on shakespeare.txt: dense vs block-sparse-causal (gate, and
cd ~/devel/ssa/src
# gate+distill). Equal params/budget; compare validation bits-per-char + ppl.
set -euo pipefail
S=4000; L=512; BLK=32; TOPK=4; D=256; LAYERS=4

echo "=== dense-causal ==="
python llm.py --attn dense  --steps $S --L $L --d $D --layers $LAYERS --out llm_dense.json
echo "=== block-sparse-causal (gate) ==="
python llm.py --attn sparse --gate --steps $S --L $L --d $D --layers $LAYERS \
  --block $BLK --topk $TOPK --out llm_sparse.json
echo "=== block-sparse-causal (gate + distill) ==="
python llm.py --attn sparse --gate --distill_lambda 1.0 --steps $S --L $L --d $D --layers $LAYERS \
  --block $BLK --topk $TOPK --out llm_sparse_distill.json
echo DONE_LLM
