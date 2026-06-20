# Fused Block-Sparse Attention with Learned Content-Dependent Selection

A sub-quadratic attention mechanism for long-context transformers, implemented
with a fused Triton kernel that achieves memory parity with dense
FlashAttention and 7.4x latency speedup at 262k tokens.

## Results

| Metric | Dense (baseline) | Block-Sparse + Triton |
|--------|-----------------|----------------------|
| Val perplexity (TinyStories, 45M) | 14.41 | 15.00 |
| Memory at 262k tokens | 1.12 GB | **1.11 GB** |
| Latency at 262k tokens | 797 ms | **114 ms** (7.4x faster) |
| Latency crossover | - | ~32k tokens |
| MQAR retrieval hit | - | 0.99 |

All raw experiment outputs are in `results/` (48 JSON/CSV files).

## Directory Structure

```
fused-block-sparse-attention/
├── src/                    # Python source code
│   ├── ssa_model.py        # Core attention modules (Dense, BlockSparse, Centroid)
│   ├── ssa_data.py         # MQAR data generator
│   ├── train.py            # MQAR training loop
│   ├── triton_kernel.py    # Fused Triton kernel (forward + backward)
│   ├── llm.py              # Char-level LM (shakespeare)
│   ├── lm_tri.py           # Tokenized LM (TinyStories, Triton)
│   ├── niah.py             # Needle-in-a-haystack test
│   ├── bench.py            # Latency/memory benchmark (PyTorch)
│   ├── bench_triton.py     # Latency/memory benchmark (Triton)
│   ├── bench2.py           # Extended benchmark (causal+gate, centroid)
│   ├── stream_bench.py     # Streaming online-softmax (PyTorch reference)
│   ├── scale_sel.py        # Selection cost scaling proof
│   ├── route_probe.py      # Routing subspace probe (arXiv:2603.20997)
│   └── summarize.py        # Results summary utility
├── scripts/                # Shell scripts for DGX execution
├── results/                # Experiment outputs (48 JSON/CSV files)
├── paper/                  # Workshop paper
│   ├── main.tex
│   └── main.pdf
└── README.md
```

## Key Components

### 1. Gate-Coupled Selector Training

The block selector is trained jointly with the model via MoE-style gate
coupling: the selector score is injected as `log_sigmoid(score)` bias into
attention logits, creating a differentiable path. An auxiliary routing loss
(cross-entropy for MQAR, InfoNCE for centroid routing) provides cold-start
supervision.

### 2. Fused Triton Kernel (`src/triton_kernel.py`)

- **Forward:** Online-softmax in SRAM, iterating over selected key blocks via
  index list. Never materializes gathered K/V.
- **Backward:** 3-pass recompute (forward, D-accumulation, gradients).
  `atomic_add` for dK/dV (shared key blocks), direct write for dQ.
- **Custom autograd:** `BSAttnFunction` routes forward/backward through Triton.
- **Verified:** forward 9.77e-4, backward <5e-3 vs PyTorch autograd.

### 3. Centroid Routing (`CentroidSSA`)

Linear selection (O(nblk * C)) via learned centroids. InfoNCE contrastive
training fixes the circular routing loss (hit 0.19 -> 1.0). Negative finding:
perfect recall but accuracy bottlenecked by representation quality under
coarse routing.

## Hardware

- NVIDIA GB10 Grace-Blackwell, 128 GB unified memory
- aarch64, Triton 3.7.1, PyTorch 2.12.1+cu130
- All experiments run on a single GPU

## Reproduction

From the `src/` directory with a CUDA GPU:

```bash
# MQAR training (block-sparse with gate coupling)
python train.py --attn sparse --gate --route_lambda 1.0 --route_anneal \
    --steps 6000 --n_pairs 8 --curriculum --block 32 --topk 4

# Triton benchmark
python bench_triton.py --lengths 4096,16384,65536,131072,262144

# TinyStories LM (dense vs sparse-Triton)
python lm_tri.py --attn sparse --use_triton --gate --steps 5000 \
    --bs 16 --L 512 --d 512 --h 8 --layers 6
```

## License

Research use only. See authors for details.

## Author

Miroslav Drbal (mdrbal@nymfe.net)
