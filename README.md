# Fused Block-Sparse Attention with Learned Content-Dependent Selection

A study of content-dependent sparse attention for long-context transformers:
a fused Triton kernel that reaches memory parity with dense FlashAttention and
7.4x latency speedup at 262k tokens, plus an honest characterization of why
making the *selection* step genuinely linear is hard.

## Results

### Fused kernel vs dense FlashAttention

| Metric | Dense (baseline) | Block-Sparse + Triton |
|--------|-----------------|----------------------|
| Val perplexity (TinyStories, 45M) | 14.41 | 15.00 |
| Memory at 262k tokens | 1.12 GB | **1.11 GB** |
| Latency at 262k tokens | 797 ms | **114 ms** (7.4x faster) |
| Latency crossover | - | ~32k tokens |
| MQAR retrieval hit | - | 0.99 |

### Selection vs attention read (`bench_crossover.py`, to 16M tokens)

The fused attention *read* is linear (fit exponent 0.96 to 1.02). The block-pair
*selection* is quadratic (exponent 1.5 to 1.9) and is the real bottleneck:

- Single-level all-pairs selection has an **O(N^4/3) compute floor**: no
  block-size schedule yields end-to-end linearity, the quadratic only moves
  between the selection and read stages (B ~ sqrt(N) makes selection linear but
  the read becomes N^1.5).
- The O(nblk^2) score matrix **OOMs before any latency crossover**: at 4M
  (block 32), 8M (block 64), 16M (block 128), while the linear read runs to 16M
  (86 GB) on the same GPU.

### Linear-selection probes (MQAR, accuracy / recall)

| Selector | cost | nblk=16 | nblk=64 | gate-only (nblk=64) |
|----------|------|---------|---------|---------------------|
| Block-sparse | O(N^2/B^2) | 0.82 / 1.00 | 0.96 / 1.00 | **0.96 / 0.78** |
| **LSH bucketing** | linear | 0.76 / 0.84 | 0.92 / 0.84 | **0.20 to 0.34** (fragile) |
| Centroid routing | linear | 0.17 / 1.00 | - | - |

- **LSH** prunes the search, not the representation, so a selected block is
  scored at full fidelity. Under supervised routing it matches block-sparse and
  far exceeds centroid: linear cost *can* preserve accuracy when the
  representation stays lossless. Recall plateaus at hit 0.84 (hash-collision miss).
- **Unsupervised (gate-only) LSH is fragile**: recall is sensitive to the cosine
  gate temperature (hit 0.09, 0.34, 0.02 at scale 5.66, 30, 50) and never reaches
  block-sparse's gate-only level. Mechanism unresolved; training a lossless-refine
  linear selector without supervision is left open.
- **Centroid** reaches linear cost and perfect recall but its lossy summary
  breaks accuracy (representation failure).

All raw experiment outputs are in `results/` (62 JSON/CSV files).

## Directory Structure

```
fused-block-sparse-attention/
├── src/                    # Python source code
│   ├── ssa_model.py        # Attention modules (Dense, BlockSparse, Centroid, LSHBucket)
│   ├── ssa_data.py         # MQAR data generator
│   ├── train.py            # MQAR training loop
│   ├── triton_kernel.py    # Fused Triton kernel (forward + backward, int64 offsets)
│   ├── verify_kernel.py    # Kernel correctness check vs gather reference
│   ├── llm.py              # Char-level LM (shakespeare)
│   ├── lm_tri.py           # Tokenized LM (TinyStories, Triton)
│   ├── niah.py             # Needle-in-a-haystack test
│   ├── bench.py            # Latency/memory benchmark (PyTorch)
│   ├── bench_triton.py     # Latency/memory benchmark (Triton)
│   ├── bench_crossover.py  # Selection vs read scaling to 16M (the N^4/3 floor)
│   ├── bench2.py           # Extended benchmark (causal+gate, centroid)
│   ├── stream_bench.py     # Streaming online-softmax (PyTorch reference)
│   ├── scale_sel.py        # Selection cost scaling proof
│   ├── route_probe.py      # Routing subspace probe
│   └── summarize.py        # Results summary utility
├── scripts/                # Shell scripts for DGX execution (run_*.sh)
├── results/                # Experiment outputs (62 JSON/CSV files)
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
supervision on synthetic tasks.

### 2. Fused Triton Kernel (`src/triton_kernel.py`)

- **Forward:** Online-softmax in SRAM, iterating over selected key blocks via
  index list. Never materializes gathered K/V.
- **Backward:** 3-pass recompute (forward, D-accumulation, gradients).
  `atomic_add` for dK/dV (shared key blocks), direct write for dQ.
- **Custom autograd:** `BSAttnFunction` routes forward/backward through Triton.
- **int64 offsets + nblk on the grid x-axis**, so it runs past 4M tokens
  (avoids the int32 offset overflow and the CUDA gridDim.z 65535 cap).
- **Verified:** forward 9.77e-4, backward <5e-3, large-nblk (70000) 4.6e-3 vs
  the PyTorch gather reference.

### 3. The Selection Floor (`src/bench_crossover.py`)

Isolates selection from the attention read across context length and block size.
Establishes that single-level block-pair selection is quadratic with an O(N^4/3)
floor, and that the score matrix OOMs before the latency crossover. See Results.

### 4. Linear-Selection Probes (`CentroidSSA`, `LSHBucketSSA`)

Two linear-time replacements for the O(nblk^2) block-pair scoring. Centroid
routing fails on representation; LSH bucketing matches block-sparse under
supervised routing but is fragile without supervision. See Results.

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

# Linear-selection probe (LSH bucketing)
python train.py --attn lsh --gate --route_lambda 1.0 --route_anneal \
    --steps 6000 --n_pairs 8 --curriculum --block 32 --topk 4 \
    --n_rounds 4 --n_buckets 8 --cap 8

# Kernel correctness check
python verify_kernel.py

# Triton read benchmark
python bench_triton.py --lengths 4096,16384,65536,131072,262144

# Selection vs read scaling (the N^4/3 floor)
python bench_crossover.py --blocks 32,64,128 \
    --lengths 262144,1048576,4194304,8388608,16777216

# TinyStories LM (dense vs sparse-Triton)
python lm_tri.py --attn sparse --use_triton --gate --steps 5000 \
    --bs 16 --L 512 --d 512 --h 8 --layers 6
```

## License

Research use only. See authors for details.

## Author

Miroslav Drbal (mdrbal@nymfe.net)
