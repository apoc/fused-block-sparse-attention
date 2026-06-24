# Design: Block-Sparse Attention Swap into Qwen3.6-35B-A3B (Plumbing PoC)

Date: 2026-06-22
Status: Draft, awaiting user review

## 1. Goal and success criteria

Validate that our content-dependent block-sparse selection can stand in for the
softmax attention in the full-attention layers of `Qwen3.6-35B-A3B` without
materially degrading language-model quality, measured by held-out perplexity. This
is a plumbing-first proof of concept: prove the swap works inside a real frontier
hybrid model and get a quality number, not a production conversion.

Success is a number either way:
1. Correctness gate: the patched model with all key-blocks selected reproduces the
   stock model's perplexity within noise (target < 1% relative). Until this passes,
   no sparse result is trusted.
2. Primary result: perplexity delta (block-sparse vs stock) at 16K context across a
   top-k budget sweep.
3. Stage 2: the distilled selector's perplexity delta vs the training-free selector
   at a matched budget.

A small delta means our selection preserves a real frontier model's quality at its
quadratic layers (strong). A large delta is an honest negative that quantifies what
distillation buys.

## 2. Background (ground truth from the model config)

`Qwen3.6-35B-A3B` (`model_type: qwen3_5_moe`) is a hybrid linear/full-attention MoE
vision-language model:
- 40 layers; `layer_types` alternates 3x `linear_attention` then 1x
  `full_attention` (`full_attention_interval: 4`). Only 10 of 40 layers are full
  softmax attention; the other 30 are linear attention. The model is already mostly
  sub-quadratic. The experiment is therefore "replace the residual quadratic
  full-attention layers with content-sparse selection," not "make it sub-quadratic."
- MoE: 256 experts, 8 active per token, shared expert. Vision tower present (VLM).
- Full-attention config: GQA (16 query heads, 2 KV heads), `head_dim` 256, partial
  RoPE (`partial_rotary_factor` 0.25) with MRoPE, `attn_output_gate: true`.
- bf16 weights ~70 GB; single NVIDIA GB10, 130 GB unified memory.

Our `BlockSparseSSA` and Triton kernel support none of GQA, RoPE/MRoPE, the output
gate, or `head_dim` 256. The PoC sidesteps this by reusing Qwen's projections and
positional code and replacing only the core attention computation.

## 3. Scope

In scope:
- The 10 `full_attention` layers (indices 3, 7, 11, ..., 39).
- Text-only forward; PyTorch chunked block-sparse; held-out perplexity.
- Stage 1 training-free selector, then Stage 2 local-distilled selector.

Out of scope (this PoC):
- The 30 linear-attention layers, MoE routing, and vision tower (untouched).
- The Triton kernel (GQA/RoPE port) — a speed/long-context optimization, not needed
  for a forward-only quality measurement.
- Full fine-tuning, base-weight training, generation/decode KV-cache.
- Long-context evals (RULER, NIAH) and context beyond 128K.

## 4. Design

### 4.1 Integration

Load the bf16 model via transformers (`qwen3_5_moe`, transformers 4.57.1 is present
on the DGX). Locate the `full_attention` layers and replace the inner attention
computation (post-RoPE q,k,v to attention output) while reusing the layer's q/k/v
projections, partial MRoPE, GQA grouping, output gate, and o_proj. The exact hook
point is determined by reading the `qwen3_5_moe` modeling source; implementation is
a monkeypatch of those modules' attention forward (or a registered custom attention
function).

Per converted layer the data flow is:
hidden -> (Qwen) q,k,v + partial MRoPE + GQA -> OUR select + SDPA over top-k key
blocks -> (Qwen) output gate -> o_proj.

### 4.2 Block-sparse core (PyTorch, GQA-aware, chunked)

Inputs: post-RoPE q `(B, Hq, L, d)`, k/v `(B, Hkv, L, d)`, causal positions.
- Partition keys into blocks of `Bs = 128`; `nblk = L / Bs`.
- For each query block i, select key blocks:
  - Stage 1 (training-free): `score_j = pool(q_i) . pool(k_j)` (mean-pool per
    block), pick the top-k content blocks; always include the own block i and the
    sink block 0; causal mask requires j <= i.
  - Stage 2 (distilled): scores from a small learned `sel_q`/`sel_k` trained to
    match the teacher per-block attention mass.
- Gather selected k/v, run SDPA(q_i, k_sel, v_sel) with a causal mask on the
  diagonal block. GQA handled by grouping query heads to their shared KV head.
- Chunk over query blocks so peak memory is one query block's working set (flat in
  context length). Reassemble output `(B, Hq, L, d)`.

### 4.3 Memory strategy (addresses the gather blow-up)

The naive gather is memory-heavy only at long context (our prior benchmark: ~42 GB
at 262k). Two measures keep PyTorch viable without the kernel:
1. Compact index-gather, never a dense L x L score/mask tensor.
2. Chunk the query-block loop so the working set is flat across context, at the cost
   of speed (acceptable for forward-only evaluation).
Additionally, compute the LM head and loss in sequence chunks (full logits at 16K
are ~16 GB for this vocab). Stage 1 runs forward-only, no grad.

Estimated transient per layer (gather + scores + probs, bf16, batch 1), processed
one layer at a time [estimates, to be confirmed on box]: 16K ~3 GB, 32K ~6 GB,
64K ~12 GB, 128K ~24 GB, 256K ~48 GB. With the chunked loop these collapse to under
1 GB at any length.

### 4.4 Stage 2 distillation (local, no 35B backprop)

Run the stock model once (no grad) over a small calibration set and cache, per
full-attention layer, the teacher per-block attention mass (softmax over keys,
pooled to blocks). Train each layer's `sel_q`/`sel_k` with KL(student block-scores
|| teacher block-mass) for a few hundred steps, training only the selector
parameters. No end-to-end backward through the frozen 35B. Calibrate at a shorter
context (4K to 8K) to keep the teacher's softmax affordable, then evaluate
perplexity at 16K.

### 4.5 Components

All code authored in `src/`, synced to the DGX flat layout (per AGENTS.md).
- `qwen_blocksparse.py`: GQA-aware chunked block-sparse core + selector (Stage 1/2).
- `patch_qwen.py`: load bf16 model, locate full-attention layers, patch attention.
- `eval_qwen_ppl.py`: held-out perplexity, stock vs sparse A/B, top-k sweep, chunked
  loss.
- `distill_qwen_sel.py`: cache teacher block-mass, train selectors, re-eval.
- `scripts/run_qwen_poc.sh`: orchestration in tmux on the DGX.

### 4.6 Data

Held-out English text tokenized with the model's own tokenizer, a few hundred
thousand tokens. Exact public source selected at implementation time and recorded
in the results JSON for reproducibility.

## 5. Decisions and defaults (locked unless changed at review)

- Model: bf16 `Qwen3.6-35B-A3B`.
- Evaluation context: 16K (default); 32K to 128K as a stretch if the 16K result is
  promising.
- Block size `Bs = 128`; content budget top-k in {4, 8, 16}, plus own block and
  sink block.
- Batch 1, bf16, no-grad for Stage 1.

## 6. Risks and mitigations

- Patch correctness (RoPE/GQA/output gate/causal mask): gated by the all-blocks
  correctness check before any sparse number is reported.
- Memory at long context: chunked query loop and chunked LM head; cap at 16K by
  default.
- Custom modeling code: read the `qwen3_5_moe` source; verify shapes against a tiny
  forward before the full run.
- VLM path: feed text-only inputs; confirm no image-token branch is exercised.
- Tokenizer/data mismatch: use the model's own tokenizer for the eval corpus.

## 7. Verification plan

1. Correctness gate: patched all-blocks path reproduces stock perplexity within
   noise.
2. Primary: sparse perplexity delta vs stock at 16K across the top-k sweep.
3. Stage 2: distilled vs training-free perplexity delta at a matched budget.
All numbers recorded to `results/` JSON with config and data source.

## 8. Out of scope / future work

- GQA + partial-MRoPE Triton kernel for throughput and very long context.
- Training base weights; full donor conversion with continued pretraining.
- RULER / NIAH / long-context retrieval; generation-time KV-cache decode.
