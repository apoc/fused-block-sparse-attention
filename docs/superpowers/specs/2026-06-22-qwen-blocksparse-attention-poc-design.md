# Design: Block-Sparse Attention Swap into Qwen3.6-35B-A3B (Plumbing PoC)

Date: 2026-06-22
Status: Reviewed, amended (env prerequisite, RoPE-safe selection, baselines,
distillation pooling, concrete thresholds).

## 1. Goal and success criteria

Validate that our content-dependent block-sparse selection can stand in for the
softmax attention in the full-attention layers of `Qwen3.6-35B-A3B` without
materially degrading language-model quality, measured by held-out perplexity. This
is a plumbing-first proof of concept: prove the swap works inside a real frontier
hybrid model and get a quality number, not a production conversion.

Note: this is a HuggingFace `transformers` monkeypatch, not a vllm modification.
vllm has its own attention kernels and is out of scope; we target the HF modeling
code path.

Success is a number, with a concrete bar:
1. Correctness gate: the patched model with all key-blocks selected reproduces the
   stock model's perplexity within noise (target < 1% relative, allowing flash vs
   SDPA numerical differences, not exact). Until this passes, no sparse result is
   trusted.
2. Primary result: perplexity delta (block-sparse vs stock) at 16K context across a
   top-k budget sweep, reported as mean and spread over 50 to 100 eval sequences.
3. Baselines (required, not optional): an **oracle** selection (top-k key blocks by
   the stock model's own per-block attention mass; isolates the budget ceiling) and
a
   **random-k + own + sink** baseline (lower bound; shows content selection does
   work). Without these a delta is uninterpretable as budget vs method.
4. Stage 2: the distilled selector's perplexity delta vs the training-free selector
   at a matched budget.

Thresholds: at top-k=16 (own + 16 content + sink at 16K context, ~13% of blocks), a
relative perplexity increase below 5% is a strong positive; above 20% is a clear
negative; in between is partial and reported as such.

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

Load the bf16 model via HuggingFace `transformers` (the model type `qwen3_5_moe`
requires transformers >= 4.57.1). **Prerequisite (Phase 0):** no `transformers` is
presently installed on the DGX host; the Qwen weights currently run only inside a
vllm container (`~/devel/spark-vllm-docker`). Stand up a Python venv with
transformers >= 4.57.1 that can `AutoModelForCausalLM.from_pretrained` the model
and run a text forward pass, before any patching code is written. Phase 0 is a
gate: if the model type is not loadable on the host aarch64/GB10 environment, the
PoC falls back to running inside the vllm container or a custom image, and the plan
is re-scoped. This is not a vllm modification; we target the HF modeling code path.
Locate the `full_attention` layers and replace the inner attention computation
(post-RoPE q,k,v to attention output) while reusing the layer's q/k/v projections,
partial MRoPE, GQA grouping, output gate, and o_proj. The exact hook point is
determined by reading the `qwen3_5_moe` modeling source; implementation is a
monkeypatch of those modules' attention forward.

Per converted layer the data flow is:
hidden -> (Qwen) q,k,v + partial MRoPE + GQA -> OUR select + SDPA over top-k key
blocks -> (Qwen) output gate -> o_proj.

### 4.2 Block-sparse core (PyTorch, GQA-aware, chunked)

Inputs: post-RoPE q `(B, Hq, L, d)`, k/v `(B, Hkv, L, d)`, causal positions.
- Partition keys into blocks of `Bs = 128`; `nblk = L / Bs`.
- For each query block i, select key blocks. **Selection is RoPE-safe:** do not pool
  post-RoPE vectors and dot them (RoPE rotates a different subset per position,
  contaminating pooled similarity). Instead score key block j for query block i by
  a max over per-token post-RoPE dot products, `score_j = max_{t in i, s in j}
  q_t . k_s`, a Quest-style estimate of block importance that uses post-RoPE
  similarity directly.
  - Stage 1 (training-free): pick the top-k content blocks by `score_j`; always
    include the own block i and the sink block 0; causal mask requires j <= i.
  - Stage 2 (distilled): scores from a small learned `sel_q`/`sel_k` (operating on
    pre-RoPE hidden states, so the selector is position-independent) trained to
    match the teacher per-block attention mass.
- Gather selected k/v, run SDPA(q_i, k_sel, v_sel) with a causal mask on the
  diagonal block. **GQA granularity:** select per KV head (2 selections, each shared
  across its 8 query heads); per-query-head selection is a noted quality variant,
  not the default.
- Chunk over query blocks so peak memory is one query block's working set (flat in
  context length). Reassemble output `(B, Hq, L, d)`.

### 4.3 Memory strategy (addresses the gather blow-up)

The naive gather is memory-heavy only at long context (our prior benchmark: ~42 GB
at 262k). Two measures keep PyTorch viable without the kernel:
1. Compact index-gather, never a dense L x L score/mask tensor.
2. Chunk the query-block loop so the working set is flat across context, at the cost
   of speed (acceptable for forward-only evaluation).
Additionally, compute the LM head and loss in sequence chunks. Full logits at 16K
are ~8 GB in bf16 or ~16 GB upcast to fp32 for the loss; we upcast, hence chunking.
Stage 1 runs forward-only, no grad.

Estimated transient per layer (gather + scores + probs, bf16, batch 1), processed
one layer at a time [estimates, to be confirmed on box]: 16K ~3 GB, 32K ~6 GB,
64K ~12 GB, 128K ~24 GB, 256K ~48 GB. With the chunked loop these collapse to under
1 GB at any length.

### 4.4 Stage 2 distillation (local, no 35B backprop)

Run the stock model once (no grad) over a small calibration set and cache, per
full-attention layer, the teacher per-block attention mass. Compute the full
attention **per layer, one layer at a time**, pool to `(nblk, nblk, Hkv)`
immediately, and **drop the L x L matrix** so only the block-mass is cached:
`teacher_mass[i, j, h] = mean_{t in i} ( sum_{s in j} attn_weight[t, s, h] )`.
Train each layer's `sel_q`/`sel_k` with KL(student block-scores || teacher
block-mass) for a few hundred steps, training only the selector parameters. No
end-to-end backward through the frozen 35B. Calibrate at a shorter context (4K to
8K; full attention there is ~2 to 4 GB/layer bf16 and transient) then evaluate
perplexity at 16K.

### 4.5 Components

All code authored in `src/`, synced to the DGX flat layout (per AGENTS.md).
- `qwen_blocksparse.py`: GQA-aware chunked block-sparse core + selector (Stage 1/2).
- `patch_qwen.py`: load bf16 model, locate full-attention layers, patch attention.
- `eval_qwen_ppl.py`: held-out perplexity, stock vs sparse vs oracle vs random
  baselines A/B, top-k sweep, chunked loss, mean and spread over 50 to 100
  sequences.
- `distill_qwen_sel.py`: cache teacher block-mass, train selectors, re-eval.
- `scripts/run_qwen_poc.sh`: orchestration in tmux on the DGX.

### 4.6 Data

Held-out English long-context text tokenized with the model's own tokenizer: PG19
or GovReport slices at a few hundred thousand tokens, chosen at implementation time
for license availability and recorded in the results JSON. Standard long-context
sets, so cross-paper comparison is possible.

## 5. Decisions and defaults (locked unless changed at review)

- Model: bf16 `Qwen3.6-35B-A3B`.
- Evaluation context: 16K (default); 32K to 128K as a stretch if the 16K result is
  promising.
- Block size `Bs = 128`; content budget top-k in {4, 8, 16}, plus own block and
  sink block.
- Batch 1, bf16, no-grad for Stage 1.

### 4.7 Wall-clock budget

Chunked attention is slow. Budget ~1 forward second/layer/query-block estimate and
size eval runs accordingly: 50 to 100 sequences at 16K with 10 converted layers is
a run of minutes to low tens of minutes per config. Stage 2 calibration and
training add a few hundred steps at 4K to 8K. Keep all long runs in tmux per
AGENTS.md.


## 6. Risks and mitigations

- Environment (Phase 0): transformers not on host; gate on a loadable model before
  patching.
- Patch correctness (RoPE/GQA/output gate/causal mask): gated by the all-blocks
  correctness check before any sparse number is reported.
- Memory at long context: chunked query loop and chunked LM head; cap at 16K by
  default.
- Custom modeling code: read the `qwen3_5_moe` source; verify shapes against a tiny
  forward before the full run.
- VLM path: feed text-only inputs; confirm no image-token branch is exercised.
- Tokenizer/data mismatch: use the model's own tokenizer for the eval corpus.

## 7. Verification plan
1. Phase 0 env gate: the model loads and a text forward returns finite logits on
   the host (or a documented container fallback) before any patch code is written.
2. Correctness gate: patched all-blocks path reproduces stock perplexity within
   noise.
3. Primary: sparse perplexity delta vs stock at 16K across the top-k sweep, with
   oracle and random baselines.
4. Stage 2: distilled vs training-free perplexity delta at a matched budget.

## 8. Out of scope / future work

- GQA + partial-MRoPE Triton kernel for throughput and very long context.
- Training base weights; full donor conversion with continued pretraining.
- RULER / NIAH / long-context retrieval; generation-time KV-cache decode.
