"""Phase 0 gate: load Qwen3.6-35B-A3B in bf16 and run a text-only forward."""
import torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer

MP = ("/home/apoc/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/"
      "snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0")

t0 = time.time()
tok = AutoTokenizer.from_pretrained(MP)
model = AutoModelForCausalLM.from_pretrained(MP, dtype=torch.bfloat16, device_map="cuda")
model.eval()
print(f"loaded in {time.time()-t0:.1f}s; params={sum(p.numel() for p in model.parameters())/1e9:.1f}B")

ids = tok("The quick brown fox jumps over the lazy dog.", return_tensors="pt").input_ids.cuda()
with torch.no_grad():
    out = model(ids)
logits = out.logits
print("logits shape", tuple(logits.shape), "finite", torch.isfinite(logits).all().item())
print("max mem GB", round(torch.cuda.max_memory_allocated()/1e9, 1))
# config sanity: which layers are full attention (text config is unwrapped)
lt = model.config.layer_types
full = [i for i, t in enumerate(lt) if t == "full_attention"]
print("full_attention layers:", full, "count", len(full))
