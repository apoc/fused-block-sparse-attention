"""Load Qwen3.6-35B-A3B and monkeypatch its full-attention layers to use our
block-sparse core. Reuses Qwen's q/k/v projections, q/k RMSNorm, partial MRoPE,
and output gate; replaces only the softmax-over-all-keys with select+SDPA.

This is an HF transformers patch, not a vllm modification.
"""
import types
import torch
import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as _M
from qwen_blocksparse import blocksparse_forward

ATTN_CLS = "Qwen3_5MoeAttention"
MODEL_PATH = ("/home/apoc/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/"
              "snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0")


def load_model(path=MODEL_PATH, dtype=torch.bfloat16, device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, device_map=device)
    model.eval()
    return tok, model


def find_attn_modules(model):
    """All full-attention modules, ordered by layer index."""
    mods = [m for m in model.modules() if type(m).__name__ == ATTN_CLS]
    mods.sort(key=lambda m: m.layer_idx)
    return mods


def _make_forward(get_selector):
    apply_rope = _M.apply_rotary_pos_emb

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_values=None, **kwargs):
        # past_key_values: transformers passes an (empty) Cache on a normal forward;
        # we run prefill/eval only (use_cache=False), so we ignore it.
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(*input_shape, -1)
        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rope(query_states, key_states, cos, sin)

        selector, topk, bs = get_selector(self.layer_idx)
        attn_output = blocksparse_forward(query_states, key_states, value_states,
                                          selector=selector, topk=topk, bs=bs,
                                          causal=True, scale=self.scaling)
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)
        return attn_output, None

    return forward


def patch_attention(model, get_selector):
    """get_selector(layer_idx) -> (selector_or_None, topk, bs)."""
    mods = find_attn_modules(model)
    saved = []
    fwd = _make_forward(get_selector)
    for m in mods:
        saved.append(m)
        m.forward = types.MethodType(fwd, m)
    model._bs_patched = saved
    return [m.layer_idx for m in mods]


def unpatch(model):
    for m in getattr(model, "_bs_patched", []):
        if "forward" in m.__dict__:
            del m.__dict__["forward"]
    model._bs_patched = []
