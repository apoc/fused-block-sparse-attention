"""Stage 2: distill per-layer learned block selectors for Qwen3.6-35B-A3B.

One model load:
  1. Capture per full-attention layer the teacher per-block attention mass plus the
     block-pooled post-RoPE q/k, on a short calibration set (no grad).
  2. Train per-layer Wq/Wk (d->sel_dim) with KL(student||teacher), local, no 35B
     backprop.
  3. Re-eval perplexity at 16K: SelectLearned vs SelectMax at matched budget.
"""
import types, math, json, statistics
import torch
import torch.nn.functional as F
import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as _M
from patch_qwen import load_model, patch_attention, unpatch, find_attn_modules
from qwen_blocksparse import SelectMax, SelectLearned

CORPUS = "eval_corpus_raw.txt"
BS = 128
SEL_DIM = 64
CALIB_CTX = 4096
CALIB_NSEQ = 4
EVAL_CTX = 16384
EVAL_NSEQ = 2
STEPS = 300
TOPKS = [8, 16]
apply_rope = _M.apply_rotary_pos_emb


def make_capture(cache):
    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kw):
        ishape = hidden_states.shape[:-1]
        hshape = (*ishape, -1, self.head_dim)
        qy, gate = torch.chunk(self.q_proj(hidden_states).view(*ishape, -1, self.head_dim * 2), 2, -1)
        gate = gate.reshape(*ishape, -1)
        q = self.q_norm(qy.view(hshape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states).view(hshape)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hshape).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rope(q, k, cos, sin)
        B, Hq, L, d = q.shape
        Hkv = k.shape[1]; grp = Hq // Hkv; nblk = (L + BS - 1) // BS
        kr, vr = k.repeat_interleave(grp, 1), v.repeat_interleave(grp, 1)
        # teacher per-block mass, flash-style chunked over query blocks (no L x L matrix)
        kpos = torch.arange(L, device=q.device)
        ablk = torch.zeros(B, Hq, nblk, nblk, device=q.device)
        for i in range(nblk):
            qi = q[:, :, i * BS:(i + 1) * BS, :]                              # (B,Hq,bs,d)
            s = (qi @ kr.transpose(-1, -2)) * self.scaling                    # (B,Hq,bs,L) bf16
            qpos = i * BS + torch.arange(qi.shape[2], device=q.device)
            s = s.float().masked_fill(kpos[None, None, None, :] > qpos[None, None, :, None], float('-inf'))
            a = s.softmax(-1)                                                 # (B,Hq,bs,L)
            ablk[:, :, i, :] = a.view(B, Hq, qi.shape[2], nblk, BS).sum(-1).mean(2)
        ablk = ablk.view(B, Hkv, grp, nblk, nblk).mean(2)                    # (B,Hkv,nblk,nblk)
        qpool = q.view(B, Hkv, grp, nblk, BS, d).mean(2).mean(3)
        kpool = k.view(B, Hkv, nblk, BS, d).mean(3)
        c = cache.setdefault(self.layer_idx, {"q": [], "k": [], "mass": []})
        c["q"].append(qpool[0].cpu()); c["k"].append(kpool[0].cpu()); c["mass"].append(ablk[0].cpu())
        out = F.scaled_dot_product_attention(q, kr, vr, is_causal=True, scale=self.scaling)
        out = out.transpose(1, 2).reshape(*ishape, -1).contiguous() * torch.sigmoid(gate)
        return self.o_proj(out), None
    return forward


def main():
    tok, model = load_model()
    mods = find_attn_modules(model)
    assert len(mods) == 10
    toks = tok(open(CORPUS).read()[5000:], return_tensors="pt").input_ids[0]

    # ---- 1. capture ----
    cache = {}
    fwd = make_capture(cache)
    saved = []
    for m in mods:
        saved.append(m); m.forward = types.MethodType(fwd, m)
    with torch.no_grad():
        for i in range(CALIB_NSEQ):
            w = toks[i * CALIB_CTX:(i + 1) * CALIB_CTX]
            if len(w) < CALIB_CTX:
                break
            model.model(input_ids=w[None].cuda(), use_cache=False)
    for m in saved:
        if "forward" in m.__dict__:
            del m.__dict__["forward"]
    torch.cuda.empty_cache()    # release capture high-water mark before re-eval
    print("captured layers:", sorted(cache.keys()), "windows:", len(cache[mods[0].layer_idx]["q"]), flush=True)

    # ---- 2. train per-layer selectors ----
    scale = SEL_DIM ** -0.5
    selectors = {}
    for li, c in cache.items():
        Q = torch.stack(c["q"]).cuda().float()      # (W,Hkv,nblk,d)
        K = torch.stack(c["k"]).cuda().float()
        M = torch.stack(c["mass"]).cuda().float()
        Mn = M.clamp(min=0)
        Mn = Mn / Mn.sum(-1, keepdim=True).clamp(min=1e-9)
        d = Q.shape[-1]; nblk = Q.shape[2]
        diag = torch.arange(nblk, device="cuda")
        causal = diag[None, :] <= diag[:, None]      # (nblk,nblk) allowed j<=i
        Wq = (0.02 * torch.randn(d, SEL_DIM, device="cuda")).requires_grad_(True)
        Wk = (0.02 * torch.randn(d, SEL_DIM, device="cuda")).requires_grad_(True)
        opt = torch.optim.Adam([Wq, Wk], lr=1e-2)
        for step in range(STEPS):
            score = scale * ((Q @ Wq) @ (K @ Wk).transpose(-1, -2))
            score = score.masked_fill(~causal, -1e9)   # finite: avoid 0*-inf NaN in KL
            loss = -(Mn * F.log_softmax(score, -1)).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        selectors[li] = (Wq.detach(), Wk.detach())
        if li == mods[0].layer_idx:
            print(f"layer {li}: final KL loss {loss.item():.4f}", flush=True)

    # ---- 3. re-eval at EVAL_CTX ----
    wins = [toks[i * EVAL_CTX:(i + 1) * EVAL_CTX] for i in range(EVAL_NSEQ)]
    wins = [w for w in wins if len(w) == EVAL_CTX]
    ids_all = torch.stack(wins).cuda()

    @torch.no_grad()
    def ppl():
        ls = []
        for i in range(ids_all.shape[0]):
            ids = ids_all[i:i + 1]; L = ids.shape[1]
            hidden = model.model(input_ids=ids, use_cache=False)[0]
            tot = nt = 0
            for c0 in range(0, L - 1, 2048):
                e = min(c0 + 2048, L - 1)
                lg = model.lm_head(hidden[:, c0:e]).float()
                tg = ids[:, c0 + 1:e + 1]
                tot += F.cross_entropy(lg.reshape(-1, lg.size(-1)), tg.reshape(-1), reduction="sum").item()
                nt += tg.numel()
            ls.append(tot / nt)
        return math.exp(statistics.mean(ls))

    def run(name, factory):
        patch_attention(model, factory) if factory else None
        p = ppl()
        if factory:
            unpatch(model)
        print(f"{name:16s} ppl={p:.4f}", flush=True)
        return p

    res = {"ctx": EVAL_CTX, "nseq": len(wins), "sel_dim": SEL_DIM, "calib_ctx": CALIB_CTX, "steps": STEPS}
    res["stock"] = run("stock", None)
    for k in TOPKS:
        sm = SelectMax(k, BS)
        res[f"max{k}"] = run(f"max{k}", (lambda li, s=sm, kk=k: (s, kk, BS)))
        res[f"learned{k}"] = run(f"learned{k}",
                                 (lambda li, kk=k: (SelectLearned(*selectors[li], topk=kk, bs=BS), kk, BS)))
    st = res["stock"]
    print("\n=== Stage 2: learned vs training-free (vs stock) ===", flush=True)
    for k in TOPKS:
        print(f"top{k}: max {res[f'max{k}']:.4f} ({100*(res[f'max{k}']-st)/st:+.2f}%)  "
              f"learned {res[f'learned{k}']:.4f} ({100*(res[f'learned{k}']-st)/st:+.2f}%)", flush=True)
    json.dump(res, open("qwen_poc_distill.json", "w"), indent=2)
    print("wrote qwen_poc_distill.json", flush=True)


if __name__ == "__main__":
    main()
