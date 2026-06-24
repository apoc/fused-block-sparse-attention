"""Two follow-ups on the block-sparse Qwen3.6-35B-A3B swap, one model load:

 1. NIAH (needle-in-a-haystack): does training-free content selection preserve
    long-context retrieval, the capability perplexity cannot measure? A magic
    number is inserted at varying depths into long filler text, then the model is
    asked for it. The sparse path is prefill-only (no KV-cache decode), so we score
    retrieval by teacher-forced greedy match on the answer span (one forward per
    sample): the answer tokens are appended and we check whether argmax at each
    preceding position reproduces them. Stock vs SelectMax top-k vs random.

 2. Min/max-feature ablation: the mean-pool learned selector (Stage 2) lost to the
    training-free max heuristic. We retrain a learned selector that sees per-block
    key min/max (the extremes max uses) instead of the mean, and compare. If
    learned-minmax reaches max while learned-mean does not, the pooling was the
    cause (measuring what Section 8 currently only states).

Each phase is wrapped so a failure in one does not abort the other.
"""
import argparse, json, math, types, random, statistics
import torch
import torch.nn.functional as F
import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as _M
from patch_qwen import load_model, patch_attention, unpatch, find_attn_modules
from qwen_blocksparse import SelectMax, SelectRandom, SelectLearned, SelectLearnedMinMax

CORPUS = "eval_corpus_raw.txt"
BS = 128
apply_rope = _M.apply_rotary_pos_emb
KEYS = ["Aldebaran", "Brightwater", "Cinnabar", "Driftwood", "Evermoor", "Foxglove"]


# ===================== 1. NIAH =====================

def build_niah(tok, filler, ctx, depth, key, value):
    """Insert a magic-number needle at `depth` into filler; append the question and
    the answer span. Returns (ids[1,L], n_answer_tokens)."""
    needle = tok(f"\nThe special magic number for {key} is {value}.\n", add_special_tokens=False).input_ids
    quest = tok(f"\n\nWhat is the special magic number for {key}? The special magic number for {key} is",
                add_special_tokens=False).input_ids
    ans = tok(f" {value}", add_special_tokens=False).input_ids
    budget = ctx - len(needle) - len(quest) - len(ans)
    assert budget > 0, f"ctx {ctx} too small"
    hay = filler[:budget]
    pos = int(budget * depth)
    ids = hay[:pos] + needle + hay[pos:] + quest + ans
    return torch.tensor(ids, dtype=torch.long)[None], len(ans)


@torch.no_grad()
def niah_hit(model, ids, nans):
    """Teacher-forced greedy match on the answer span. One forward; lm_head only on
    the answer positions (so 64k fits). Returns (strict_all_tokens, per_token_acc)."""
    L = ids.shape[1]
    hidden = model.model(input_ids=ids.cuda(), use_cache=False)[0]
    pos = list(range(L - nans - 1, L - 1))           # predict answer[i] from logits[pos_i]
    lg = model.lm_head(hidden[:, pos]).float()        # (1, nans, V)
    pred = lg.argmax(-1)[0]                            # (nans,)
    tgt = ids[0, L - nans:L].to(pred.device)
    match = (pred == tgt)
    return bool(match.all().item()), float(match.float().mean().item())


def run_niah(tok, model, a):
    print("\n========== NIAH ==========", flush=True)
    filler = tok(open(CORPUS).read()[5000:], add_special_tokens=False).input_ids
    rng = random.Random(0)
    samples = []                                       # (depth, key, value)
    for d in a.depths:
        for _ in range(a.needles_per_depth):
            samples.append((d, rng.choice(KEYS), rng.randint(1000000, 9999999)))
    out = {}
    for ctx in a.niah_ctxs:
        built = [build_niah(tok, filler, ctx, d, k, v) for (d, k, v) in samples]
        print(f"\n##### NIAH ctx={ctx} ({len(built)} needles) #####", flush=True)

        def eval_sel(name, factory):
            if factory is not None:
                patch_attention(model, factory)
            strict, soft = [], []
            for ids, nans in built:
                s, t = niah_hit(model, ids, nans)
                strict.append(s); soft.append(t)
            if factory is not None:
                unpatch(model)
            acc = sum(strict) / len(strict)
            print(f"  {name:10s} exact={acc:.2f} ({sum(strict)}/{len(strict)})  "
                  f"tok={statistics.mean(soft):.2f}", flush=True)
            return {"exact": acc, "tok": statistics.mean(soft)}

        cell = {"stock": eval_sel("stock", None)}
        for k in a.niah_topks:
            cell[f"max{k}"] = eval_sel(f"max{k}", (lambda li, kk=k: (SelectMax(kk, BS), kk, BS)))
        cell[f"rand{a.niah_rand}"] = eval_sel(
            f"rand{a.niah_rand}", lambda li: (SelectRandom(a.niah_rand, BS), a.niah_rand, BS))
        out[str(ctx)] = cell
        json.dump(out, open("qwen_niah.json", "w"), indent=2)
        print(f"  wrote qwen_niah.json", flush=True)
    return out


# ===================== 2. Min/max ablation =====================

def make_capture(cache):
    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kw):
        ishape = hidden_states.shape[:-1]; hshape = (*ishape, -1, self.head_dim)
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
        kpos = torch.arange(L, device=q.device)
        ablk = torch.zeros(B, Hq, nblk, nblk, device=q.device)
        for i in range(nblk):
            qi = q[:, :, i * BS:(i + 1) * BS, :]
            s = (qi @ kr.transpose(-1, -2)) * self.scaling
            qp = i * BS + torch.arange(qi.shape[2], device=q.device)
            s = s.float().masked_fill(kpos[None, None, None, :] > qp[None, None, :, None], float('-inf'))
            aw = s.softmax(-1)
            ablk[:, :, i, :] = aw.view(B, Hq, qi.shape[2], nblk, BS).sum(-1).mean(2)
        ablk = ablk.view(B, Hkv, grp, nblk, nblk).mean(2)
        qpool = q.view(B, Hkv, grp, nblk, BS, d).mean(2).mean(3)
        kb = k.view(B, Hkv, nblk, BS, d)
        c = cache.setdefault(self.layer_idx, {"q": [], "kpool": [], "kmin": [], "kmax": [], "mass": []})
        c["q"].append(qpool[0].cpu())
        c["kpool"].append(kb.mean(3)[0].cpu())
        c["kmin"].append(kb.min(3).values[0].cpu())
        c["kmax"].append(kb.max(3).values[0].cpu())
        c["mass"].append(ablk[0].cpu())
        out = F.scaled_dot_product_attention(q, kr, vr, is_causal=True, scale=self.scaling)
        out = out.transpose(1, 2).reshape(*ishape, -1).contiguous() * torch.sigmoid(gate)
        return self.o_proj(out), None
    return forward


def train_students(cache, sel_dim, steps, first_li):
    scale = sel_dim ** -0.5
    mean_sel, mm_sel = {}, {}
    for li, c in cache.items():
        Q = torch.stack(c["q"]).cuda().float()
        Kp = torch.stack(c["kpool"]).cuda().float()
        Kmm = torch.cat([torch.stack(c["kmin"]).cuda().float(),
                         torch.stack(c["kmax"]).cuda().float()], dim=-1)
        M = torch.stack(c["mass"]).cuda().float()
        Mn = M.clamp(min=0); Mn = Mn / Mn.sum(-1, keepdim=True).clamp(min=1e-9)
        d = Q.shape[-1]; nblk = Q.shape[2]
        diag = torch.arange(nblk, device="cuda")
        causal = diag[None, :] <= diag[:, None]

        def fit(Kfeat, din):
            Wq = (0.02 * torch.randn(d, sel_dim, device="cuda")).requires_grad_(True)
            Wk = (0.02 * torch.randn(din, sel_dim, device="cuda")).requires_grad_(True)
            opt = torch.optim.Adam([Wq, Wk], lr=1e-2)
            loss = None
            for _ in range(steps):
                score = scale * ((Q @ Wq) @ (Kfeat @ Wk).transpose(-1, -2))
                score = score.masked_fill(~causal, -1e9)
                loss = -(Mn * F.log_softmax(score, -1)).sum(-1).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            return (Wq.detach(), Wk.detach()), float(loss.item())

        mean_sel[li], lm = fit(Kp, d)
        mm_sel[li], lmm = fit(Kmm, 2 * d)
        if li == first_li:
            print(f"  layer {li} final KL: mean {lm:.4f}  minmax {lmm:.4f}", flush=True)
    return mean_sel, mm_sel


def run_ablation(tok, model, a):
    print("\n========== MIN/MAX ABLATION ==========", flush=True)
    toks = tok(open(CORPUS).read()[5000:], return_tensors="pt").input_ids[0]
    mods = find_attn_modules(model)
    first_li = mods[0].layer_idx

    # ---- capture teacher mass + pooled/min/max features on a short calibration set ----
    cache = {}
    fwd = make_capture(cache)
    for m in mods:
        m.forward = types.MethodType(fwd, m)
    with torch.no_grad():
        for i in range(a.calib_nseq):
            w = toks[i * a.calib_ctx:(i + 1) * a.calib_ctx]
            if len(w) < a.calib_ctx:
                break
            model.model(input_ids=w[None].cuda(), use_cache=False)
    for m in mods:
        if "forward" in m.__dict__:
            del m.__dict__["forward"]
    torch.cuda.empty_cache()
    print(f"  captured {len(cache)} layers", flush=True)

    mean_sel, mm_sel = train_students(cache, a.sel_dim, a.steps, first_li)

    # ---- eval ppl at ablation_ctx ----
    wins = [toks[i * a.ablation_ctx:(i + 1) * a.ablation_ctx] for i in range(a.ablation_nseq)]
    wins = [w for w in wins if len(w) == a.ablation_ctx]
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
        if factory is not None:
            patch_attention(model, factory)
        p = ppl()
        if factory is not None:
            unpatch(model)
        print(f"  {name:16s} ppl={p:.4f}", flush=True)
        return p

    res = {"ctx": a.ablation_ctx, "nseq": len(wins), "sel_dim": a.sel_dim,
           "calib_ctx": a.calib_ctx, "steps": a.steps}
    res["stock"] = run("stock", None)
    for k in a.ablation_topks:
        res[f"max{k}"] = run(f"max{k}", (lambda li, kk=k: (SelectMax(kk, BS), kk, BS)))
        res[f"mean{k}"] = run(f"learned_mean{k}",
                              (lambda li, kk=k: (SelectLearned(*mean_sel[li], topk=kk, bs=BS), kk, BS)))
        res[f"minmax{k}"] = run(f"learned_minmax{k}",
                                (lambda li, kk=k: (SelectLearnedMinMax(*mm_sel[li], topk=kk, bs=BS), kk, BS)))
    st = res["stock"]
    print("\n  === learned mean vs minmax vs max (delta vs stock) ===", flush=True)
    for k in a.ablation_topks:
        print(f"  top{k}: max {100*(res[f'max{k}']-st)/st:+.2f}%  "
              f"mean {100*(res[f'mean{k}']-st)/st:+.2f}%  "
              f"minmax {100*(res[f'minmax{k}']-st)/st:+.2f}%", flush=True)
    json.dump(res, open("qwen_minmax_ablation.json", "w"), indent=2)
    print("  wrote qwen_minmax_ablation.json", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--niah_ctxs", default="16384,32768,65536")
    ap.add_argument("--niah_topks", default="8,16,32")
    ap.add_argument("--niah_rand", type=int, default=16)
    ap.add_argument("--depths", default="0.1,0.5,0.9")
    ap.add_argument("--needles_per_depth", type=int, default=2)
    ap.add_argument("--sel_dim", type=int, default=64)
    ap.add_argument("--calib_ctx", type=int, default=4096)
    ap.add_argument("--calib_nseq", type=int, default=4)
    ap.add_argument("--ablation_ctx", type=int, default=16384)
    ap.add_argument("--ablation_nseq", type=int, default=2)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--ablation_topks", default="8,16")
    ap.add_argument("--skip_niah", action="store_true")
    ap.add_argument("--skip_ablation", action="store_true")
    a = ap.parse_args()
    a.depths = [float(x) for x in a.depths.split(",")]
    a.niah_ctxs = [int(x) for x in a.niah_ctxs.split(",")]
    a.niah_topks = [int(x) for x in a.niah_topks.split(",")]
    a.ablation_topks = [int(x) for x in a.ablation_topks.split(",")]

    tok, model = load_model()
    assert len(find_attn_modules(model)) == 10

    if not a.skip_niah:
        try:
            run_niah(tok, model, a)
        except Exception:
            import traceback; traceback.print_exc()
    if not a.skip_ablation:
        try:
            run_ablation(tok, model, a)
        except Exception:
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
