import argparse, json, time, torch, torch.nn.functional as F
import ssa_data as D
from ssa_model import TinyTransformer

def run(attn, steps, batch, n_pairs, seq_len, d, h, layers, lr, warmup,
        block, topk, sel_dim, device, log_every, curriculum,
        gate=False, route_lambda=0.0, balance_lambda=0.0, route_anneal=False,
        n_centroids=16, cpq=2, cap=4, refine_topk=None, eval_refine_topk=None,
        n_rounds=4, n_buckets=8, lsh_scale=None):
    # Train with refine_topk (or None = no refinement, full bucket).
    kw = dict(block=block, topk=topk, sel_dim=sel_dim, gate=gate)
    if attn == "centroid":
        kw.update(n_centroids=n_centroids, cpq=cpq, cap=cap, refine_topk=refine_topk)
    if attn == "lsh":
        kw.update(n_rounds=n_rounds, n_buckets=n_buckets, cap=cap, scale=lsh_scale)
    m = TinyTransformer(D.VOCAB, D.NV, d=d, h=h, layers=layers, max_len=seq_len+8,
                        attn=attn, **kw).to(device)
    if attn == "lsh":
        m.set_dense_select(True)   # train with dense all-pairs top-k; LSH only at inference
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    def lr_at(s):
        if s < warmup: return lr * (s + 1) / warmup
        return lr * 0.5 * (1 + torch.cos(torch.tensor((s - warmup) / max(1, steps - warmup) * 3.14159)).item())
    def route_at(s):
        if route_lambda <= 0 or not route_anneal: return route_lambda
        half = steps // 2
        if s < half: return route_lambda
        return route_lambda * max(0.0, 1.0 - (s - half) / max(1, steps - half))
    amp = device == "cuda"
    hist, t0 = [], time.time()
    for s in range(steps):
        for g in opt.param_groups: g["lr"] = lr_at(s)
        np_now = (2 + (n_pairs - 2) * s // max(1, steps // 2)) if curriculum else n_pairs
        np_now = min(np_now, n_pairs)
        toks, tgt, npos = D.make_batch(batch, np_now, seq_len, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits = m(toks, needle_pos=npos)
            task_loss = F.cross_entropy(logits, tgt)
        # ---- auxiliary selector losses (train the router jointly) ----
        # NOTE: L_route uses the known needle block (npos // block) as the
        # supervision target — a supervised cheat available only on MQAR (where
        # needle positions are known). For real LMs (llm.py), the selector is
        # trained by gate coupling alone (no route loss) or self-distillation
        # (KL to dense per-block attention mass), which is the legitimate path.
        rl_cur = route_at(s)
        route_loss = torch.zeros((), device=logits.device)
        balance_loss = torch.zeros((), device=logits.device)
        rlogits = m.route_logits()
        if rlogits:
            nb = (npos // block).long()
            if rl_cur > 0:
                route_loss = sum(F.cross_entropy(lg.float(), nb) for lg in rlogits) / len(rlogits)
                # centroid InfoNCE: query block's centroid distribution should
                # match the needle key block's (positive) and differ from all
                # other key blocks (negatives). Differentiable — no argmax.
                aligns = m.centroid_align() if hasattr(m, "centroid_align") else []
                tau = 0.5
                for aq, ak_all in aligns:
                    aq, ak_all = aq.float(), ak_all.float()
                    q_dist = F.softmax(aq / tau, -1)                      # (B,C)
                    k_dist = F.softmax(ak_all / tau, -1)                   # (B,nblk,C)
                    sim = (q_dist.unsqueeze(1) * k_dist).sum(-1) / tau     # (B,nblk)
                    route_loss = route_loss + F.cross_entropy(sim, nb)
            if balance_lambda > 0:
                ent = 0.0
                for lg in rlogits:
                    p = lg.float().softmax(-1).mean(0)              # avg block usage
                    ent = ent + -(p * (p + 1e-9).log()).sum()
                balance_loss = -(ent / len(rlogits))               # minimize -entropy => uniform
        loss = task_loss + rl_cur * route_loss + balance_lambda * balance_loss
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if s % log_every == 0 or s == steps - 1:
            acc = (logits.argmax(-1) == tgt).float().mean().item()
            hit = m.selection_hit()
            rec = {"step": s, "loss": round(task_loss.item(), 4), "acc": round(acc, 4),
                   "hit": None if hit is None else round(hit, 4),
                   "route": round(float(route_loss.detach()), 4),
                   "pairs": np_now, "sec": round(time.time() - t0, 1)}
            hist.append(rec); print(rec, flush=True)
    # final eval at full difficulty — train config
    m.eval()
    toks, tgt, npos = D.make_batch(1024, n_pairs, seq_len, device)
    def _eval():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            lg = m(toks, needle_pos=npos)
            return (lg.argmax(-1) == tgt).float().mean().item(), m.selection_hit()
    acc0, hit0 = _eval()                       # lsh: dense_select=True (oracle for the trained reps)
    acc_lsh, hit_lsh, lsh_sweep = None, None, None
    if attn == "lsh":
        m.set_dense_select(False)              # LSH bucketing at inference (linear)
        acc_lsh, hit_lsh = _eval()
        # inference-rounds/buckets sweep (R unused during dense training -> free to retune; still linear)
        lsh_sweep = {}
        for nr, nb in [(4, 8), (8, 8), (16, 8), (8, 4), (16, 4), (32, 4)]:
            for b in m.blocks:
                at = b["attn"]
                if hasattr(at, "R"):
                    at.n_rounds, at.n_buckets = nr, nb
                    at.R = torch.randn(at.sel_dim, nr, nb, device=device)
            ac, hi = _eval()
            lsh_sweep[f"{nr}r{nb}b"] = [round(ac, 4), None if hi is None else round(hi, 4)]
        m.set_dense_select(True)
    acc1, hit1 = None, None
    if attn == "centroid" and eval_refine_topk is not None:
        for b in m.blocks:
            if hasattr(b["attn"], "refine_topk"):
                b["attn"].refine_topk = eval_refine_topk
        acc1, hit1 = _eval()
    final = {"final_acc": round(acc0, 4), "final_hit": None if hit0 is None else round(hit0, 4),
             "lsh_acc": None if acc_lsh is None else round(acc_lsh, 4),
             "lsh_hit": None if hit_lsh is None else round(hit_lsh, 4),
             "refine_acc": None if acc1 is None else round(acc1, 4),
             "refine_hit": None if hit1 is None else round(hit1, 4),
             "lsh_sweep": lsh_sweep}
    print("FINAL", final, flush=True)
    return {"attn": attn, "gate": gate, "route_lambda": route_lambda,
            "balance_lambda": balance_lambda, "history": hist, **final}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--attn", choices=["dense", "sparse", "centroid", "lsh"], default="sparse")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--n_pairs", type=int, default=32)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--h", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--block", type=int, default=32)
    p.add_argument("--topk", type=int, default=4)
    p.add_argument("--sel_dim", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--curriculum", action="store_true")
    p.add_argument("--gate", action="store_true",
                   help="MoE-style score->logit coupling so the selector gets task gradient")
    p.add_argument("--route_lambda", type=float, default=0.0,
                   help="weight of the auxiliary routing loss (supervise needle block)")
    p.add_argument("--balance_lambda", type=float, default=0.0,
                   help="weight of the load-balancing (entropy) term")
    p.add_argument("--route_anneal", action="store_true",
                   help="linearly anneal route_lambda to 0 over the second half")
    p.add_argument("--n_centroids", type=int, default=16, help="centroid count (centroid attn)")
    p.add_argument("--cpq", type=int, default=2, help="centroids per query block (centroid attn)")
    p.add_argument("--cap", type=int, default=4, help="bucket capacity per centroid (centroid attn)")
    p.add_argument("--n_rounds", type=int, default=4, help="LSH hash rounds (lsh attn)")
    p.add_argument("--n_buckets", type=int, default=8, help="LSH buckets per round (lsh attn)")
    p.add_argument("--lsh_scale", type=float, default=None, help="LSH gate/route cosine temperature (default sqrt(sel_dim))")
    p.add_argument("--refine_topk", type=int, default=None,
                   help="within-bucket refinement during TRAINING (None = no refine)")
    p.add_argument("--eval_refine_topk", type=int, default=None,
                   help="within-bucket refinement at INFERENCE (train-high/eval-low)")
    p.add_argument("--out", default="result.json")
    a = p.parse_args()
    dev = a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu"
    res = run(a.attn, a.steps, a.batch, a.n_pairs, a.seq_len, a.d, a.h, a.layers,
              a.lr, a.warmup, a.block, a.topk, a.sel_dim, dev, a.log_every, a.curriculum,
              gate=a.gate, route_lambda=a.route_lambda,
              balance_lambda=a.balance_lambda,
              route_anneal=a.route_anneal, n_centroids=a.n_centroids, cpq=a.cpq,
              cap=a.cap, refine_topk=a.refine_topk, eval_refine_topk=a.eval_refine_topk,
              n_rounds=a.n_rounds, n_buckets=a.n_buckets, lsh_scale=a.lsh_scale)
    json.dump(res, open(a.out, "w"), indent=2)
    print("wrote", a.out)