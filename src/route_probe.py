"""Can the routing subspace be written SUB-QUADRATICALLY?

Follow-up to "When Does Content-Based Routing Work?" (arXiv:2603.20997), which
shows: a learned pairwise top-k router fails (~chance) on raw/recurrent reps but
succeeds (~98%) once ONE full O(N^2) softmax-attention layer has WRITTEN a latent
"routing subspace" into the representations (via value aggregation). They never
test whether that writer must be full attention.

Here we hold the ROUTER fixed (learned pairwise q.k + top-k, supervised by a
routing CE loss to the true needle) and vary only the WRITER:
  none      : identity (their <=2.5% endpoint)
  full      : 1 full softmax self-attn layer, O(N^2)  (their ~98% endpoint)
  linear    : 1 linear-attention layer (global value aggregation), O(N)
  window    : 1 sliding-window softmax layer, O(N*w)  (content-blind long-range)
  blocksparse: 1 content-dependent block-sparse layer, O((N/Bs)^2)

Task: relational distant recall. A bigram key (a,b) -> value c is planted far
away; partial-match distractors near the query share exactly ONE of {a,b}; the
query is the bigram (a,b). Single-token (embedding) matching is therefore
insufficient -> the writer must bind a->b. The binding distance `gap` between a
and b is configurable, so we can map WHEN a cheap writer suffices (local binding)
vs fails (long-range binding). Metric: routing precision = argmax of the router's
scores == the needle position (top-1), and top-8 recall.
"""
import argparse, time, json, math, torch, torch.nn as nn, torch.nn.functional as F

V = 64  # shared vocabulary for ALL tokens (no role-based shortcut)


def make_batch(B, L, device, gap=1, n_distract=6, task="bigram"):
    """Generate a batch for the routing probe.
    task='bigram': bigram (a,b) binding — induction, needs ≥2 layers.
    task='same':   same-symbol distant match — the paper's task, 1 layer suffices.
                   Symbol s at random pos (needle), value at next pos, same s at
                   query pos (end). Distractors: other instances of s.
    """
    x = torch.randint(0, V, (B, L), device=device)
    needle = torch.zeros(B, dtype=torch.long, device=device)
    if task == "same":
        s = torch.randint(0, V, (B,), device=device)
        c = torch.randint(0, V, (B,), device=device)    # unused (no value prediction)
        hi = L - 2
        for i in range(B):
            p = int(torch.randint(0, hi, (1,)))           # needle position
            x[i, p] = s[i]
            needle[i] = p
            for _ in range(n_distract):                   # same-symbol distractors
                dp = int(torch.randint(0, hi, (1,)))
                x[i, dp] = s[i]
            x[i, L - 1] = s[i]                             # query symbol at end
        return x, c, needle
    # bigram task (original)
    a = torch.randint(0, V, (B,), device=device)
    b = torch.randint(0, V, (B,), device=device)
    c = torch.randint(0, V, (B,), device=device)
    hi = L - 2 - gap                                  # placements avoid the query region
    for i in range(B):
        p = int(torch.randint(0, hi, (1,)))          # needle at RANDOM position
        x[i, p] = a[i]; x[i, p + gap] = b[i]         # bigram (a, b) at distance gap
        needle[i] = p + gap                          # routing target = position of b
        for _ in range(n_distract):                  # partial-match distractors, SCATTERED
            dp = int(torch.randint(0, hi, (1,)))
            if torch.rand(1).item() < 0.5:
                x[i, dp] = a[i]; x[i, dp + gap] = int(torch.randint(0, V, (1,)))   # (a, ~b)
            else:
                x[i, dp] = int(torch.randint(0, V, (1,))); x[i, dp + gap] = b[i]   # (~a, b)
        x[i, L - 1 - gap] = a[i]; x[i, L - 1] = b[i]  # query bigram at the end
    return x, c, needle


def heads(t, H):
    B, L, d = t.shape
    return t.view(B, L, H, d // H).transpose(1, 2)


class _Mix(nn.Module):
    """Token mixers; all return (B,L,d). 'none' does no cross-position mixing."""
    def __init__(self, kind, d, h, w):
        super().__init__()
        self.kind, self.h, self.w = kind, h, w
        if kind != "none":
            self.qkv = nn.Linear(d, 3 * d); self.o = nn.Linear(d, d)
        if kind == "blocksparse":
            from ssa_model import BlockSparseSSA
            self.bs = BlockSparseSSA(d, h, block=16, topk=4, sel_dim=32, gate=True)

    def forward(self, x):
        B, L, d = x.shape
        if self.kind == "none":
            return torch.zeros_like(x)
        if self.kind == "blocksparse":
            return self.bs(x)
        q, k, v = (heads(t, self.h) for t in self.qkv(x).chunk(3, -1))
        if self.kind == "full":
            o = F.scaled_dot_product_attention(q, k, v)
        elif self.kind == "window":
            idx = torch.arange(L, device=x.device)
            band = (idx[:, None] - idx[None, :]).abs() > self.w
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=(~band).view(1, 1, L, L))
        elif self.kind == "linear":
            q = F.elu(q) + 1; k = F.elu(k) + 1
            kv = k.transpose(-1, -2) @ v
            num = q @ kv
            den = (q @ k.sum(2, keepdim=True).transpose(-1, -2)) + 1e-6
            o = num / den
        else:
            raise ValueError(self.kind)
        return self.o(o.transpose(1, 2).reshape(B, L, d))


class WriterBlock(nn.Module):
    """A real transformer block: x + mix(LN x); x + FFN(LN x). Only `mix` varies."""
    def __init__(self, kind, d, h, w):
        super().__init__()
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
        self.mix = _Mix(kind, d, h, w)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        x = x + self.mix(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


def make_writer(kind, d, h, w):
    return WriterBlock(kind, d, h, w)


class Probe(nn.Module):
    def __init__(self, kind, d=128, h=4, L=256, sel_dim=64, w=16, gap=1, layers=1,
                 task="bigram"):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.pos = nn.Parameter(torch.randn(1, L + 4, d) * 0.02)
        self.writers = nn.ModuleList([make_writer(kind, d, h, w) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.rq = nn.Linear(d, sel_dim); self.rk = nn.Linear(d, sel_dim)
        self.sel_dim = sel_dim
        self.gap = gap
        self.task = task

    def forward(self, toks):
        B, L = toks.shape
        x = self.emb(toks) + self.pos[:, :L]
        for wr in self.writers:
            x = wr(x)
        x = self.ln(x)
        q = self.rq(x[:, L - 1])                      # query token at last pos
        k = self.rk(x)                                # (B,L,s)
        scores = (q.unsqueeze(1) * k).sum(-1) / math.sqrt(self.sel_dim)   # (B,L)
        if self.task == "same":
            scores[:, L - 1] = float("-inf")          # mask only the query position
        else:
            scores[:, L - 1 - self.gap:] = float("-inf")  # mask the whole query bigram
        return scores


def run(kind, steps, B, L, gap, w, lr, device, layers=1, wd=0.1, warmup=500, task="bigram"):
    m = Probe(kind, L=L, w=w, gap=gap, layers=layers, task=task).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=wd)
    def lr_at(s):
        if s < warmup: return lr * (s + 1) / warmup
        prog = (s - warmup) / max(1, steps - warmup)
        return lr * 0.5 * (1 + math.cos(math.pi * prog))
    amp = device == "cuda"
    best = 0.0
    for s in range(steps):
        for g in opt.param_groups: g["lr"] = lr_at(s)
        x, _, needle = make_batch(B, L, device, gap=gap, task=task)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            scores = m(x)
            loss = F.cross_entropy(scores.float(), needle)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if s % 1000 == 0:
            with torch.no_grad():
                xv, _, nv = make_batch(512, L, device, gap=gap, task=task)
                acc = (m(xv).float().argmax(-1) == nv).float().mean().item()
            best = max(best, acc)
            print(f"    [{kind} L{layers} g{gap} {task}] step {s} loss {loss.item():.3f} routeP {acc:.3f}", flush=True)
    # eval
    m.eval()
    top1 = top8 = n = 0
    with torch.no_grad():
        for _ in range(20):
            x, _, needle = make_batch(1024, L, device, gap=gap, task=task)
            scores = m(x).float()
            top1 += (scores.argmax(-1) == needle).sum().item()
            t8 = scores.topk(8, dim=-1).indices
            top8 += (t8 == needle.unsqueeze(-1)).any(-1).sum().item()
            n += needle.numel()
    return {"writer": kind, "gap": gap, "top1": round(top1 / n, 4), "top8": round(top8 / n, 4)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--writers", default="none,full,linear,window,blocksparse")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--B", type=int, default=64)
    p.add_argument("--L", type=int, default=256)
    p.add_argument("--gap", type=int, default=1)
    p.add_argument("--w", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--task", choices=["bigram", "same"], default="bigram",
                   help="'same' = paper's same-symbol match (1 layer should suffice); "
                        "'bigram' = induction binding (needs ≥2 layers)")
    p.add_argument("--out", default="route_probe.json")
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = []
    print(f"task={a.task} L={a.L} gap={a.gap} window_w={a.w} steps={a.steps}  (chance top1 ~ {1.0/a.L:.3f})")
    print(f"{'writer':>12} {'top1':>8} {'top8':>8}")
    for kind in a.writers.split(","):
        r = run(kind, a.steps, a.B, a.L, a.gap, a.w, a.lr, dev, layers=a.layers, task=a.task)
        res.append(r)
        print(f"{kind:>12} {r['top1']:>8.4f} {r['top8']:>8.4f}", flush=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print("wrote", a.out)
