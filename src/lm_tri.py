"""Tokenized LM: dense vs block-sparse-Triton. Trains on TinyStories (GPT-2 BPE)
and reports validation perplexity + ppl-vs-context. Uses Triton fused kernel
for training and inference when --use_triton is set."""
import argparse, time, math, json, struct, torch, torch.nn as nn, torch.nn.functional as F
from ssa_model import DenseAttention, BlockSparseSSA, LSHBucketSSA


def load_tokens(path):
    with open(path, "rb") as f:
        n = struct.unpack("I", f.read(4))[0]
        data = struct.unpack(f"{n}i", f.read(4 * n))
    return torch.tensor(data, dtype=torch.long)


class Block(nn.Module):
    def __init__(self, d, h, attn, **kw):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = _make_attn(d, h, attn, kw)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


def _make_attn(d, h, attn, kw):
    if attn == "dense":
        return DenseAttention(d, h, causal=True)
    if attn == "lsh":
        return LSHBucketSSA(d, h, causal=True, **kw)
    return BlockSparseSSA(d, h, causal=True, **kw)


class CausalLM(nn.Module):
    def __init__(self, vocab, d=512, h=8, layers=6, max_len=1024, attn="dense", **kw):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(1, max_len, d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, h, attn, **kw) for _ in range(layers)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        # weight tying
        self.head.weight = self.tok.weight
        self.sparse_attns = [b.attn for b in self.blocks if isinstance(b.attn, BlockSparseSSA)]
        self.lsh_attns = [b.attn for b in self.blocks if isinstance(b.attn, LSHBucketSSA)]

    def forward(self, idx):
        x = self.tok(idx) + self.pos[:, :idx.size(1)]
        for b in self.blocks:
            x = b(x)
        return self.head(self.lnf(x))

    def set_distill(self, flag):
        for a in self.sparse_attns:
            a.distill = flag

    def distill_loss(self):
        ls = []
        for a in self.sparse_attns:
            if a.distill_logits is not None:
                lp = F.log_softmax(a.distill_logits.float(), -1)
                ls.append(-(a.distill_target.float() * lp).sum(-1).mean())
        return sum(ls) / len(ls) if ls else torch.zeros((), device=self.head.weight.device)

    def hash_loss(self):
        ls = [a.last_hash_loss for a in self.lsh_attns if a.last_hash_loss is not None]
        return sum(ls) / len(ls) if ls else torch.zeros((), device=self.head.weight.device)

    def set_dense_select(self, flag):
        for b in self.blocks:
            if hasattr(b.attn, "dense_select"):
                b.attn.dense_select = flag

    def set_lsh_rounds(self, nr, nb, device):
        for b in self.blocks:
            a = b.attn
            if hasattr(a, "R") and not isinstance(a.R, nn.Parameter):
                a.n_rounds, a.n_buckets = nr, nb
                a.R = torch.randn(a.sel_dim, nr, nb, device=device)


def get_batch(data, bs, L, device):
    ix = torch.randint(0, len(data) - L - 1, (bs,))
    x = torch.stack([data[i:i + L] for i in ix]).to(device)
    y = torch.stack([data[i + 1:i + 1 + L] for i in ix]).to(device)
    return x, y


@torch.no_grad()
def evaluate(m, val, bs, L, device, iters=40):
    m.eval()
    tot = 0.0
    for _ in range(iters):
        x, y = get_batch(val, bs, L, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = m(x)
            tot += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
    m.train()
    return tot / iters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--attn", choices=["dense", "sparse", "lsh"], default="dense")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--L", type=int, default=512)
    p.add_argument("--d", type=int, default=512)
    p.add_argument("--h", type=int, default=8)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--block", type=int, default=64)
    p.add_argument("--topk", type=int, default=4)
    p.add_argument("--n_rounds", type=int, default=4, help="LSH hash rounds")
    p.add_argument("--n_buckets", type=int, default=8, help="LSH buckets per round")
    p.add_argument("--cap", type=int, default=8, help="LSH bucket capacity")
    p.add_argument("--learn_hash", action="store_true", help="learned hash planes + alignment loss")
    p.add_argument("--lambda_h", type=float, default=1.0, help="hash-alignment loss weight")
    p.add_argument("--lambda_h_final", type=float, default=None,
                   help="anneal lambda_h linearly to this by the last step (default: constant)")
    p.add_argument("--hash_detach_reps", action="store_true",
                   help="train only R from the hash loss; keep reps for selection")
    p.add_argument("--gate", action="store_true")
    p.add_argument("--use_triton", action="store_true")
    p.add_argument("--data", default="../tinystories.bin")
    p.add_argument("--vocab", type=int, default=50257)
    p.add_argument("--out", default="lm_tri.json")
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_tokens(a.data)
    n_train = int(0.95 * len(data))
    train, val = data[:n_train], data[n_train:]
    print(f"train: {len(train)} val: {len(val)} tokens")

    if a.attn == "lsh":
        kw = dict(block=a.block, topk=a.topk, sel_dim=32, gate=a.gate,
                  n_rounds=a.n_rounds, n_buckets=a.n_buckets, cap=a.cap,
                  learn_hash=a.learn_hash, hash_detach_reps=a.hash_detach_reps)
    else:
        kw = dict(block=a.block, topk=a.topk, sel_dim=32, gate=a.gate, use_triton=a.use_triton)
    m = CausalLM(a.vocab, a.d, a.h, a.layers, max_len=a.L + 8, attn=a.attn, **kw).to(dev)
    if a.attn == "lsh":
        m.set_dense_select(True)   # train dense all-pairs top-k; LSH only at inference
    nparams = sum(p.numel() for p in m.parameters())
    print(f"params: {nparams/1e6:.1f}M")

    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    def lr_at(s):
        if s < a.warmup: return a.lr * (s + 1) / a.warmup
        return a.lr * 0.5 * (1 + math.cos(math.pi * (s - a.warmup) / max(1, a.steps - a.warmup)))

    t0 = time.time()
    for s in range(a.steps):
        for g in opt.param_groups: g["lr"] = lr_at(s)
        x, y = get_batch(train, a.bs, a.L, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            logits = m(x)
            ce = F.cross_entropy(logits.reshape(-1, a.vocab), y.reshape(-1))
            hl = m.hash_loss() if (a.attn == "lsh" and a.learn_hash) else None
            lam_h = a.lambda_h if a.lambda_h_final is None else \
                a.lambda_h + (a.lambda_h_final - a.lambda_h) * (s / max(1, a.steps - 1))
            loss = ce + lam_h * hl if hl is not None else ce
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if s % 500 == 0 or s == a.steps - 1:
            print({"step": s, "ce": round(ce.item(), 4),
                   "hash": round(hl.item(), 4) if hl is not None else None,
                   "lam_h": round(lam_h, 4) if hl is not None else None,
                   "lr": round(lr_at(s), 6), "sec": round(time.time() - t0, 1)}, flush=True)

    def _ppl():
        return round(math.exp(evaluate(m, val, a.bs, a.L, dev)), 3)
    lsh_sweep = None
    if a.attn == "lsh":
        m.set_dense_select(True)
        val_loss = evaluate(m, val, a.bs, a.L, dev)   # dense oracle
        m.set_dense_select(False)                      # LSH inference (linear)
        lsh_sweep = {}
        if a.learn_hash:
            lsh_sweep[f"learned_{a.n_rounds}r{a.n_buckets}b"] = _ppl()
        else:
            for nr, nb in [(4, 64), (8, 64), (16, 64), (32, 64)]:
                m.set_lsh_rounds(nr, nb, dev)
                lsh_sweep[f"{nr}r{nb}b"] = _ppl()
        m.set_dense_select(True)
    else:
        val_loss = evaluate(m, val, a.bs, a.L, dev)
    ppl_by_len = {}
    for el in [128, 256, 512, 1024]:
        if el <= a.L:
            vl = evaluate(m, val, a.bs, el, dev, iters=30)
            ppl_by_len[el] = round(math.exp(vl), 3)
    res = {"attn": a.attn, "use_triton": a.use_triton, "gate": a.gate,
           "learn_hash": a.learn_hash, "lambda_h": a.lambda_h,
           "lambda_h_final": a.lambda_h_final, "hash_detach_reps": a.hash_detach_reps,
           "block": a.block, "topk": a.topk,
           "n_rounds": a.n_rounds, "n_buckets": a.n_buckets, "cap": a.cap,
           "params": nparams, "val_loss": round(val_loss, 4),
           "val_ppl": round(math.exp(val_loss), 3), "ppl_by_len": ppl_by_len,
           "lsh_sweep": lsh_sweep,
           "steps": a.steps, "L": a.L, "d": a.d, "layers": a.layers}
    print("FINAL", json.dumps(res), flush=True)
    json.dump(res, open(a.out, "w"), indent=2)
    torch.save(m.state_dict(), a.out.replace(".json", ".pt"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
