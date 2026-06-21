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
                  n_rounds=a.n_rounds, n_buckets=a.n_buckets, cap=a.cap)
    else:
        kw = dict(block=a.block, topk=a.topk, sel_dim=32, gate=a.gate, use_triton=a.use_triton)
    m = CausalLM(a.vocab, a.d, a.h, a.layers, max_len=a.L + 8, attn=a.attn, **kw).to(dev)
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
        opt.zero_grad(); ce.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if s % 500 == 0 or s == a.steps - 1:
            print({"step": s, "ce": round(ce.item(), 4),
                   "lr": round(lr_at(s), 6), "sec": round(time.time() - t0, 1)}, flush=True)

    val_loss = evaluate(m, val, a.bs, a.L, dev)
    ppl_by_len = {}
    for el in [128, 256, 512, 1024]:
        if el <= a.L:
            vl = evaluate(m, val, a.bs, el, dev, iters=30)
            ppl_by_len[el] = round(math.exp(vl), 3)
    res = {"attn": a.attn, "use_triton": a.use_triton, "gate": a.gate,
           "params": nparams, "val_loss": round(val_loss, 4),
           "val_ppl": round(math.exp(val_loss), 3), "ppl_by_len": ppl_by_len,
           "steps": a.steps, "L": a.L, "d": a.d, "layers": a.layers}
    print("FINAL", json.dumps(res), flush=True)
    json.dump(res, open(a.out, "w"), indent=2)
    torch.save(m.state_dict(), a.out.replace(".json", ".pt"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
