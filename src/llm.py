"""Tiny causal char-LM: dense-causal vs block-sparse-causal (with selector
self-distillation). Trains next-token on real text (shakespeare.txt) and reports
validation bits/char + perplexity. Demonstrates the block-sparse mechanism in an
actual LM, where the selector is trained by DISTILLING the dense per-block
attention mass (no needle labels exist)."""
import argparse, time, math, json, torch, torch.nn as nn, torch.nn.functional as F
from ssa_model import DenseAttention, BlockSparseSSA


def load_data(path):
    text = open(path).read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:], len(chars)


class Block(nn.Module):
    def __init__(self, d, h, attn, **kw):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.attn = DenseAttention(d, h, causal=True) if attn == "dense" \
            else BlockSparseSSA(d, h, causal=True, **kw)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class CausalLM(nn.Module):
    def __init__(self, vocab, d=256, h=4, layers=4, max_len=520, attn="dense", **kw):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(1, max_len, d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, h, attn, **kw) for _ in range(layers)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.sparse_attns = [b.attn for b in self.blocks if isinstance(b.attn, BlockSparseSSA)]

    def forward(self, idx):
        x = self.tok(idx) + self.pos[:, : idx.size(1)]
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
                ls.append(-(a.distill_target.float() * lp).sum(-1).mean())  # CE distill (stable)
        return sum(ls) / len(ls) if ls else torch.zeros((), device=self.head.weight.device)


def get_batch(data, bs, L, device):
    ix = torch.randint(0, len(data) - L - 1, (bs,))
    x = torch.stack([data[i:i + L] for i in ix]).to(device)
    y = torch.stack([data[i + 1:i + 1 + L] for i in ix]).to(device)
    return x, y


@torch.no_grad()
def evaluate(m, val, bs, L, device, iters=40):
    m.eval(); m.set_distill(False)
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
    p.add_argument("--attn", choices=["dense", "sparse"], default="dense")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--L", type=int, default=512)
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--h", type=int, default=4)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--block", type=int, default=32)
    p.add_argument("--topk", type=int, default=4)
    p.add_argument("--gate", action="store_true")
    p.add_argument("--distill_lambda", type=float, default=0.0)
    p.add_argument("--out", default="llm.json")
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    train, val, vocab = load_data("../shakespeare.txt")
    kw = dict(block=a.block, topk=a.topk, sel_dim=32, gate=a.gate)
    m = CausalLM(vocab, a.d, a.h, a.layers, max_len=a.L + 8, attn=a.attn, **kw).to(dev)
    if a.distill_lambda > 0:
        m.set_distill(True)
    nparams = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    t0 = time.time()
    for s in range(a.steps):
        x, y = get_batch(train, a.bs, a.L, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            logits = m(x)
            ce = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
        dl = m.distill_loss() if a.distill_lambda > 0 else torch.zeros((), device=dev)
        loss = ce + a.distill_lambda * dl
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if s % 300 == 0 or s == a.steps - 1:
            print({"step": s, "ce": round(ce.item(), 4),
                   "distill": round(float(dl.detach()), 4), "sec": round(time.time() - t0, 1)}, flush=True)
    val_loss = evaluate(m, val, a.bs, a.L, dev)
    # ppl vs context length (does the model exploit longer context?)
    ppl_by_len = {}
    for el in [64, 128, 256, 512]:
        if el <= a.L:
            vl = evaluate(m, val, a.bs, el, dev, iters=30)
            ppl_by_len[el] = round(math.exp(vl), 3)
    res = {"attn": a.attn, "gate": a.gate, "distill_lambda": a.distill_lambda,
           "params": nparams, "val_loss": round(val_loss, 4),
           "val_bpc": round(val_loss / math.log(2), 4), "val_ppl": round(math.exp(val_loss), 3),
           "ppl_by_len": ppl_by_len}
    print("FINAL", res, flush=True)
    json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
