"""Multi-query associative recall (MQAR) — the standard probe for whether a
sub-quadratic mechanism preserves exact long-range retrieval.

Each sequence is a bag of (key,value) pairs placed as ADJACENT tokens at random
positions, followed by a QUERY marker and one queried key. The model must output
the value bound to that key. Because the needle's position depends on content,
fixed-pattern sparsity cannot solve it — the selector must learn to route.
"""
import torch

NK, NV = 64, 64                 # distinct key / value symbols
KEY0 = 2                         # 0=PAD, 1=QUERY
VAL0 = KEY0 + NK
VOCAB = VAL0 + NV
QUERY = 1


def make_batch(batch, n_pairs, seq_len, device):
    assert n_pairs * 2 + 2 <= seq_len
    toks = torch.zeros(batch, seq_len, dtype=torch.long, device=device)
    target = torch.zeros(batch, dtype=torch.long, device=device)
    needle = torch.zeros(batch, dtype=torch.long, device=device)
    n_cells = (seq_len - 2) // 2
    for b in range(batch):
        keys = torch.randperm(NK, device=device)[:n_pairs]
        vals = torch.randint(0, NV, (n_pairs,), device=device)
        cells = torch.randperm(n_cells, device=device)[:n_pairs]
        for i in range(n_pairs):
            p = cells[i].item() * 2
            toks[b, p] = KEY0 + keys[i]
            toks[b, p + 1] = VAL0 + vals[i]
        qi = torch.randint(0, n_pairs, (1,), device=device).item()
        toks[b, seq_len - 2] = QUERY
        toks[b, seq_len - 1] = KEY0 + keys[qi]
        target[b] = vals[qi]
        needle[b] = cells[qi].item() * 2
    return toks, target, needle
