"""Needle-in-a-haystack (NIAH) retrieval test for trained dense vs sparse LM.

Inserts a specific fact ("needle") at a random position in a long context of
TinyStories text, then asks a question about it. Measures retrieval accuracy
at multiple context lengths and needle depths.
"""
import argparse, json, struct, torch, torch.nn.functional as F
from lm_tri import CausalLM, load_tokens

NEEDLE_TEMPLATE = "The magic number is {num}. Remember this number."
QUESTION_TEMPLATE = "What is the magic number? The magic number is"
ANSWER_PREFIX = " The magic number is"


def make_niah_batch(tokens, B, L, device, enc=None):
    """Create a batch of NIAH examples.
    Each example: [context tokens] + [needle tokens] + [more context] + [question]
    The needle is placed at a random depth in the context.
    Returns: (input_ids, answer_num)
    """
    import random
    examples = []
    answers = []
    for _ in range(B):
        # pick a random needle number (1-100)
        num = random.randint(1, 100)
        needle_text = NEEDLE_TEMPLATE.format(num=num)
        # simple tokenization: use char-level since we don't have enc here
        # actually use the GPT-2 tokenizer
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        needle_tokens = enc.encode(needle_text)
        question_tokens = enc.encode(QUESTION_TEMPLATE)

        # pick a random position in the context for the needle
        needle_pos = random.randint(L // 4, 3 * L // 4)
        # build the sequence: context[:needle_pos] + needle + context[needle_pos:L-len(needle)-len(question)] + question
        ctx_len = L - len(needle_tokens) - len(question_tokens)
        start = random.randint(0, len(tokens) - L - 1)
        ctx = tokens[start:start + ctx_len].tolist()
        # insert needle
        seq = ctx[:needle_pos] + needle_tokens + ctx[needle_pos:ctx_len] + question_tokens
        # pad or truncate to L
        if len(seq) < L:
            seq = seq + [0] * (L - len(seq))
        else:
            seq = seq[:L]
        examples.append(seq)
        answers.append(num)
    return torch.tensor(examples, dtype=torch.long, device=device), answers


@torch.no_grad()
def niah_eval(model, tokens, L, device, n_samples=50, batch_size=10):
    """Evaluate NIAH accuracy at context length L."""
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    model.eval()
    correct = 0
    total = 0
    for _ in range(n_samples // batch_size):
        x, answers = make_niah_batch(tokens, batch_size, L, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(x)
        # the answer is at the last position: " The magic number is {num}"
        # check if the model predicts the correct number token
        for i, ans in enumerate(answers):
            # get the last token prediction
            pred_token = logits[i, -1].argmax().item()
            pred_text = enc.decode([pred_token]).strip()
            # check if the prediction contains the number
            try:
                if int(pred_text) == ans:
                    correct += 1
            except ValueError:
                # check if the number appears in top-10 predictions
                top10 = logits[i, -1].topk(10).indices.tolist()
                for t in top10:
                    txt = enc.decode([t]).strip()
                    try:
                        if int(txt) == ans:
                            correct += 1
                            break
                    except ValueError:
                        continue
            total += 1
    return correct / total


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--attn", choices=["dense", "sparse"], default="dense")
    p.add_argument("--use_triton", action="store_true")
    p.add_argument("--gate", action="store_true")
    p.add_argument("--checkpoint", required=True, help="path to .pt checkpoint")
    p.add_argument("--d", type=int, default=512)
    p.add_argument("--h", type=int, default=8)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--block", type=int, default=64)
    p.add_argument("--topk", type=int, default=4)
    p.add_argument("--vocab", type=int, default=50257)
    p.add_argument("--lengths", default="512,1024,2048")
    p.add_argument("--n_samples", type=int, default=50)
    p.add_argument("--out", default="niah.json")
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_tokens("../tinystories.bin")
    n_val = int(0.05 * len(data))
    val_tokens = data[-n_val:]

    kw = dict(block=a.block, topk=a.topk, sel_dim=32, gate=a.gate, use_triton=a.use_triton)
    m = CausalLM(a.vocab, a.d, a.h, a.layers, max_len=520,
                 attn=a.attn, **kw).to(dev)
    m.load_state_dict(torch.load(a.checkpoint, map_location=dev))
    m.eval()

    results = {}
    for L in [int(x) for x in a.lengths.split(",")]:
        acc = niah_eval(m, val_tokens, L, dev, n_samples=a.n_samples)
        results[L] = round(acc, 4)
        print(f"L={L}: NIAH acc={acc:.3f}", flush=True)

    res = {"attn": a.attn, "use_triton": a.use_triton, "niah_acc": results, "n_samples": a.n_samples}
    json.dump(res, open(a.out, "w"), indent=2)
    print("wrote", a.out)
