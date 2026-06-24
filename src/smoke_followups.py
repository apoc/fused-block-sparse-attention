"""Tokenizer-only smoke test for run_followups: confirms imports and that the NIAH
answer span tokenizes consistently (so teacher-forced greedy match is well-defined).
Loads only the tokenizer, not the 35B model."""
from transformers import AutoTokenizer
from patch_qwen import MODEL_PATH
import run_followups as R


def sub(a, b):
    return any(b[i:i + len(a)] == a for i in range(len(b) - len(a) + 1))


tok = AutoTokenizer.from_pretrained(MODEL_PATH)
ok = True
for v in [4839201, 1000000, 9999999, 5550123]:
    key = "Cinnabar"
    needle = tok(f"\nThe special magic number for {key} is {v}.\n", add_special_tokens=False).input_ids
    quest_str = f"\n\nWhat is the special magic number for {key}? The special magic number for {key} is"
    ans = tok(f" {v}", add_special_tokens=False).input_ids
    # (a) value tokenizes the same inside the needle as standalone (so the model saw these exact tokens)
    in_needle = sub(ans, needle)
    # (b) appending the answer to the question as a joined string preserves the answer tokens at the tail
    #     (i.e. separate-then-concatenate == joint tokenization at the boundary)
    joint = tok(quest_str + f" {v}", add_special_tokens=False).input_ids
    boundary_clean = joint[-len(ans):] == ans
    print(f"v={v} ans={ans} in_needle={in_needle} boundary_clean={boundary_clean}")
    ok = ok and in_needle and boundary_clean

ids, nans = R.build_niah(tok, list(range(70000)), 16384, 0.5, "Cinnabar", 4839201)
tail_matches = ids[0, -nans:].tolist() == tok(" 4839201", add_special_tokens=False).input_ids
print(f"build_niah ids={tuple(ids.shape)} nans={nans} tail_matches_answer={tail_matches}")
print("SMOKE_OK" if (ok and tail_matches) else "SMOKE_FAIL")
