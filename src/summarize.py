import json, os

def get(f):
    if not os.path.exists(f):
        return None
    return json.load(open(f))

def fmt(d):
    return "acc={:.4f} hit={}".format(d["final_acc"], d["final_hit"]) if d else "(pending)"

print("=== Exp A (capability parity) ===")
for f in ["dense.json", "sparse.json", "dense_cur.json", "dense_t.json",
          "dense_h.json", "sparse_h.json", "sparse_sel.json"]:
    d = get(f)
    print("{:18s} {}".format(f, fmt(d)))

print("\n=== Exp C topk sweep (n_pairs=8): untrained vs trained selector ===")
print("{:>5} {:>22} {:>22}".format("topk", "untrained(acc/hit)", "trained(acc/hit)"))
for k in [1, 2, 4, 8]:
    u = get("sparse_h_topk{}.json".format(k))
    t = get("sparse_sel_topk{}.json".format(k))
    us = "{:.3f}/{}".format(u["final_acc"], u["final_hit"]) if u else "-"
    ts = "{:.3f}/{}".format(t["final_acc"], t["final_hit"]) if t else "-"
    print("{:>5} {:>22} {:>22}".format(k, us, ts))

print("\n=== Harder regimes (trained selector) ===")
for f in ["dense_np16.json", "sparse_sel_np16.json", "sparse_sel_np32.json", "dense_cur.json"]:
    d = get(f)
    print("{:20s} {}".format(f, fmt(d)))

print("\n=== Exp B (bench.csv) ===")
if os.path.exists("bench.csv"):
    print(open("bench.csv").read())
