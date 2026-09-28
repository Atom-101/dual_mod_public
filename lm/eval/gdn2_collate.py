"""Collate the distributed GDN-2 eval jsons for a run against the GDN-2 targets.
  python lm/eval/gdn2_collate.py <run>   (reads analysis/lmeval_<run>_*.json)"""
import glob, json, os, re, sys
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
run = sys.argv[1]
files = sorted(glob.glob(f"{R}/analysis/lmeval_{run}_*.json"), key=os.path.getmtime)
byK = {}          # engine_K (None = exact decode / eager) -> {task: results}
for f in files:
    d = json.load(open(f))
    k = d.get("engine_K")
    for t, r in d["results"].items():
        byK.setdefault(k, {})[t] = r

def g(res, t, m):
    r = res.get(t, {})
    for key in (m, f"{m},none"):
        if key in r:
            return float(r[key])
    return None

# Table 2 columns (metric keys per GDN2_TABLE.md)
T2 = [("Wiki ppl", "wikitext", "word_perplexity"), ("LMB ppl", "lambada_openai", "perplexity"),
      ("LMB acc", "lambada_openai", "acc"), ("PIQA", "piqa", "acc"), ("Hella(n)", "hellaswag", "acc_norm"),
      ("Wino", "winogrande", "acc"), ("ARC-e", "arc_easy", "acc"), ("ARC-c", "arc_challenge", "acc"),
      ("OBQA", "openbookqa", "acc"), ("SIQA", "social_iqa_pq", "acc"), ("BoolQ", "boolq", "acc")]
tbl = open(f"{R}/docs/gdn2_table.md").read().splitlines()
gdn2 = [l for l in tbl if l.startswith("| **Gated DeltaNet-2**")]
print(f"== {run}: {len(files)} result files")
print("Table 2 (acc x100; Avg = mean of 9 acc cols)")
hdr = ["model"] + [c for c, _, _ in T2] + ["Avg"]
print(" | ".join(f"{h:>9s}" for h in hdr))
for k, res in sorted(byK.items(), key=lambda kv: (kv[0] is None, kv[0] or 0)):
    if not any(t in res for _, t, _ in T2):
        continue
    row = []; accs = []
    for c, t, m in T2:
        v = g(res, t, m)
        if v is None: row.append("   -"); continue
        if "ppl" not in c: v *= 100; accs.append(v)
        row.append(f"{v:9.2f}")
    avg = sum(accs) / len(accs) if len(accs) == 9 else float("nan")
    print(" | ".join([f"{('DM K'+str(k)) if k else 'DM exact':>9s}"] + row + [f"{avg:9.2f}"]))
for l in gdn2[:2]:
    cells = [c.strip().strip("*") for c in l.strip("|").split("|")]
    print(" | ".join(f"{c:>9s}" for c in ["GDN-2"] + cells[1:]))
# Table 4 recall + Table 3 RULER: everything on the exact-decode path
res = {}
for k in sorted(byK, key=lambda kk: (kk is None, kk or 0), reverse=True):   # exact first, then K24 wins over K64
    res.update(byK[k])
res.update(byK.get(24, {}))
print("\nTable 4 recall (engine generation; TQA/NQ/DROP = Based doc-grounded protocol)")
for c, t in [("SWDE", "jrt_swde"), ("SQuAD", "jrt_squad"), ("FDA", "jrt_fda"), ("TriviaQA", "jrt_triviaqa"), ("NQ", "jrt_nq"), ("DROP", "jrt_drop"),
             ("(tail-trunc) SWDE", "swde"), ("(tail-trunc) SQuAD", "squad_completion"), ("(tail-trunc) FDA", "fda"), ("(\\n-prompt) TriviaQA", "based_triviaqa"), ("(\\n-prompt) NQ", "based_nq"), ("(\\n-prompt) DROP", "based_drop"),
             ("(closed-book) TriviaQA", "triviaqa"), ("(closed-book) NQ", "nq_open"), ("(closed-book) DROP", "drop")]:
    r = res.get(t)
    if r: print(f"  {c:9s} " + "  ".join(f"{kk}={float(v):.4f}" for kk, v in r.items() if isinstance(v, (int, float)) and "stderr" not in kk))
print("\nTable 3 RULER (engine generation K24)")
for t, r in sorted(res.items()):
    if t.startswith("niah"):
        print(f"  {t:28s} " + "  ".join(f"{kk}={float(v):.4f}" for kk, v in r.items() if isinstance(v, (int, float)) and "stderr" not in kk))
for l in gdn2[2:4]:
    print("  GDN-2:", l[:160])
