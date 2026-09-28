"""Pool sharded JRT runs: python analysis/merge_shards.py analysis/lmeval_<run>_<step>_<hash>_<TAG>s*of*_<task>_TEXT.jsonl ..."""
import json, sys
hits = n = 0
for f in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f)]
    h = sum(r["metric"].get("contains", 0) >= 0.5 for r in rows)
    print(f"{f}: {h}/{len(rows)} = {100*h/len(rows):.2f}")
    hits += h; n += len(rows)
print(f"POOLED contains = {100*hits/n:.2f}  ({hits}/{n})")
