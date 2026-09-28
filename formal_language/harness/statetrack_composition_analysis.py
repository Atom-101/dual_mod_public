"""Partial-composition analysis over stdump artifacts (webui follow-up):
exact accuracy hides how wrong the wrong answers are.

Per position octile and eval factor:
 (a) top-1/5/10 accuracy and mean/median gold rank — graded degradation
     (rank drifts up smoothly) vs walled (rank -> uniform ~n/2 past the
     stall depth).
 (b) correction-element histogram: for wrong argmax s_hat, the correction
     c = s_hat^{-1} . s_gold. If c concentrates at low word-length (BFS
     distance from identity under standard generators), the model tracked
     most of the composition and dropped few factors; if c is uniform,
     state was lost entirely. Metrics: P(wordlen(c) <= 2 | wrong),
     entropy(c)/log(n), top-5 correction mass.

  python formal_language/harness/statetrack_composition_analysis.py --group a5 \
      --dumps analysis/stdump_group_a5_tag_63m_*.npz
"""
import argparse
import glob
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from formal_language.harness.statetrack_gen import BASE, cayley


def inverse_table(group):
    mul, n = cayley(group)
    inv = np.zeros(n, dtype=np.int64)
    for g in range(n):
        inv[g] = int(np.where(mul[g] == 0)[0][0])   # g^-1: g*x = e
    return mul, inv, n


def word_lengths(group, mul, n):
    """BFS distance from identity under standard generators."""
    if group.startswith("z"):
        gens = [1 % n, (n - 1) % n]                 # +/-1
    elif group == "a5":
        perms = sorted(p for p in itertools.permutations(range(5))
                       if sum(1 for i in range(5) for j in range(i + 1, 5)
                              if p[i] > p[j]) % 2 == 0)
        idx = {p: i for i, p in enumerate(perms)}
        gens = [idx[(1, 2, 0, 3, 4)], idx[(2, 0, 1, 3, 4)],   # (012), (021)
                idx[(1, 2, 3, 4, 0)], idx[(4, 0, 1, 2, 3)]]   # 5-cycles
    else:                                           # s3/s4/s5: adjacent swaps
        k = {"s3": 3, "s4": 4, "s5": 5}[group]
        perms = sorted(itertools.permutations(range(k)))
        idx = {p: i for i, p in enumerate(perms)}
        gens = []
        for i in range(k - 1):
            t = list(range(k))
            t[i], t[i + 1] = t[i + 1], t[i]
            gens.append(idx[tuple(t)])
    dist = np.full(n, -1, dtype=np.int64)
    dist[0] = 0
    frontier = [0]
    while frontier:
        nxt = []
        for u in frontier:
            for g in gens:
                v = int(mul[g, u])
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    nxt.append(v)
        frontier = nxt
    assert (dist >= 0).all(), "generators do not generate the group"
    return dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--dumps", nargs="+", required=True)
    args = ap.parse_args()
    mul, inv, n = inverse_table(args.group)
    wl = word_lengths(args.group, mul, n)
    uni_H = np.log(n)

    report = {}
    for f in sorted(sum([glob.glob(p) for p in args.dumps], [])):
        z = np.load(f)
        gold, rank, top10, pos, T = (z["gold"], z["gold_rank"], z["top10"],
                                     z["pos"], int(z["T"]))
        gold_id = gold - (BASE + n)                # state ids
        pred_id = top10[:, 0] - (BASE + n)
        valid = (pred_id >= 0) & (pred_id < n)
        oct_ = np.minimum(7, pos * 8 // T)
        # lag analysis: (row,pos)->gold map; is a wrong prediction the state
        # from k steps earlier? ("carrying the composition, running behind")
        lagged = {}
        if "row" in z:
            gm = {}
            for r_, p_, g_ in zip(z["row"], pos, gold_id):
                gm[(int(r_), int(p_))] = int(g_)
            for k in (1, 2, 3):
                lg = np.full(len(pos), -1, dtype=np.int64)
                for i, (r_, p_) in enumerate(zip(z["row"], pos)):
                    lg[i] = gm.get((int(r_), int(p_) - k), -1)
                lagged[k] = lg
        out = {}
        for o in range(8):
            m = oct_ == o
            if m.sum() == 0:
                continue
            r = rank[m]
            topk = {k: float((r < k).mean()) for k in (1, 5, 10)}
            wrong = m & (rank > 0) & valid
            row = {"n": int(m.sum()),
                   "top1": round(topk[1], 4), "top5": round(topk[5], 4),
                   "top10": round(topk[10], 4),
                   "rank_mean": round(float(r.mean()), 1),
                   "rank_med": int(np.median(r))}
            if wrong.sum() > 20:
                corr = mul[inv[pred_id[wrong]], gold_id[wrong]]
                cnt = np.bincount(corr, minlength=n).astype(float)
                p = cnt / cnt.sum()
                pe = p[p > 0]
                row["corr_wl<=2|wrong"] = round(
                    float((wl[corr] <= 2).mean()), 4)
                row["corr_entropy_frac"] = round(
                    float(-(pe * np.log(pe)).sum() / uni_H), 4)
                row["corr_top5_mass"] = round(
                    float(np.sort(p)[-5:].sum()), 4)
                for k, lg in lagged.items():
                    ok_ = wrong & (lg >= 0)
                    if ok_.sum() > 20:
                        row[f"lag{k}|wrong"] = round(
                            float((pred_id[ok_] == lg[ok_]).mean()), 4)
            out[f"oct{o}"] = row
        report[os.path.basename(f)] = out
    print(json.dumps(report, indent=1))
    outp = f"analysis/composition_{args.group}.json"
    json.dump(report, open(outp, "w"), indent=1)
    print("->", outp)


if __name__ == "__main__":
    main()
