"""TinyStories ablation (single seed each, gate-init 0 for DM arms): val loss vs tokens."""
import json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
runs=[("Transformer++, width-matched (d=512)","figures/data/tinystories/A-vanilla/metrics.jsonl","#000000"),
      ("Transformer++, param-matched (d=576)","figures/data/tinystories/E-vanilla576/metrics.jsonl","#7f7f7f"),
      ("v-mod only","figures/data/tinystories/TS0-C-ddp-s2001/metrics.jsonl","#1f77b4"),
      ("k-mod only","figures/data/tinystories/TS0-D-ddp-s2001/metrics.jsonl","#2ca02c"),
      ("DM (k-mod + v-mod)","figures/data/tinystories/B-ginit0/metrics.jsonl","#d62728")]
fig,ax=plt.subplots(figsize=(7.2,4.6))
ins=ax.inset_axes([0.40,0.40,0.57,0.55])
for n,f,c in runs:
    rows=[json.loads(l) for l in open(f)]; pts=[(r["tokens"]/1e6,r["val/loss"]) for r in rows if "val/loss" in r]
    xs,ys=zip(*pts); ax.plot(xs,ys,color=c,lw=1.8,marker="o",ms=2.5,label=f"{n}  ({ys[-1]:.3f})")
    ins.plot(xs,ys,color=c,lw=1.6,marker="o",ms=2.5)
ax.set_xlabel("tokens (M)"); ax.set_ylabel("validation loss"); ax.set_xlim(0,305); ax.set_ylim(1.6,3.3); ax.grid(alpha=0.25)
ins.set_xlim(145,305); ins.set_ylim(1.66,1.90); ins.grid(alpha=0.25); ins.tick_params(labelsize=7); ins.set_title("last half",fontsize=8)
ax.legend(fontsize=8,frameon=False,loc="upper center",bbox_to_anchor=(0.5,-0.16),ncol=3); fig.tight_layout(); fig.savefig("figures/tinystories_ablation.png",dpi=170); print("saved")
for n,f,c in runs:
    rows=[json.loads(l) for l in open(f)]; v=[r["val/loss"] for r in rows if "val/loss" in r]; print(f"{n}: final {v[-1]:.4f} min {min(v):.4f} n_evals {len(v)}")
