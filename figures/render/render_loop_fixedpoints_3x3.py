"""fig:fixedpoints (3x3) — loop iterate dynamics in- vs out-of-distribution. Rows: A5 (T=64 -> 128), keyed A5 (K=8, D=8 -> 16),
CVP (64 gates -> 128; K∝length: 16 -> 32). Cols: iteration-agnostic, K∝length, recurrent-depth. Solid = relative state change
||x_k-x_{k-1}||/||x_k|| (left axis, log); dashed = accuracy (right axis). Dark = in-distribution input, light = out-of-distribution.
Shaded band = iteration counts seen in training. Traces: figures/data/loop_fixed_point/ood/<row>_<col>_{in,ood}.txt"""
import os, sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D="figures/data/loop_fixed_point/ood"; DEEP="--deepest" in sys.argv
def load(f):
    if not os.path.exists(f): return None
    r=[l.split() for l in open(f) if l.strip() and l.strip()[0].isdigit()]
    k=np.array([int(x[0]) for x in r]); ch=np.array([float(x[1]) if x[1]!="nan" else np.nan for x in r]); acc=np.array([float(x[3 if DEEP and len(x)>3 else 2]) for x in r])
    return k,ch,acc
rows=[("$A_5$, $T{=}64$",  "a5",   "$T{=}128$",   1/60),
      ("keyed $A_5$, $K{=}8$, $D{=}8$", "keyed", "$D{=}16$", 1/60),
      ("CVP, 64 gates",     "cvp",  "128 gates",   0.5)]
cols=[("iteration-agnostic","ag",   {"a5":(4,16),"keyed":(16,64),"cvp":(32,64)}, "#7f7f7f"),
      ("$K\\propto$ length", "kt",   {"a5":(16,16),"keyed":(16,16),"cvp":(16,16)},  "#9467bd"),
      ("recurrent-depth",   "hug",  {"a5":(1,32),"keyed":(1,32),"cvp":(1,32)},   "#ff7f0e")]
fig,axes=plt.subplots(3,3,figsize=(12,8.2),sharex=True)
for i,(rname,rk,oodname,chance) in enumerate(rows):
    for j,(cname,ck,band,col) in enumerate(cols):
        ax=axes[i,j]; ax2=ax.twinx(); lo,hi=band[rk]
        ax.axvspan(lo-0.5 if lo>1 else 0.8,hi,color="0.92",zorder=0)
        for tag,alpha,lw,lab in (("in",1.0,2.0,"in-dist."),("ood",0.45,1.6,oodname)):
            d=load(f"{D}/{rk}_{ck}_{tag}.txt")
            if d is None: ax.text(0.5,0.5,"running",transform=ax.transAxes,ha="center",color="0.5"); continue
            k,ch,acc=d
            ax.plot(k[1:],ch[1:],color=col,alpha=alpha,lw=lw,label=f"state change, {lab}")
            ax2.plot(k,acc,color=col,alpha=alpha,lw=lw,ls="--",label=f"accuracy, {lab}")
        ax2.axhline(chance,color="0.35",lw=1.0,ls=(0,(2,3)),zorder=0)
        ax.set_xscale("log",base=2); ax.set_yscale("log"); ax.set_ylim(2e-3,1.5); ax2.set_ylim(-0.03,1.05); ax.set_xlim(0.9,140)
        ax.set_xticks([1,2,4,8,16,32,64,128]); ax.set_xticklabels(["1","2","4","8","16","32","64","128"],fontsize=8)
        ax.tick_params(labelsize=8); ax2.tick_params(labelsize=8)
        if j: ax.set_yticklabels([])
        if j<2: ax2.set_yticklabels([])
        if i==0: ax.set_title(cname,fontsize=11)
        if j==0: ax.set_ylabel(rname+"\nrel. state change",fontsize=9)
        if j==2: ax2.set_ylabel("accuracy",fontsize=9)
        if i==2: ax.set_xlabel("iteration $k$",fontsize=9)
        ax.grid(alpha=0.2,which="major")
h1=[plt.Line2D([],[],color="0.3",lw=2,label="rel. state change (solid), in-distribution"),
    plt.Line2D([],[],color="0.3",lw=2,ls="--",label=("deepest-position accuracy" if DEEP else "accuracy")+" (dashed), in-distribution"),
    plt.Line2D([],[],color="0.3",alpha=0.45,lw=1.6,label="same, out-of-distribution input (2$\\times$ length / 2$\\times$ depth)"),
    plt.Line2D([],[],color="0.35",lw=1.0,ls=(0,(2,3)),label="chance"),
    plt.Rectangle((0,0),1,1,color="0.92",label="iteration counts used in training")]
fig.legend(handles=h1,loc="lower center",ncol=3,fontsize=8.5,frameon=False,bbox_to_anchor=(0.5,-0.005))
fig.tight_layout(rect=(0,0.06,1,1)); fig.savefig("figures/loop_fixedpoints_3x3"+("_deepest" if DEEP else "")+".png",dpi=170); print("saved")
