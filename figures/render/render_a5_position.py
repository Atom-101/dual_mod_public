"""fig:a5pos — A5 accuracy by position at T=64 (evaluated to 256), every arm at its best lr."""
import json,ast
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
def from_log(f):
    l=[x for x in open(f,errors="ignore") if x.startswith("FINAL")][-1]
    return ast.literal_eval("{"+l.split(": {",1)[1].rsplit(" -> ",1)[0])
def from_json(f): return json.load(open(f))["final"]
def curve(d):
    g=lambda f:[v for k,v in d.items() if str(k).endswith(f"x{f}_by_pos8")][0]
    return g(1)+g(2)[4:]+g(4)[4:]
xs=[(i+1)*8 for i in range(8)]+[64+(i+1)*16 for i in range(4)]+[128+(i+1)*32 for i in range(4)]
def loop_log(f,pre):   # tuned loops (lr 1e-4): the eval-K sweep is keyed 'K<k>_...'; take one eval K
    d=from_log(f); return {k.replace(pre,""):v for k,v in d.items() if str(k).startswith(pre)}
dag=loop_log("figures/data/looplr/a5_ag_lr1e-4_st24k.log","K32_")          # iteration-agnostic, lr 1e-4, 24k steps, eval K=32
dhug=loop_log("figures/data/looplr/a5_hug_lr1e-4_s1338.log","K64_")        # Huginn recipe, lr 1e-4, seed 1338, eval K=64
series=[
 ("DM, 1 layer, 12M", from_log("figures/data/depth_sweep/dm_a5_L1_s1338.log"), "#d62728"),
 ("DM, 9 layers, 106M", from_json("figures/data/statetrack_group_a5_tag_63m_s50_mult_res_lr0.0004_dualmod.json"), "#ff7f0e"),
 ("Full-bandwidth Tr., 1 layer, 8M", from_log("figures/data/fbt/a5T64_L1_s1337.log"), "#8c564b"),
 ("LSTM, 4 layers, 19M", from_json("figures/data/statetrack_group_a5_tag_63m_lstm.json"), "#1f77b4"),
 ("Transformer++, 9 layers, 64M (lr $10^{-4}$)", from_log("figures/data/lr1e4/a5T64_tpp_s1343.log"), "#000000"),
 ("GDN, 9 layers, 56M", from_json("figures/data/statetrack_group_a5_tag_63m_gdn.json"), "#2ca02c"),
 ("Looped iter.-agnostic, 2 blocks, 59M (24k steps)", dag, "#e377c2"),
 ("Looped recurrent-depth, 2 blocks, 82M", dhug, "#17becf"),
 ("Looped $K\\propto$ length, 2 blocks, 59M (24k steps)", from_log("figures/data/looplr/a5_kt_lr1e-4_st24k.log"), "#9467bd"),
]
fig,ax=plt.subplots(figsize=(13,3.0))
ax.axvspan(0,64,color="0.92",zorder=0); ax.text(2,1.03,"training range (T=64)",fontsize=8.5,color="0.3"); ax.text(66,1.03,"extrapolation",fontsize=8.5,color="0.3")
for n,d,c in series: ax.plot(xs,curve(d),marker="o",ms=3.5,lw=2,color=c,label=n)
ax.axhline(1/60,color="0.5",lw=0.8,ls="--"); ax.text(236,0.035,"chance",fontsize=7.5,color="0.4")
ax.set_xticks(xs); ax.set_xticklabels([str(v) for v in xs],fontsize=8); ax.set_xlim(0,258); ax.set_ylim(-0.02,1.08)
ax.set_xlabel("position (bin end)  =  number of $A_5$ products composed"); ax.set_ylabel("accuracy")
ax.legend(fontsize=8,loc="lower center",bbox_to_anchor=(0.5,1.05),ncol=3,frameon=False); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig("figures/a5_T64_position_curves.png",dpi=170); print("saved")
