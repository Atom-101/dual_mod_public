"""fig:lengthgen — C-RASP protocol (DFA state prediction, train lengths 2-50), every arm, six languages.
Best seed at 10x per (arm, language); looped arms at their best evaluation iteration count. Flip-flop = separator protocol."""
import ast, glob, os, re, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
FACT=[1,2,4,6,8,10,16,20,32]
def final(f):
    ls=[l for l in open(f,errors="ignore") if l.startswith("FINAL")]
    if not ls: return None
    l=ls[-1]; b="{"+l.split(": {",1)[1]; b=b.rsplit(" -> ",1)[0].strip() if " -> " in b else b.strip()
    try: return ast.literal_eval(b)
    except Exception: return None
def curve(d):
    """factor -> acc; loops: max over eval K."""
    out={}
    for f in FACT:
        vals=[v for k,v in d.items() if (k==f) or (isinstance(k,str) and re.fullmatch(rf"K\d+_{f}",k))]
        if vals: out[f]=max(vals)
    return out
def best(paths):
    cs=[(curve(d),p) for p in paths for d in [final(p)] if d]
    cs=[(c,p) for c,p in cs if 10 in c]
    return max(cs,key=lambda cp:cp[0][10]) if cs else (None,None)
C="figures/data/crasp"; L="figures/data/looplr_dfa"; F="figures/data/fbt"
langs=[("z2","Z2 (parity)","#7f0000"),("z5","Z5","#d62728"),("a5","A5","#ff7f7f"),("ff","flip-flop $\\Sigma^*b$","#e6550d"),("ca","$\\Sigma^*a\\Sigma^*$","#006d2c"),("cab","$\\Sigma^*ab\\Sigma^*$","#74c476")]
crasp={"z2":"z2","z5":"z5","a5":"a5","ff":"flipflopSEP","ca":"contains_a","cab":"contains_ab"}
def paths(arm,lg):
    n=crasp[lg]
    if arm=="dm":   return glob.glob(f"{C}/dm_{n}_s*.log")+(glob.glob(f"{C}/dm_s*_flipflopSEP32x.log") if lg=="ff" else [])+(glob.glob("figures/data/prung_b300/dfaz2_logn_dm_base_s*.log") if lg=="z2" else [])
    if arm=="dmnope": return glob.glob(f"{C}/dmNoPE_{n}_s*.log")
    if arm=="tpp":  return glob.glob(f"{C}/tpp_{n}_s*.log")+(glob.glob(f"{C}/tpp_s*_flipflopSEP32x.log") if lg=="ff" else [])+(glob.glob(f"{L}/dfaz2_tpp_32x.log") if lg=="z2" else [])
    if arm=="tppnope": return glob.glob(f"{C}/tppNoPE_{n}_s*.log")+(glob.glob(f"{C}/tppNoPE_s*_flipflopSEP32x.log") if lg=="ff" else [])
    if arm=="lstm": return glob.glob(f"{C}/lstm_{n}_s*.log")+(glob.glob(f"{C}/lstm_s*_flipflopSEP32x.log") if lg=="ff" else [])+(glob.glob(f"{L}/dfaz2_lstm_32x.log") if lg=="z2" else [])
    if arm=="fbt":  return {"z2":glob.glob(f"{F}/z2_s*.log")+glob.glob(f"{L}/dfaz2_fbt_32x.log"),"ff":glob.glob(f"{F}/flipflopSEP32x_s*.log")}.get(lg, glob.glob(f"{L}/dfa{lg}_fbt.log"))
    if arm in ("ag","kt","hug"): return glob.glob(f"{L}/dfa{lg}_{arm}_lr1e-4*.log")
arms=[("dm","DM (RoPE)"),("dmnope","DM (NoPE)"),("tpp","Transformer (RoPE)"),("tppnope","Transformer (NoPE)"),
      ("fbt","Full-bandwidth Tr. (RoPE)"),("ag","Looped, iter.-agnostic (RoPE)"),("kt","Looped, $K\\propto$ length (RoPE)"),("hug","Looped, recurrent-depth (RoPE)"),("lstm","LSTM")]
fig=plt.figure(figsize=(15.5,10.5)); gs=fig.add_gridspec(3,8)
axes=[fig.add_subplot(gs[r,c:c+2]) for r in (0,1) for c in (0,2,4,6)]+[fig.add_subplot(gs[2,3:5])]
used={}
for i,(arm,title) in enumerate(arms):
    ax=axes[i]; ax.axvspan(0,50,color="0.9",zorder=0)
    for lg,ln,col in langs:
        c,p=best(paths(arm,lg))
        if not c: continue
        used[(arm,lg)]=os.path.basename(p)
        xs=[50*f for f in sorted(c) if f<=10]; ys=[c[f] for f in sorted(c) if f<=10]
        ax.plot(xs,ys,marker="o",ms=3.5,lw=1.8,color=col,alpha=0.95,ls="-" if lg not in ("ff","cab") else "--",label=ln if i==0 else None)
    ax.set_title(title,fontsize=11); ax.set_ylim(-0.02,1.05); ax.set_xlim(0,520); ax.grid(alpha=0.25); ax.set_xlabel("eval length")
    if i%4==0 or i==8: ax.set_ylabel("state-prediction acc")
axes[0].legend(fontsize=8,loc="lower left",frameon=True)
fig.suptitle("C-RASP protocol (train lengths 2–50, DFA state prediction): greens = in C-RASP, reds/orange = outside (groups, flip-flop). Best seed at 10$\\times$; looped arms at their best iteration count.",fontsize=10.5)
fig.tight_layout(rect=(0,0,1,0.96)); fig.savefig("figures/crasp_lengthgen.png",dpi=170); print("saved")
for k,v in sorted(used.items()): print(k,v)
