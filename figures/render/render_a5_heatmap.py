"""A5 depth x length heatmap (DM vs Transformer++), d=768, 12k steps; single min-depth frontier (all seeds >= 0.95)."""
import re,glob,ast,numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
g={}
# TPP cells: lr 1e-4 grid (figures/data/lr1e4, 2026-09-23; the lr 4e-4 grid under-trained the Transformer); DM cells: depth_sweep (3 seeds)
files=[(f,"tpp") for f in glob.glob("figures/data/lr1e4/a5T*_L*_tpp.log")]+[(f,"dm") for f in glob.glob("figures/data/depth_sweep/dm_a5T*_L*_s*.log")]
for f,a in files:
    m=re.search(r"a5T(\d+)_L(\d+)",f); T,L=int(m.group(1)),int(m.group(2))
    l=[x for x in open(f,errors="ignore") if x.startswith("FINAL")]
    if l: g.setdefault((a,T,L),[]).append(ast.literal_eval("{"+l[-1].split(": {",1)[1].rsplit(" -> ",1)[0])[1])
Ts=[5,10,15,20]; Ld={"dm":[1,2,3,4],"tpp":[1,2,3,4,6,8,12]}
fig,axes=plt.subplots(1,2,figsize=(10,4.2),gridspec_kw=dict(width_ratios=[4,7]))
for ax,a,title in zip(axes,("dm","tpp"),("DM (3 seeds; formed/seeds)","Transformer++ (lr $10^{-4}$, 1 seed per cell)")):
    Ls=Ld[a]; M=np.full((len(Ts),len(Ls)),np.nan)
    for i,T in enumerate(Ts):
        for j,L in enumerate(Ls):
            v=g.get((a,T,L)); 
            if v: M[i,j]=max(v)
    im=ax.imshow(M,origin="lower",cmap="viridis",vmin=0,vmax=1,aspect="auto")
    for i,T in enumerate(Ts):
        for j,L in enumerate(Ls):
            v=g.get((a,T,L))
            if not v: continue
            s=f"{max(v):.2f}"; n=sum(x>=0.95 for x in v)
            if len(v)>1: s+=f"\n{n}/{len(v)}"
            ax.text(j,i,s,ha="center",va="center",fontsize=8,color="k" if max(v)>0.6 else "w")
    ax.set_xticks(range(len(Ls))); ax.set_xticklabels(Ls); ax.set_yticks(range(len(Ts))); ax.set_yticklabels(Ts)
    ax.set_xlabel("layers $L$"); ax.set_ylabel("sequence length $T$"); ax.set_title(title,fontsize=10)
    if a=="tpp":
        # frontier: min L with ALL seeds >= 0.95 per T; None where no depth qualifies
        fr={}
        for T in Ts:
            ok=[L for L in Ls if g.get((a,T,L)) and all(x>=0.95 for x in g[(a,T,L)])]; fr[T]=min(ok) if ok else None
        xi=lambda L: Ls.index(L); yi=lambda T: Ts.index(T)
        fr2={T:next((L for L in Ls if g.get((a,T,L)) and max(g[(a,T,L)])>=0.95),None) for T in Ts}
        pts=[(fr2[T],T) for T in Ts if fr2[T] is not None]
        ax.plot([xi(L) for L,T in pts],[yi(T) for L,T in pts],color="red",lw=2.2,marker="o",ms=6)
        print("frontier(any seed)",fr2)
        print("frontier",fr)
cb=fig.colorbar(im,ax=axes,fraction=0.025,pad=0.02); cb.set_label("accuracy (in-distribution)")
fig.suptitle("$A_5$ word problem: accuracy over depth × length, $d{=}768$, 12k steps (each arm at its best learning rate)",fontsize=11)
fig.savefig("figures/a5_depth_heatmap.png",dpi=170,bbox_inches="tight"); print("saved")
