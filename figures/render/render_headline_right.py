"""Headline figure, right column: CVP deepest exact depth vs n (top) over keyed A5 deepest solved depth vs K (bottom)."""
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
C={"dm":"#d62728","lstm":"#1f77b4","lstm2":"#17becf","tpp":"#000000","gdn":"#2ca02c","lag":"#e377c2","lkt":"#9467bd","fbt":"#ff7f0e","hug":"#8c564b"}
fig,(b,a)=plt.subplots(2,1,figsize=(5.4,8.2))
b.plot([8,1200],[8,1200],color="0.6",lw=0.8,ls="--"); b.text(230,330,"depth = $n$",fontsize=7.5,color="0.4",rotation=40)
cvp=[("dm","DM, 1 layer, 12M",[16,32,64,128,256,512,1024],[16,32,64,128,256,512,1024]),
     ("fbt","Full-bandwidth Tr., 1 layer, 8M",[16,32,64,128,256],[16,32,64,128,256]),
     ("hug","Looped, recurrent-depth, 2 blocks, 30M",[16,32,64,128,256],[16,32,64,128,134]),
     ("lag","Looped, iter.-agnostic, 2 blocks, 21M",[16,32,64,128],[16,32,64,29]),
     ("lstm","LSTM, 4 layers, 19M",[16,32,64,128],[16,32,46,50]),
     ("gdn","GDN, 9 layers, 56M",[16,32,64,128],[16,23,25,26]),
     ("tpp","Transformer++, 9 layers, 64M",[16,32,64,128],[16,19,16,13]),
     ("lkt","Looped, $K\\propto$ length, 2 blocks, 21M",[16,32],[16,1])]
jit={"dm":1.0,"fbt":0.93,"hug":0.87,"lag":0.81,"lstm":1.0,"gdn":1.0,"tpp":1.0,"lkt":1.0}
for k,n,x,y in cvp: b.plot(x,[v*jit[k] for v in y],marker="o",ms=4.5,lw=2,color=C[k],label=n)
b.set_xscale("log",base=2); b.set_yscale("log",base=2); b.set_xticks([16,32,64,128,256,512,1024]); b.set_xticklabels([16,32,64,128,256,512,1024]); b.set_yticks([1,4,16,64,256,1024]); b.set_yticklabels([1,4,16,64,256,1024])
b.set_xlim(12,1400); b.set_ylim(0.6,1500); b.text(13,1.35,"shortcut floor",fontsize=7.5,color="0.3")
b.set_xlabel("circuit size $n$ (gates)"); b.set_ylabel("deepest exact depth"); b.set_title("Circuit value problem (P-complete), each arm at its best lr",fontsize=10)
b.legend(fontsize=7.2,loc="lower right",frameon=False,bbox_to_anchor=(1.0,0.02)); b.grid(alpha=0.25,which="both")
K=[2,8,16,32]
a.plot([2,8,16,32,64,128,256,512,1024],[8.15]*9,marker="o",ms=5,lw=2,color=C["dm"],label="DM (1 layer), 12M")
a.plot(K,[7.25,2,2,1],marker="o",ms=5,lw=2,color=C["lstm2"],label="LSTM (4 layers), 303M")
a.plot(K,[7.1,2,1,1],marker="o",ms=5,lw=2,color=C["lstm"],label="LSTM (4 layers), 53M")
a.plot([2,8,16,32,64,128],[8,8,8,8,8,3],marker="o",ms=5,lw=2,color=C["gdn"],label="GDN (9 layers), 161M")
a.plot(K,[8,8,8,2],marker="o",ms=5,lw=2,color=C["tpp"],label="Transformer++ (9 layers), 176M",zorder=1)
a.plot([2,8,16,32,64,128,256,512],[2,7.85,7.85,7.85,7.85,7.85,7.85,1.07],marker="o",ms=5,lw=2,color=C["hug"],label="Looped, recurrent-depth (2 blocks), 82M")
a.plot([2,8],[7.58,0.93],marker="o",ms=5,lw=2,color=C["fbt"],label="Full-bandwidth Tr. (1 layer), 8M")
a.plot([2,8,16],[7.72,7.72,7.72],marker="o",ms=5,lw=2,color="#bcbd22",label="Full-bandwidth Tr. (9 layers), 180M")
a.plot([2,8],[7.44,1.0],marker="o",ms=5,lw=2,color=C["lkt"],label="Looped, $K\\propto$ length (2 blocks), 59M")
a.plot(K,[0.86,0.86,0.86,0.86],marker="o",ms=5,lw=2,color=C["lag"],label="Looped, iter.-agnostic (2 blocks), 59M")
a.set_xscale("log",base=2); a.set_xticks([2,8,32,128,512,1024]); a.set_xticklabels([2,8,32,128,512,1024]); a.set_yticks([1,2,4,8]); a.set_ylim(0.5,8.8); a.set_xlim(1.6,1300)
a.axhline(8,color="0.6",lw=0.8,ls="--"); a.text(2.1,8.15,"full circuit (depth 8)",fontsize=7.5,color="0.4")
a.text(2.1,1.3,"depth 1 = free retrieval",fontsize=7.5,color="0.4")
a.set_xlabel("interleaved registers $K$"); a.set_ylabel("deepest solved depth"); a.set_title("Keyed $A_5$: $K$ registers, circuit depth 8, each arm at its best lr",fontsize=10)
a.legend(fontsize=6.6,loc="upper center",frameon=False,bbox_to_anchor=(0.5,-0.17),ncol=2,handlelength=1.4,labelspacing=0.35,columnspacing=1.0); a.grid(alpha=0.25)
fig.tight_layout(); fig.savefig("figures/headline_right_keyed_cvp.png",dpi=170); print("saved")
