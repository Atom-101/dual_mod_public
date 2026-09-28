"""CVP: minimum layers to solve (in-dist acc >= 0.95, cold 30k, d=768) vs circuit size n. DM / GDN / LSTM (+FBT when done).
Open marker + arrow = no depth up to the largest tried reached 0.95 (LSTM to 16 layers, GDN to 18)."""
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(7.0,3.8))
C={"dm":"#d62728","lstm":"#1f77b4","gdn":"#2ca02c","fbt":"#ff7f0e"}
ax.plot([16,32,64,128,256,512,1024],[1]*7,marker="o",ms=5,lw=2,color=C["dm"],label="DM")
ax.plot([16],[2],marker="o",ms=5,lw=2,color=C["lstm"],label="LSTM")
ax.plot([16,32],[2,16],lw=2,ls=":",color=C["lstm"]); 
for n in (32,64,128): ax.plot([n],[16],marker="^",ms=7,mfc="white",color=C["lstm"],ls="none")
ax.plot([32,128],[16,16],lw=2,ls=":",color=C["lstm"])
ax.plot([16],[9],marker="o",ms=5,lw=2,color=C["gdn"],label="GDN")
ax.plot([16,32],[9,18],lw=2,ls=":",color=C["gdn"]); ax.plot([32,64],[18,18],lw=2,ls=":",color=C["gdn"])
for n in (32,64): ax.plot([n],[18],marker="^",ms=7,mfc="white",color=C["gdn"],ls="none")
ax.set_xscale("log",base=2); ax.set_yscale("log",base=2)
ax.set_xticks([16,32,64,128,256,512,1024]); ax.set_xticklabels([16,32,64,128,256,512,1024]); ax.set_yticks([1,2,4,8,16]); ax.set_yticklabels([1,2,4,8,16])
ax.set_xlim(13,1200); ax.set_ylim(0.8,30)
ax.set_xlabel("circuit size $n$ (gates)"); ax.set_ylabel("min. layers to solve (acc $\\geq 0.95$)")
ax.set_title("CVP: minimum depth vs circuit size (each arm at its best lr)",fontsize=10)
ax.text(150,12.6,"$>16$",fontsize=8,color=C["lstm"]); ax.text(70,20.5,"$>18$",fontsize=8,color=C["gdn"])
ax.plot([16,32,64,128,256],[1.08]*5,marker="o",ms=5,lw=2,color=C["fbt"],label="Full-bandwidth Tr.")
ax.plot([16],[9],marker="o",ms=5,lw=2,color="#000000",label="Transformer++"); ax.plot([16,32],[9,9],lw=2,ls=":",color="#000000"); ax.plot([32],[9],marker="^",ms=7,mfc="white",color="#000000",ls="none"); ax.text(36,7.6,"$>9$",fontsize=8,color="#000000")
# looped arms: unique blocks (each applied K times per token), not layers
ax.plot([16,32,64,128],[1.17]*4,marker="s",ms=4.5,lw=2,color="#8c564b",label="Looped, recurrent-depth (blocks)")
ax.plot([128,256],[1.17,2.0],lw=2,ls=":",color="#8c564b"); ax.plot([256],[2.0],marker="^",ms=7,mfc="white",color="#8c564b",ls="none"); ax.text(275,1.75,"$>2$",fontsize=8,color="#8c564b")
ax.plot([16,32,64],[1.26]*3,marker="s",ms=4.5,lw=2,color="#e377c2",label="Looped, iter.-agnostic (blocks)")
ax.plot([64,128],[1.26,2.15],lw=2,ls=":",color="#e377c2"); ax.plot([128],[2.15],marker="^",ms=7,mfc="white",color="#e377c2",ls="none"); ax.text(138,1.9,"$>2$",fontsize=8,color="#e377c2")
ax.plot([16],[2.3],marker="s",ms=4.5,lw=2,color="#9467bd",label="Looped, $K\\propto$ length (blocks)")
ax.plot([16,32],[2.3,2.3],lw=2,ls=":",color="#9467bd"); ax.plot([32],[2.3],marker="^",ms=7,mfc="white",color="#9467bd",ls="none"); ax.text(36,2.45,"$>2$",fontsize=8,color="#9467bd")
ax.legend(fontsize=7.4,loc="center right",frameon=False,handlelength=1.6); ax.grid(alpha=0.25,which="both")
fig.tight_layout(); fig.savefig("figures/cvp_minlayers.png",dpi=170); print("saved")
