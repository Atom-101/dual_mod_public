"""fig:cvpdepth — CVP accuracy by circuit depth (in-distribution), one curve per arm at the largest circuit it was trained on,
so each arm's wall is visible. Colours shared with the headline figure. Sources: statetrack jsons / run logs (FINAL line)."""
import json,ast,re,os,glob,numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
def load(path,pre=""):
    if path.endswith(".json"):
        d=json.load(open(path))["final"]
    else:
        l=[x for x in open(path,errors="ignore") if x.startswith("FINAL")][-1]; b="{"+l.split(": {",1)[1]; b=b.rsplit(" -> ",1)[0].strip() if " -> " in b else b.strip(); d=ast.literal_eval(b)
    if pre=="best":   # looped: pick the eval K with the deepest exact prefix
        best=None
        for k,v in d.items():
            if isinstance(k,str) and re.fullmatch(r"K\d+_x1_by_k",k):
                dep=0
                for i in sorted(v, key=int):
                    if v[i]>=0.95: dep=int(i)
                    else: break
                if best is None or dep>best[0]: best=(dep,v)
        b=best[1]
    else: b=d[pre+"x1_by_k"]
    ks=sorted(int(k) for k in b); return ks,[b[k] if k in b else b[str(k)] for k in ks]
A="figures/data/"
series=[("DM, 1 layer, $n{=}1024$",A+"statetrack_cvp_ws_63m_mult_res_L1_tagged_m12n1024_st60k_s1339_warm-cvp_ws_63m_m_dualmod.json","","#d62728",2.0,"-"),
        ("Full-bandwidth Tr., 1 layer, $n{=}256$",A+"lr1e4/fbt_n256_from_n128_fastcur_s2.log","","#ff7f0e",1.6,"-"),
        ("Looped, recurrent-depth, 2 blocks, $n{=}256$",A+"looplr/cvp256_hug_from128_lr1e-4.log","best","#8c564b",1.5,"-"),
        ("Looped, iter.-agnostic, 2 blocks, $n{=}128$",A+"lr1e4/chain_looped_loop2x128r64_n128_from_n64_lr0.0001.log","best","#e377c2",1.5,"-"),
        ("LSTM, 4 layers, $n{=}128$",A+"statetrack_cvp_63m_st30k_warm-cvp_63m_st30_lr0.0002_tagged_m12n128_cont2e-4_lstm.json","","#1f77b4",1.6,"-"),
        ("GDN, 9 layers, $n{=}128$",A+"statetrack_cvp_63m_st30k_warm-cvp_63m_st30_lr0.0001_tagged_m12n128_from64_gdn.json","","#2ca02c",1.6,"-"),
        ("Transformer++, 9 layers, $n{=}64$",A+"statetrack_cvp_63m_st30k_lr0.0001_tagged_m12n64_vanilla.json","","#000000",1.4,"--"),
        ("Looped, $K\\propto$ length, 2 blocks, $n{=}32$",None,"","#9467bd",1.4,"--")]
kt=glob.glob(A+"statetrack_cvp_loop2x*p1*m12n32*looped.json")+glob.glob(A+"lr2e4/cvp32_loop_kt*.log")
if kt: series[-1]=(series[-1][0],kt[0],"best" if kt[0].endswith(".log") else "",*series[-1][3:])
fig,a=plt.subplots(1,1,figsize=(11,4.3))
used=[]
for n,f,pre,c,lw,ls in series:
    if not f or not os.path.exists(f): print("skip",n); continue
    try: ks,v=load(f,pre)
    except Exception as e: print("fail",n,e); continue
    v=np.array(v,float); a.plot(ks,v,color=c,lw=lw,ls=ls,label=n); used.append((n,os.path.basename(f)))
a.axhline(0.95,color="0.55",lw=0.8,ls=(0,(2,3))); a.text(1.05,0.955,"exact (0.95)",fontsize=7.5,color="0.4")
a.axhline(0.5,color="0.6",lw=0.7); a.text(1.05,0.51,"chance",fontsize=7.5,color="0.4")
a.set_xscale("log",base=2); a.set_ylim(0.4,1.02); a.set_xlim(1,1200); a.set_xticks([1,2,4,8,16,32,64,128,256,512,1024]); a.set_xticklabels(["1","2","4","8","16","32","64","128","256","512","1024"])
a.set_xlabel("circuit depth (gate index)"); a.set_ylabel("accuracy at that depth"); a.set_title("CVP: accuracy by circuit depth, in-distribution, each arm at the largest circuit it was trained on",fontsize=10)
a.legend(fontsize=7.8,loc="upper center",bbox_to_anchor=(0.5,-0.2),ncol=4,frameon=False,handlelength=1.6); a.grid(alpha=0.2,which="major")
fig.tight_layout(); fig.savefig("figures/cvp_by_depth.png",dpi=170); print("saved"); [print(u) for u in used]
