#!/usr/bin/env python3
"""CANONICAL N-model UNAAGI results plotter (generalizes the old compare_ctmc_vs_ddpm.py).

Reference output this script must reproduce: ~/unaagi_v03_plots/ (4 SVGs — the gold-standard
example figures). Any future plotting must match that style:
  - proteingym_comparison.svg : 25 ProteinGym assays, each model drawn as a STAR on the SAME
    vertical line per assay, with per-iteration error bars (from spearman_per_iter.csv std),
    overlaid on faint baseline scatter (selected ProteinGym predictors + SaProt).
  - barplot_{all,ncaa,naa}_mutations.svg : CP2/PUMA grouped bars, models solid + literature
    baselines hatched.

Inputs (in --data-dir):
  <model>_modeldata.csv  : columns model,assay,subset(all|naa|ncaa),mean,std  (from spearman_per_iter.csv)
  baselines_pg/<assay>_*/results.csv : ProteinGym baseline table (model,spearmanr_pred,ndcg_pred),
    produced by scripts/aggregate_splits.py --baselines.
"""
import os, glob, argparse, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

_p=argparse.ArgumentParser()
_p.add_argument("--data-dir", default=os.path.expanduser("~/pdb_nglyc_dataset"),
                help="dir holding <model>_modeldata.csv and baselines_pg/")
_p.add_argument("--output-dir", default=os.path.expanduser("~/unaagi_v03_plots"))
_p.add_argument("--models", default="",
                help="comma-separated model tags to plot (must have <tag>_modeldata.csv in --data-dir). "
                     "Default: the v0.3 reference group if present, else auto-discover all.")
_A=_p.parse_args()
DATA=_A.data_dir; OUT=_A.output_dir; os.makedirs(OUT,exist_ok=True)

# fixed colors so a given model keeps its colour across runs; p0203_ctmc is the black reference.
_PALETTE=["#1f77b4","#ff7f0e","#9467bd","#2ca02c","#d62728","#8c564b","#e377c2","#17becf"]
_FIXED={"p0203_ctmc":"#000000"}
def _label(tag): return tag.replace("v0.3","v0.3 ")
def _resolve_models():
    if _A.models.strip():
        tags=[t.strip() for t in _A.models.split(",") if t.strip()]
    else:
        ref=["p0203_ctmc","v0.3ring_base","v0.3ring_cont","v0.3weighted_end"]
        have=lambda t: os.path.isfile(f"{DATA}/{t}_modeldata.csv")
        tags=[t for t in ref if have(t)] or sorted(
            os.path.basename(f)[:-len("_modeldata.csv")]
            for f in glob.glob(f"{DATA}/*_modeldata.csv"))
    out=[]; ci=0
    for t in tags:
        if t in _FIXED: c=_FIXED[t]
        else: c=_PALETTE[ci % len(_PALETTE)]; ci+=1
        out.append((t,_label(t),c))
    return out
MODELS=_resolve_models()
assert MODELS, f"no <tag>_modeldata.csv found in {DATA}"
# sort key for proteingym uses the first model as reference
_REF=MODELS[0][0]
# ---- baseline constants (verbatim from compare_ctmc_vs_ddpm.py) ----
SAPROT={"SBI_STAAM":0.62,"VRPI_BPT7":0.662,"ARGR_ECOLI":0.604,"HCP_LAMBD":0.768,"FKBP3_HUMAN":0.581,"OTU7A_HUMAN":0.642,"RS15_GEOSE":0.433,"SQSTM_MOUSE":0.682,"PKN1_HUMAN":0.324,"SCIN_STAAR":0.62,"ENV_HV1B9":0.15,"DLG4_RAT":0.504,"SUMO1_HUMAN":0.479,"ILF3_HUMAN":0.319,"DN7A_SACS2":0.556,"VG08_BPP22":0.619,"A0A247D711_LISMN":0.427,"SOX30_HUMAN":0.333,"IF1_ECOLI":0.616,"B2L11_HUMAN":0.242,"CCDB_ECOLI":0.438,"AICDA_HUMAN":0.257,"TAT_HV1BR":0.157,"ENVZ_ECOLI":0.148,"ERBB2_HUMAN":0.525}
SELECTED=['MSA_Transformer_ensemble','ESM2_15B','Progen2_xlarge','ProtGPT2','Tranception_L','MIFST','ESM-IF1','ProteinMPNN']
BCOL={'MSA_Transformer_ensemble':'#1f77b4','ESM2_15B':'#ff7f0e','Progen2_xlarge':'#2ca02c','ProtGPT2':'#d62728','Tranception_L':'#9467bd','MIFST':'#8c564b','ESM-IF1':'#e377c2','ProteinMPNN':'#17becf','SaProt(650M)':'#bcbd22'}
BAR={"all":{("CP2","PepINVENT"):-0.0339,("PUMA","PepINVENT"):0.1554,("CP2","NCFlow(AEV-PLIG)"):-0.08,("PUMA","NCFlow(AEV-PLIG)"):0.19,("CP2","NCFlow(ATM)"):0.15,("CP2","Rosetta A"):0.277,("PUMA","Rosetta A"):0.184,("CP2","Rosetta B"):0.283,("PUMA","Rosetta B"):0.313,("CP2","Rosetta C"):0.114,("PUMA","Rosetta C"):0.211},
"ncaa":{("CP2","PepINVENT"):0.0984,("PUMA","PepINVENT"):0.1644,("CP2","Rosetta A"):0.382,("PUMA","Rosetta A"):0.064,("CP2","Rosetta C"):0.114,("PUMA","Rosetta C"):0.211},
"naa":{("CP2","PepINVENT"):-0.0339,("PUMA","PepINVENT"):0.1554,("CP2","Rosetta A"):0.298,("PUMA","Rosetta A"):0.189,("CP2","Rosetta B"):0.283,("PUMA","Rosetta B"):0.313}}
BARCOL={"PepINVENT":"#457B9D","NCFlow(AEV-PLIG)":"#2A9D8F","NCFlow(ATM)":"#F4A261","Rosetta A":"#6A4C93","Rosetta B":"#1982C4","Rosetta C":"#8AC926"}

# ---- model data (precomputed) ----
data={}
for m,_,_ in MODELS:
    d=pd.read_csv(f"{DATA}/{m}_modeldata.csv"); data[m]=d
def pg(m):   # proteingym {assay:(mean,std)}
    d=data[m]; d=d[(d.subset=="all")&(~d.assay.isin(["CP2","PUMA"]))]
    return {r.assay:(r["mean"],r["std"]) for _,r in d.iterrows()}
def uaa(m,assay,subset):
    d=data[m]; r=d[(d.assay==assay)&(d.subset==subset)]
    return (float(r["mean"].values[0]),float(r["std"].values[0])) if len(r) else (np.nan,0)

# ---- baseline_df from ProteinGym results.csv ----
brows=[]
for f in glob.glob(f"{DATA}/baselines_pg/*/results.csv"):
    assay=os.path.basename(os.path.dirname(f)).split("_v0.3")[0]
    for _,r in pd.read_csv(f).iterrows(): brows.append({"benchmark_name":assay,"model":r["model"],"spearmanr_pred":r["spearmanr_pred"]})
bdf=pd.DataFrame(brows)

# ---- Plot 1: proteingym ----
def plot_pg():
    pgs={m:pg(m) for m,_,_ in MODELS}
    assays=sorted(pgs[_REF], key=lambda b: pgs[_REF][b][0], reverse=True)
    x=np.arange(len(assays))
    plt.rcParams.update({"font.size":15,"font.weight":"bold","axes.titleweight":"bold"})
    fig,ax=plt.subplots(figsize=(22,10))
    for bm in SELECTED:
        bd=bdf[bdf.model==bm].set_index("benchmark_name")["spearmanr_pred"]
        ax.scatter(x,[bd.get(b,np.nan) for b in assays],s=70,alpha=.40,color=BCOL.get(bm,"#999"),edgecolor="black",lw=.5,label=bm,zorder=1)
    ax.scatter(x,[SAPROT.get(b,np.nan) for b in assays],s=70,alpha=.40,color=BCOL["SaProt(650M)"],edgecolor="black",lw=.5,label="SaProt(650M)",zorder=1)
    offs=np.zeros(len(MODELS))  # all models on the same vertical line per assay
    for (m,lab,c),xo in zip(MODELS,offs):
        y=[pgs[m].get(b,(np.nan,0))[0] for b in assays]; e=[pgs[m].get(b,(np.nan,0))[1] for b in assays]
        ax.errorbar(x+xo,y,yerr=e,fmt="*",ms=11,color=c,ecolor="black",elinewidth=1.3,capsize=4,capthick=1.3,markeredgecolor="black",markeredgewidth=0.8,label=lab,zorder=4)
    ax.set_xticks(x); ax.set_xticklabels(assays,rotation=75,ha="right"); ax.set_ylabel(r"Spearman $\rho$")
    ax.set_ylim(-0.35,0.95); ax.axhline(0,color="gray",ls="--",lw=1,alpha=.5); ax.grid(axis="y",ls=":",alpha=.4)
    ax.legend(loc="center left",bbox_to_anchor=(1.01,0.5),fontsize=11)
    plt.tight_layout(); plt.savefig(f"{OUT}/proteingym_comparison.svg",format="svg",bbox_inches="tight"); plt.savefig(f"{OUT}/proteingym_comparison.png",dpi=140,bbox_inches="tight"); plt.close(); print("wrote proteingym_comparison.svg")

# ---- Plots 2-4: barplots ----
def barplot(subset,fname,title):
    rows=[]
    for m,lab,_ in MODELS:
        for bm in ["CP2","PUMA"]: v=uaa(m,bm,subset); rows.append({"benchmark":bm,"model":lab,"spr":v[0],"std":v[1]})
    for (bm,bl),spr in BAR[subset].items(): rows.append({"benchmark":bm,"model":bl,"spr":spr,"std":0})
    df=pd.DataFrame(rows); models=list(dict.fromkeys(df.model))
    cmap={lab:c for _,lab,c in MODELS}; cmap.update(BARCOL)
    x=np.arange(2); n=len(models); w=0.8/n
    plt.rcParams.update({"font.size":14,"font.weight":"bold","axes.titleweight":"bold"})
    fig,ax=plt.subplots(figsize=(14,8)); ax.set_facecolor("#f8f9fa")
    for i,mo in enumerate(models):
        pos=x+(i-n/2+0.5)*w; vals=[df[(df.benchmark==b)&(df.model==mo)]["spr"].values[0] if len(df[(df.benchmark==b)&(df.model==mo)]) else np.nan for b in ["CP2","PUMA"]]
        is_model=mo in [l for _,l,_ in MODELS]
        ax.bar(pos,vals,w,label=mo,color=cmap.get(mo,"#bbb"),edgecolor="white",lw=1.2,hatch=(None if is_model else "//"),alpha=.92)
    ax.set_xticks(x); ax.set_xticklabels(["CP2","PUMA"]); ax.set_ylabel(r"Spearman $\rho$"); ax.set_title(title)
    ax.axhline(0,color="gray",lw=.8); ax.legend(loc="center left",bbox_to_anchor=(1.01,0.5),fontsize=11)
    plt.tight_layout(); plt.savefig(f"{OUT}/{fname}",format="svg",bbox_inches="tight"); plt.savefig(f"{OUT}/{fname.replace(chr(46)+chr(115)+chr(118)+chr(103),chr(46)+chr(112)+chr(110)+chr(103))}",dpi=140,bbox_inches="tight"); plt.close(); print("wrote",fname)

plot_pg()
barplot("all","barplot_all_mutations.svg","Protein Fitness Prediction — All Mutations")
barplot("ncaa","barplot_ncaa_mutations.svg","Protein Fitness Prediction — NCAA Mutations Only")
barplot("naa","barplot_naa_mutations.svg","Protein Fitness Prediction — NAA Mutations Only")
print("OUT:",OUT)
