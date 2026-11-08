import pandas as pd, numpy as np, json
te=pd.read_pickle("test_scored.pkl")
MODELS=["p_gb_norace","p_gb_full","p_gb_norace_nouric","p_gb_ehr","p_gb_norace_nobp","p_gb_norace_2017","p_gb_norace_noras","p_lr_norace","p_lr_simple5","p_bg_refit","p_m1","p_muntner","p_gb","p_gb2"]
def wauc(y,p,w):
    o=np.argsort(-p,kind="stable"); y,w=y[o],w[o]; tp=np.cumsum(w*y); fp=np.cumsum(w*(1-y)); tp/=tp[-1]; fp/=fp[-1]
    return np.trapezoid(np.r_[0,tp],np.r_[0,fp])
def topk(p,w,pct):
    o=np.argsort(-p,kind="stable"); cw=np.cumsum(w[o])/w.sum(); s=np.zeros(len(p)); s[o[cw<=pct]]=1; return s
def run(y,w,d):
    pct=(w*d["htn"]).sum()/w.sum(); o={"pct":pct,"prev":(w*y).sum()/w.sum(),"htn":(w*d["htn"]*y).sum()/(w*y).sum()}
    for m in MODELS:
        p=d[m]; o[m+"_auc"]=wauc(y,p,w); s=topk(p,w,pct); o[m+"_c"]=(w*s*y).sum()/(w*y).sum()
    return o
cols={m:te[m].values for m in MODELS+["htn"]}; y=te.y.values; w0=te.w.values
point=run(y,w0,cols)
# Rao-Wu rescaling bootstrap: per stratum draw n_h-1 PSUs with replacement, weight x n_h/(n_h-1) x times drawn
strata={s:sorted(g.psu.unique()) for s,g in te.groupby("str")}
key=list(zip(te.str,te.psu)); rng=np.random.default_rng(2026); B=2000; reps=[]
for b in range(B):
    mult={}
    for s,ps in strata.items():
        n=len(ps); draw=rng.choice(ps,n-1,replace=True)
        for p in ps: mult[(s,p)]=(draw==p).sum()*n/(n-1)
    wb=w0*np.array([mult[k] for k in key]); keep=wb>0
    reps.append(run(y[keep],wb[keep],{k:v[keep] for k,v in cols.items()}))
def ci(f): v=np.array([f(r) for r in reps]); return [round(float(np.percentile(v,2.5)),6),round(float(np.percentile(v,97.5)),6)]
out={"point":{k:round(float(v),6) for k,v in point.items()},"B":B,"method":"Rao-Wu rescaling bootstrap, n_h-1 PSUs/stratum, 15 strata x 2 PSUs"}
out["auc_ci"]={m:ci(lambda r,m=m:r[m+"_auc"]) for m in MODELS}
out["caught_ci"]={m:ci(lambda r,m=m:r[m+"_c"]) for m in MODELS}; out["caught_ci"]["htn"]=ci(lambda r:r["htn"])
out["diff_vs_htn"]={m:[round(point[m+"_c"]-point["htn"],6)]+ci(lambda r,m=m:r[m+"_c"]-r["htn"]) for m in MODELS}
out["diff_vs_m1"]={m:[round(point[m+"_c"]-point["p_m1_c"],6)]+ci(lambda r,m=m:r[m+"_c"]-r["p_m1_c"]) for m in MODELS}
out["auc_diff_vs_m1"]={m:[round(point[m+"_auc"]-point["p_m1_auc"],6)]+ci(lambda r,m=m:r[m+"_auc"]-r["p_m1_auc"]) for m in MODELS}
out["main_minus_full_caught"]=[round(point["p_gb_norace_c"]-point["p_gb_full_c"],6)]+ci(lambda r:r["p_gb_norace_c"]-r["p_gb_full_c"])
out["main_minus_lr_caught"]=[round(point["p_gb_norace_c"]-point["p_lr_norace_c"],6)]+ci(lambda r:r["p_gb_norace_c"]-r["p_lr_norace_c"])
out["main_minus_lr_auc"]=[round(point["p_gb_norace_auc"]-point["p_lr_norace_auc"],6)]+ci(lambda r:r["p_gb_norace_auc"]-r["p_lr_norace_auc"])
out["main_minus_bgrefit_caught"]=[round(point["p_gb_norace_c"]-point["p_bg_refit_c"],6)]+ci(lambda r:r["p_gb_norace_c"]-r["p_bg_refit_c"])
out["main_minus_muntner_caught"]=[round(point["p_gb_norace_c"]-point["p_muntner_c"],6)]+ci(lambda r:r["p_gb_norace_c"]-r["p_muntner_c"])
out["main_minus_nobp_caught"]=[round(point["p_gb_norace_c"]-point["p_gb_norace_nobp_c"],6)]+ci(lambda r:r["p_gb_norace_c"]-r["p_gb_norace_nobp_c"])
out["main_minus_simple5_caught"]=[round(point["p_gb_norace_c"]-point["p_lr_simple5_c"],6)]+ci(lambda r:r["p_gb_norace_c"]-r["p_lr_simple5_c"])
out["main_minus_simple5_auc"]=[round(point["p_gb_norace_auc"]-point["p_lr_simple5_auc"],6)]+ci(lambda r:r["p_gb_norace_auc"]-r["p_lr_simple5_auc"])
out["prev_ci"]=ci(lambda r:r["prev"])
# relative gain per 1000 tests
out["per1000"]={m:round(1000*point[m+"_c"]*point["prev"]/point["pct"],1) for m in MODELS}; out["per1000"]["htn"]=round(1000*point["htn"]*point["prev"]/point["pct"],1)
out["rel_gain_main_vs_htn"]=[round(point["p_gb_norace_c"]/point["htn"]-1,6)]+ci(lambda r:r["p_gb_norace_c"]/r["htn"]-1)
# old (naive) bootstrap for comparison
json.dump(out,open("res_metrics.json","w"),indent=1); print(json.dumps(out,indent=0))
