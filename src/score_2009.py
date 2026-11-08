import pandas as pd,numpy as np,json,warnings,sys;warnings.filterwarnings("ignore")
exec(open("build_raw.py").read().split("def train_raw")[0])
from build_features import train_X
import lightgbm as lgb
D=DN
f={n:f"{D}/{n}_F.xpt" for n in names}; f["BPX"]=f"{D}/BPX_F.xpt"
a=build(f,"WTMEC2YR",1,"2009").reset_index(drop=True)
a["RIDRETH3"]=a.RIDRETH1.replace({5:7})
X=train_X(a,3)
FULL=["age","female","mexam","othhisp","black","asian","other","sbp","dbp","htn_dx","bmi","hba1c","hdl","uric","smoker","cvd","waist","glucose","trig","chol","alt","pulse","chol_dx"]
NORACE=[c for c in FULL if c not in ["mexam","othhisp","black","asian","other"]]
bst=lgb.Booster(model_file="../model/model_main_racefree.txt")  # the frozen published model
assert bst.feature_name()==NORACE, bst.feature_name()
p=bst.predict(X[NORACE]); y=a.y.values; w=a.w.values; h=X.htn_dx.values
# Bragg-Gresham M1
age=a.RIDAGEYR; re_=a.RIDRETH1; fem=a.RIAGENDR==2
lp=(-3.3723+.522*(age<25)+.2415*((age>=65)&(age<75))+.9294*(age>=75)-.3873*(~fem)+.2489*(re_==4)+.5301*(re_==1)-.1342*(re_==2)+.4185*(re_==5)
    +.2785*a.SMQ040.isin([1,2])+.1561*((a.LBXGH>=5.7)&(a.LBXGH<6.5))+1.4134*h+.2172*X.cvd)
pm=(1/(1+np.exp(-lp))).values
# persistent: second sample UACR >=30
a["uacr2"]=100*a.URXUMA2/a.URXUCR2; has2=a.uacr2.notna().values; pers=((y==1)&(a.uacr2.values>=30)).astype(int)
def wauc(yy,pp,ww):
    o=np.argsort(-pp,kind="stable"); yy,ww=yy[o],ww[o]; tp=np.cumsum(ww*yy); fp=np.cumsum(ww*(1-yy)); tp/=tp[-1]; fp/=fp[-1]; return np.trapezoid(np.r_[0,tp],np.r_[0,fp])
def topk(pp,ww,pct):
    o=np.argsort(-pp,kind="stable"); cw=np.cumsum(ww[o])/ww.sum(); s=np.zeros(len(pp)); s[o[cw<=pct]]=1; return s
def cap(ww,yy,s): return (ww*s*yy).sum()/(ww*yy).sum()
def run(ww):
    pct=(ww*h).sum()/ww.sum(); s=topk(p,ww,pct); sm=topk(pm,ww,pct)
    return dict(pct=pct,model=cap(ww,y,s),rule=cap(ww,y,h),m1=cap(ww,y,sm),auc=wauc(y,p,ww),auc_m1=wauc(y,pm,ww),prev=(ww*y).sum()/ww.sum(),
                per_model=1000*(ww*s*y).sum()/(ww*s).sum(),per_rule=1000*(ww*h*y).sum()/(ww*h).sum())
pt=run(w)
key=list(zip(a.SDMVSTRA,a.SDMVPSU)); strata={s:sorted(g.SDMVPSU.unique()) for s,g in a.groupby("SDMVSTRA")}; rng=np.random.default_rng(2009); reps=[]
for b in range(2000):
    mult={}
    for s,ps in strata.items():
        d=rng.choice(ps,len(ps)-1)
        for q in ps: mult[(s,q)]=(d==q).sum()*len(ps)/(len(ps)-1)
    reps.append(run(w*np.array([mult[k] for k in key])))
ci=lambda fn:[round(float(np.percentile([fn(r) for r in reps],q)),4) for q in (2.5,97.5)]
O=dict(n=len(a),cases=int(y.sum()),n_strata=len(strata),point={k:round(float(v),4) for k,v in pt.items()},
  model_ci=ci(lambda r:r["model"]),rule_ci=ci(lambda r:r["rule"]),gain_ci=ci(lambda r:r["model"]-r["rule"]),gain_vs_m1_ci=ci(lambda r:r["model"]-r["m1"]),
  auc_ci=ci(lambda r:r["auc"]),auc_m1_ci=ci(lambda r:r["auc_m1"]),rel_ci=ci(lambda r:r["model"]/r["rule"]-1))
# exploratory persistence (weights: exam weights among those with second sample)
m=has2; ww=w*m; pct=(ww*h).sum()/ww.sum(); s=topk(np.where(m,p,-1),ww,pct)
O["persist"]=dict(n_with2=int(m.sum()),cases_first_among2=int((y*m).sum()),persistent_cases=int(pers[m].sum()),
  confirm_rate=round(float((ww*pers).sum()/(ww*y).sum()),4),model=round(float(cap(ww,pers,s)),4),rule=round(float(cap(ww,pers,h*m)),4))
O["missing"]={c:int(X[c].isna().sum()) for c in NORACE if X[c].isna().sum()>0}
json.dump(O,open("res_2009.json","w"),indent=1); print(json.dumps(O,indent=1))
# CI for persistent comparison
def runp(wb):
    ww=wb*m; pct=(ww*h).sum()/ww.sum(); s=topk(np.where(m,p,-1),ww,pct); return cap(ww,pers,s)-cap(ww,pers,h*m), cap(ww,pers,s), cap(ww,pers,h*m)
vals=[]
rng=np.random.default_rng(7)
for b in range(2000):
    mult={}
    for st,ps in strata.items():
        d=rng.choice(ps,len(ps)-1)
        for q in ps: mult[(st,q)]=(d==q).sum()*len(ps)/(len(ps)-1)
    vals.append(runp(w*np.array([mult[k] for k in key])))
v=np.array(vals); O["persist"]["diff_ci"]=[round(float(x),4) for x in np.percentile(v[:,0],[2.5,97.5])]; O["persist"]["model_ci"]=[round(float(x),4) for x in np.percentile(v[:,1],[2.5,97.5])]; O["persist"]["rule_ci"]=[round(float(x),4) for x in np.percentile(v[:,2],[2.5,97.5])]
O["persist"]["share_tested"]=round(float((w*m*h).sum()/(w*m).sum()),4)
json.dump(O,open("res_2009.json","w"),indent=1); print(O["persist"])
