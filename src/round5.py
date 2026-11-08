"""Round 5 checks: (1) 16-input EHR model file (no uric acid, no waist); (2) testing volume matched within each sex;
(3) 2009-2010 persistent cases scored by a version without pulse and by the 16-input model;
(4) exploratory: sex-matched volume on 2009-2010. All of these were run after the main results and are reported as exploratory."""
import pandas as pd,numpy as np,json,warnings;warnings.filterwarnings("ignore")
exec(open("core.py").read())
NORACE=[f for f in FULL if f not in RACE]; EHR=[f for f in NORACE if f not in("uric","waist")]; NOPULSE=[f for f in NORACE if f!="pulse"]
te=pd.read_pickle("test_scored.pkl"); y=te.y.values; w0=te.w.values; p=te.p_gb_norace.values; h=te.htn.values; fem=te.female.values==1
def topk(pp,ww,pct,mask=None):
    m=np.ones(len(pp),bool) if mask is None else mask
    s=np.zeros(len(pp)); idx=np.where(m)[0]; o=idx[np.argsort(-pp[idx],kind="stable")]; cw=np.cumsum(ww[o])/ww[m].sum(); s[o[cw<=pct]]=1; return s
cap=lambda ww,yy,s,m=None:(ww*s*yy*(1 if m is None else m)).sum()/(ww*yy*(1 if m is None else m)).sum()
def wauc(yy,pp,ww):
    o=np.argsort(-pp,kind="stable"); yy,ww=yy[o],ww[o]; tp=np.cumsum(ww*yy); fp=np.cumsum(ww*(1-yy)); tp/=tp[-1]; fp/=fp[-1]; return np.trapezoid(np.r_[0,tp],np.r_[0,fp])
strata={s:sorted(g.psu.unique()) for s,g in te.groupby("str")}; key=list(zip(te.str,te.psu)); rng=np.random.default_rng(55); W=[]
for b in range(1000):
    mult={}
    for s,ps in strata.items():
        d=rng.choice(ps,len(ps)-1)
        for q in ps: mult[(s,q)]=(d==q).sum()*len(ps)/(len(ps)-1)
    W.append(w0*np.array([mult[k] for k in key]))
ci=lambda v:[round(float(np.percentile(v,2.5)),4),round(float(np.percentile(v,97.5)),4)]
O={}
# (1) EHR model file
m_ehr=fit_gb(EHR); m_ehr.booster_.save_model("model_ehr_16inputs.txt")  # copied to ../model/; pe=m_ehr.predict_proba(te[EHR])[:,1]
assert np.allclose(pe,te.p_gb_ehr.values)
O["ehr_model"]=dict(inputs=EHR,auc=round(wauc(y,pe,w0),4),found=round(cap(w0,y,topk(pe,w0,(w0*h).sum()/w0.sum())),4))
# (2) sex-matched volume: within each sex, test the same share as the rule tests in that sex
def sexmatch(ww,pp=p):
    s=np.zeros(len(y))
    for g in (fem,~fem): s+=topk(pp,ww,(ww*h*g).sum()/(ww*g).sum(),g)
    return s
def stats(ww,s):
    return dict(found=cap(ww,y,s),women=cap(ww,y,s,fem),men=cap(ww,y,s,~fem),sel_w=(ww*s*fem).sum()/(ww*fem).sum(),sel_m=(ww*s*~fem).sum()/(ww*~fem).sum(),vol=(ww*s).sum()/ww.sum())
pt=stats(w0,sexmatch(w0)); rule=stats(w0,h); pool=stats(w0,topk(p,w0,(w0*h).sum()/w0.sum()))
bs=[(lambda s,r:(s["found"]-r["found"],s["men"]-r["men"],s["women"]-r["women"]))(stats(w,sexmatch(w)),stats(w,h)) for w in W]; bs=np.array(bs)
O["sex_matched"]=dict(model={k:round(float(v),4) for k,v in pt.items()},rule={k:round(float(v),4) for k,v in rule.items()},pooled_model={k:round(float(v),4) for k,v in pool.items()},
  gain=round(pt["found"]-rule["found"],4),gain_ci=ci(bs[:,0]),gain_men=round(pt["men"]-rule["men"],4),gain_men_ci=ci(bs[:,1]),gain_women=round(pt["women"]-rule["women"],4),gain_women_ci=ci(bs[:,2]))
# (3) 2009-2010 persistence with pulse-free and EHR versions
exec(open("build_raw.py").read().split("def train_raw")[0]); from build_features import train_X
f={n:f"{DN}/{n}_F.xpt" for n in names}; f["BPX"]=f"{DN}/BPX_F.xpt"
a=build(f,"WTMEC2YR",1,"2009").reset_index(drop=True); a["RIDRETH3"]=a.RIDRETH1.replace({5:7}); X9=train_X(a,3)
y9=a.y.values; w9=a.w.values; h9=X9.htn_dx.values
a["uacr2"]=100*a.URXUMA2/a.URXUCR2; has2=a.uacr2.notna().values; pers=((y9==1)&(a.uacr2.values>=30)).astype(int)
import lightgbm as lgb
P9={"main (18 inputs)":lgb.Booster(model_file="../model/model_main_racefree.txt").predict(X9[NORACE]),
    "no pulse (17 inputs)":fit_gb(NOPULSE).predict_proba(X9[NOPULSE])[:,1],"EHR (16 inputs)":m_ehr.predict_proba(X9[EHR])[:,1]}
key9=list(zip(a.SDMVSTRA,a.SDMVPSU)); st9={s:sorted(g.SDMVPSU.unique()) for s,g in a.groupby("SDMVSTRA")}; rng=np.random.default_rng(9); W9=[]
for b in range(2000):
    mult={}
    for s,ps in st9.items():
        d=rng.choice(ps,len(ps)-1)
        for q in ps: mult[(s,q)]=(d==q).sum()*len(ps)/(len(ps)-1)
    W9.append(w9*np.array([mult[k] for k in key9]))
def pr(ww,pp):
    wm=ww*has2; pct=(wm*h9).sum()/wm.sum(); s=topk(np.where(has2,pp,-1),wm,pct); s1=topk(pp,ww,(ww*h9).sum()/ww.sum())
    return cap(wm,pers,s),cap(wm,pers,h9*has2),cap(ww,y9,s1),cap(ww,y9,h9)
O["persist_2009"]={}
for k,pp in P9.items():
    v=pr(w9,pp); b=np.array([pr(w,pp) for w in W9])
    O["persist_2009"][k]=dict(persistent_model=round(v[0],4),persistent_rule=round(v[1],4),diff=round(v[0]-v[1],4),diff_ci=ci(b[:,0]-b[:,1]),single_model=round(v[2],4),single_rule=round(v[3],4),single_diff_ci=ci(b[:,2]-b[:,3]),n_persistent=int(pers[has2].sum()))
# (4) exploratory: sex-matched volume on 2009-2010 (single sample and persistent)
fem9=X9.female.values==1; p9=P9["main (18 inputs)"]
def sm9(ww,pp,mask=None):
    m=np.ones(len(pp),bool) if mask is None else mask; s=np.zeros(len(pp))
    for g in (fem9&m,~fem9&m): s+=topk(np.where(g,pp,-1),ww,(ww*h9*g).sum()/(ww*g).sum(),g)
    return s
def r9(ww):
    s=sm9(ww,p9); wm=ww*has2; s2=sm9(wm,p9,has2)
    return cap(ww,y9,s),cap(ww,y9,h9),cap(wm,pers,s2),cap(wm,pers,h9*has2)
v=r9(w9); b=np.array([r9(w) for w in W9])
O["sex_matched_2009"]=dict(single_model=round(v[0],4),single_rule=round(v[1],4),single_diff_ci=ci(b[:,0]-b[:,1]),persistent_model=round(v[2],4),persistent_rule=round(v[3],4),persistent_diff_ci=ci(b[:,2]-b[:,3]))
json.dump(O,open("res_round5.json","w"),indent=1); print(json.dumps(O,indent=1))
