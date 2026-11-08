exec(open("core.py").read()); import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
raw=pd.read_pickle("train_raw.pkl"); assert len(raw)==len(tr)
tr["SEQN"]=raw.SEQN.values; tr["preg"]=(raw.RIDEXPRG==1).values*1; tr["RIDRETH3"]=raw.RIDRETH3.values
NORACE=[f for f in FULL if f not in RACE]
DN=os.environ.get("NHANES_DIR","../data")
SETS={"gb_full":FULL,"gb_norace":NORACE,"gb_norace_nouric":[f for f in NORACE if f!="uric"],
      "gb_ehr":[f for f in NORACE if f not in("uric","waist")],"gb_norace_nobp":[f for f in NORACE if f not in("sbp","dbp")]}
R={}
# CV on training only
skf=StratifiedKFold(5,shuffle=True,random_state=11); yv,wv=tr.y.values,tr.w.values
R["cv_auc"]={}; R["cv_auc_weighted"]={}
SIMPLE=["age","female","sbp","bmi","htn_dx"]
for k,cols in list(SETS.items())+[("lr_norace",NORACE),("lr_simple5",SIMPLE)]:
    s=[];sw=[]
    for a,b in skf.split(tr,yv):
        if k.startswith("lr"): p=fit_lr(cols,tr.iloc[a])(tr.iloc[b])
        else: p=fit_gb(cols,tr.iloc[a]).predict_proba(tr.iloc[b][cols])[:,1]
        s.append(roc_auc_score(yv[b],p)); sw.append(roc_auc_score(yv[b],p,sample_weight=wv[b]))
    R["cv_auc"][k]=round(float(np.mean(s)),4); R["cv_auc_weighted"][k]=round(float(np.mean(sw)),4)
a_,b_=tr[tr.cycle<2017],tr[tr.cycle==2017]
R["temporal_auc_weighted"]={k:round(float(roc_auc_score(b_.y,fit_gb(c,a_).predict_proba(b_[c])[:,1],sample_weight=b_.w)),4) for k,c in [("gb_norace",NORACE),("gb_full",FULL)]}
for k,cols in SETS.items(): te["p_"+k]=fit_gb(cols).predict_proba(te[cols])[:,1]
te["p_gb_norace_2017"]=fit_gb(NORACE,tr[tr.cycle==2017]).predict_proba(te[NORACE])[:,1]
te["p_lr_norace"]=fit_lr(NORACE)(te); te["p_lr_simple5"]=fit_lr(SIMPLE)(te)
# Bragg-Gresham M1 predictors refit on training
def bgX(d,age,male,re,smk,hba1c,htn,cvd):
    X=pd.DataFrame(index=d.index)
    X["a25"]=(age<25)*1;X["a65"]=((age>=65)&(age<75))*1;X["a75"]=(age>=75)*1;X["male"]=male
    X["black"]=(re==4)*1;X["mex"]=(re==1)*1;X["oh"]=(re==2)*1;X["onh"]=re.isin([6,7])*1
    X["smk"]=smk;X["pre"]=((hba1c>=5.7)&(hba1c<6.5))*1;X["htn"]=htn;X["cvd"]=cvd; return X
Btr=bgX(tr,tr.age,1-tr.female,tr.RIDRETH3,tr.smoker,tr.hba1c,tr.htn_dx,tr.cvd)
Bte=bgX(te,te.age,1-te.female,te.race3,te.smoker,te.hba1c,te.htn_dx,te.cvd)
bg=LogisticRegression(max_iter=3000,C=1e6).fit(Btr,tr.y,sample_weight=tr.w); te["p_bg_refit"]=bg.predict_proba(Bte)[:,1]
ts=pd.read_csv("test_scores_2021_2023.csv").set_index("SEQN")
for k in ["p_muntner","p_gb","p_gb2"]: te[k]=ts.loc[te.SEQN,k].values
# RAS blocker sensitivity on training
rx=[]
for f in ["RXQ_RX_G","RXQ_RX_H","RXQ_RX_I","P_RXQ_RX"]:
    r=pd.read_sas(f"{DN}/{f}.xpt",format="xport",encoding="latin1")[["SEQN","RXDDRUG"]]; rx.append(r)
rx=pd.concat(rx); dn=rx.RXDDRUG.fillna("").str.upper()
ras_words=["PRIL","SARTAN"]; ras=rx[dn.str.contains("|".join(ras_words))].SEQN.unique()
tr["ras"]=tr.SEQN.isin(ras)*1
R["ras_share_train"]=round(float((tr.w*tr.ras).sum()/tr.w.sum()),4); R["ras_share_train_htn"]=round(float((tr.w*tr.ras*tr.htn_dx).sum()/(tr.w*tr.htn_dx).sum()),4)
te["p_gb_norace_noras"]=fit_gb(NORACE,tr[tr.ras==0]).predict_proba(te[NORACE])[:,1]
te.to_pickle("test_scored.pkl"); tr.to_pickle("train_aug.pkl")
json.dump(R,open("res_models.json","w"),indent=1); print(R)
