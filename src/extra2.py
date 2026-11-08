import pandas as pd, numpy as np, json, warnings; warnings.filterwarnings("ignore")
exec(open("metrics.py").read().split("cols={m")[0])
te=pd.read_pickle("test_scored.pkl"); y=te.y.values; w0=te.w.values; p=te.p_gb_norace.values; h=te.htn.values
big=(te.uacr.values>=200)*1; X={"n_uacr200_unw":int(big.sum())}
strata={s:sorted(g.psu.unique()) for s,g in te.groupby("str")}; key=list(zip(te.str,te.psu)); rng=np.random.default_rng(5)
def rw(n,ws,keys,strata):
    out=[]
    for b in range(n):
        mult={}
        for s,ps in strata.items():
            d=rng.choice(ps,len(ps)-1); 
            for q in ps: mult[(s,q)]=(d==q).sum()*len(ps)/(len(ps)-1)
        out.append(ws*np.array([mult[k] for k in keys]))
    return out
W=rw(2000,w0,key,strata)
def cap(w,yy,sel): return (w*sel*yy).sum()/(w*yy).sum()
def f(w): pct=(w*h).sum()/w.sum(); s=topk(p,w,pct); return cap(w,big,s),cap(w,big,h)
pt=f(w0); bs=np.array([f(w) for w in W])
X["uacr200_capture"]={"model":round(pt[0],4),"htn":round(pt[1],4),"diff":round(pt[0]-pt[1],4),"diff_ci":[round(float(x),4) for x in np.percentile(bs[:,0]-bs[:,1],[2.5,97.5])]}
# union strategy: htn OR model top-k filling same volume? report model top-k among non-htn added to htn: not matched; skip
# Bragg-Gresham on all eGFR 2021-23 (its own spectrum)
a=pd.read_pickle("test_all_egfr.pkl"); age=a.RIDAGEYR; re=a.RIDRETH3; fem=a.RIAGENDR==2
ht=((a.BPQ020==1)|(a.BPQ150==1))*1; cvd=(a[["MCQ160B","MCQ160C","MCQ160D","MCQ160E","MCQ160F"]]==1).any(axis=1)*1
lp=(-3.3723+.522*(age<25)+.2415*((age>=65)&(age<75))+.9294*(age>=75)-.3873*(~fem)+.2489*(re==4)+.5301*(re==1)-.1342*(re==2)+.4185*re.isin([6,7])
    +.2785*a.SMQ040.isin([1,2])+.1561*((a.LBXGH>=5.7)&(a.LBXGH<6.5))+1.4134*ht+.2172*cvd)
pm=1/(1+np.exp(-lp)); yy=a.y.values; ww=a.w.values; pct=(ww*ht).sum()/ww.sum()
X["bg_all_egfr"]=dict(n=len(a),cases=int(yy.sum()),prev=round(float((ww*yy).sum()/ww.sum()),4),auc=round(float(wauc(yy,pm.values,ww)),4),
   htn_pct=round(float(pct),4),bg_capture=round(float(cap(ww,yy,topk(pm.values,ww,pct))),4),htn_capture=round(float(cap(ww,yy,ht.values)),4))
# BG at 37.7% tested in all-eGFR (their reported operating point 73.2%)
X["bg_all_egfr"]["capture_at_37.7"]=round(float(cap(ww,yy,topk(pm.values,ww,.377))),4)
# women/men prevalence trend by cycle
tr=pd.read_pickle("train_aug.pkl"); rows=[]
for cyc,g in list(tr.groupby("cycle"))+[(2021,te.assign(cycle=2021,sdmvpsu=te.psu,sdmvstra=te.str))]:
    for sx in [0,1]:
        q=g[g.female==sx]; rows.append(dict(cycle=int(cyc),female=sx,prev=round(float((q.w*q.y).sum()/q.w.sum()),4),n=len(q),preg=int(q.preg.sum()) if "preg" in q else None))
X["prev_by_cycle_sex"]=rows
# design-based diff women: 2021 vs pooled 2011-2020 using independent Rao-Wu in each
def rwprev(df,psu,stra,n=2000):
    st={s:sorted(g[psu].unique()) for s,g in df.groupby(stra)}; k=list(zip(df[stra],df[psu])); Ws=rw(n,df.w.values,k,st)
    return np.array([(wb*df.y.values).sum()/wb.sum() for wb in Ws])
for sx,name in [(1,"women"),(0,"men")]:
    a1=tr[tr.female==sx].copy(); a1["sk"]=a1.cycle.astype(str)+"_"+a1.sdmvstra.astype(str)
    b1=te[te.female==sx]
    pa=rwprev(a1,"sdmvpsu","sk"); pb=rwprev(b1,"psu","str")
    d=pb-pa; X[f"trend_{name}"]=dict(train=round(float((a1.w*a1.y).sum()/a1.w.sum()),4),test=round(float((b1.w*b1.y).sum()/b1.w.sum()),4),diff_ci=[round(float(x),4) for x in np.percentile(d,[2.5,97.5])])
X["preg_train_n"]=int(tr.preg.sum())
# non-pregnant trend in women
for cyc,g in list(tr.groupby("cycle"))+[(2021,te)]:
    q=g[(g.female==1)&(g.preg==0)]; X.setdefault("women_nonpreg_prev",{})[str(int(cyc))]=round(float((q.w*q.y).sum()/q.w.sum()),4)
# final model on all-eGFR adults without diabetes 2021-23
exec(open("core.py").read()); NORACE=[f for f in FULL if f not in RACE]
cA=a.reset_index(drop=True); XA=test_X(cA); pA=fit_gb(NORACE).predict_proba(XA[NORACE])[:,1]; hA=XA.htn_dx.values; yA=cA.y.values; wA=cA.w.values; qA=(wA*hA).sum()/wA.sum()
X["final_all_egfr"]=dict(n=len(cA),auc=round(float(wauc(yA,pA,wA)),4),model=round(float(cap(wA,yA,topk(pA,wA,qA))),4),htn=round(float(cap(wA,yA,hA)),4),pct=round(float(qA),4))
lo=cA.eGFR.values<60; X["egfr_lt60"]=dict(n=int(lo.sum()),cases=int(yA[lo].sum()))
json.dump(X,open("res_extra2.json","w"),indent=1); print(json.dumps(X,indent=0))
