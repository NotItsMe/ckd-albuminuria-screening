import pandas as pd,numpy as np,json,warnings;warnings.filterwarnings("ignore")
exec(open("core.py").read())
from scipy.stats import t as tdist
from sklearn.linear_model import LogisticRegression
NORACE=[f for f in FULL if f not in RACE]
te=pd.read_pickle("test_scored.pkl"); y=te.y.values; w0=te.w.values; p=te.p_gb_norace.values; h=te.htn.values
strata={s:sorted(g.psu.unique()) for s,g in te.groupby("str")}; key=list(zip(te.str,te.psu)); rng=np.random.default_rng(11)
W=[]
for b in range(2000):
    mult={}
    for s,ps in strata.items():
        d=rng.choice(ps,len(ps)-1)
        for q in ps: mult[(s,q)]=(d==q).sum()*len(ps)/(len(ps)-1)
    W.append(w0*np.array([mult[k] for k in key]))
def topk(p,w,pct,mask=None):
    m=np.ones(len(p),bool) if mask is None else mask
    s=np.zeros(len(p)); idx=np.where(m)[0]; o=idx[np.argsort(-p[idx],kind="stable")]; cw=np.cumsum(w[o])/w[m].sum(); s[o[cw<=pct]]=1; return s
def cap(w,yy,s,m=None):
    m=np.ones(len(yy),bool) if m is None else m; return (w*s*yy*m).sum()/(w*yy*m).sum()
ci=lambda v:[round(float(np.percentile(v,2.5)),4),round(float(np.percentile(v,97.5)),4)]
O={"prev_exact":float((w0*y).sum()/w0.sum())}
# A: within no-htn-dx group, model vs SBP ranking
nh=h==0; sbp=te.sbp.fillna(te.sbp.median()).values
O["nohtn_share_of_cases"]=round(float((w0*y*nh).sum()/(w0*y).sum()),4); O["nohtn_prev"]=round(float((w0*y*nh).sum()/(w0*nh).sum()),4)
A={}
for v in [.10,.15,.20,.30]:
    f=lambda w:(cap(w,y,topk(p,w,v,nh),nh),cap(w,y,topk(sbp,w,v,nh),nh))
    pt=f(w0); bs=np.array([f(w) for w in W[:1000]])
    A[str(v)]=dict(model=round(pt[0],4),sbp=round(pt[1],4),diff=round(pt[0]-pt[1],4),diff_ci=ci(bs[:,0]-bs[:,1]))
O["A_nohtn_model_vs_sbp"]=A
# overlap within no-htn at 20%
s1=topk(p,w0,.2,nh); s2=topk(sbp,w0,.2,nh)
O["A_overlap_20"]=dict(both=round(float((w0*s1*s2*nh).sum()/(w0*nh).sum()),4),model_only=round(float((w0*s1*(1-s2)*nh).sum()/(w0*nh).sum()),4),sbp_only=round(float((w0*(1-s1)*s2*nh).sum()/(w0*nh).sum()),4))
# C: masked uric, lipids, ALT
m=fit_gb(NORACE); Xm=te[NORACE].copy()
for c_ in ["uric","hdl","chol","trig","alt"]: Xm[c_]=np.nan
pm=m.predict_proba(Xm)[:,1]; pct=(w0*h).sum()/w0.sum()
def wauc(yy,pp,ww):
    o=np.argsort(-pp,kind="stable"); yy,ww=yy[o],ww[o]; tp=np.cumsum(ww*yy); fp=np.cumsum(ww*(1-yy)); tp/=tp[-1]; fp/=fp[-1]; return np.trapezoid(np.r_[0,tp],np.r_[0,fp])
f=lambda w:(cap(w,y,topk(pm,w,(w*h).sum()/w.sum())),cap(w,y,h))
pt=f(w0); bs=np.array([f(w) for w in W[:1000]])
O["C_masked"]=dict(auc=round(wauc(y,pm,w0),4),found=round(pt[0],4),gain=round(pt[0]-pt[1],4),gain_ci=ci(bs[:,0]-bs[:,1]))
# calibration intercept/slope
def calib(w,mask):
    lp=np.log(p[mask]/(1-p[mask])); lr=LogisticRegression(C=1e6).fit(lp.reshape(-1,1),y[mask],sample_weight=w[mask])
    slope=lr.coef_[0][0]
    lr2=LogisticRegression(C=1e6,fit_intercept=True).fit(np.zeros((mask.sum(),1)),y[mask],sample_weight=w[mask])
    # calibration-in-the-large: intercept with offset lp -> approximate via log-odds difference
    obs=(w[mask]*y[mask]).sum()/w[mask].sum(); pr=(w[mask]*p[mask]).sum()/w[mask].sum()
    citl=np.log(obs/(1-obs))-np.log(pr/(1-pr))
    return citl,slope
for nm,mask in [("overall",np.ones(len(y),bool)),("women",te.female.values==1),("men",te.female.values==0)]:
    pt=calib(w0,mask); bs=np.array([calib(w,mask) for w in W[:500]])
    O["calib_"+nm]=dict(citl=round(pt[0],3),citl_ci=ci(bs[:,0]),slope=round(pt[1],3),slope_ci=ci(bs[:,1]))
# equity paired CIs
race={1:"Hispanic",2:"Hispanic",3:"White",4:"Black"}
E={}
for nm,mask in [("women",te.female.values==1),("men",te.female.values==0)]+[(r,te.race3.map(race).values==r) for r in ["White","Black","Hispanic"]]:
    f=lambda w:cap(w,y,topk(p,w,(w*h).sum()/w.sum()),mask)-cap(w,y,h,mask)
    E[nm]=[round(f(w0),4)]+ci([f(w) for w in W[:1000]])
O["equity_diff_ci"]=E
# NB CIs
def nb(w,s,t): return 1000*((w*s*y).sum()-(w*s*(1-y)).sum()*t/(1-t))/w.sum()
N={}
for t in [.05,.10]:
    f=lambda w:(nb(w,(p>=t)*1,t),nb(w,h,t))
    pt=f(w0); bs=np.array([f(w) for w in W[:1000]])
    N[str(t)]=dict(model=round(pt[0],2),model_ci=ci(bs[:,0]),rule=round(pt[1],2),rule_ci=ci(bs[:,1]),diff_ci=ci(bs[:,0]-bs[:,1]))
O["nb_ci"]=N
# t-multiplier check for headline diff
f=lambda w:cap(w,y,topk(p,w,(w*h).sum()/w.sum()))-cap(w,y,h)
v=np.array([f(w) for w in W]); pt=f(w0); se=v.std(ddof=1); tq=tdist.ppf(.975,15)
O["headline_t15"]=dict(point=round(pt,4),se=round(se,4),ci=[round(pt-tq*se,4),round(pt+tq*se,4)],pct_ci=ci(v))
# weighted overlap overall (pilot sizing)
sm=topk(p,w0,pct)
O["overlap_selection_w"]=dict(both=round(float((w0*sm*h).sum()/w0.sum()),4),model_only=round(float((w0*sm*(1-h)).sum()/w0.sum()),4),rule_only=round(float((w0*(1-sm)*h).sum()/w0.sum()),4))
# pregnancy-free refit
mp=fit_gb(NORACE,tr[tr.preg==0] if "preg" in tr else tr) if False else None
tra=pd.read_pickle("train_aug.pkl"); trn=tr.copy(); trn["preg"]=tra.preg.values
pp=fit_gb(NORACE,trn[trn.preg==0]).predict_proba(te[NORACE])[:,1]
O["nopreg_train"]=dict(auc=round(wauc(y,pp,w0),4),found=round(cap(w0,y,topk(pp,w0,pct)),4))
# time-check share found: rule vs model in 2017 cycle
a_,b_=tr[tr.cycle<2017],tr[tr.cycle==2017]; pb=fit_gb(NORACE,a_).predict_proba(b_[NORACE])[:,1]; wb=b_.w.values; yb=b_.y.values; hb=b_.htn_dx.values
O["timecheck"]=dict(model=round(cap(wb,yb,topk(pb,wb,(wb*hb).sum()/wb.sum())),4),rule=round(cap(wb,yb,hb),4),pct=round(float((wb*hb).sum()/wb.sum()),4))
# D: urine albumin/creatinine medians by sex and cycle
raw=pd.read_pickle("train_raw.pkl"); te_raw=c
D={}
def wmed(x,w):
    o=np.argsort(x); x,w=x[o],w[o]; return float(x[np.searchsorted(np.cumsum(w)/w.sum(),.5)])
for cyc,g in list(raw.groupby("cycle"))+[("2021",te_raw.assign(w=te_raw.WTMEC2YR))]:
    for sx in (1,2):
        q=g[g.RIAGENDR==sx].dropna(subset=["URXUMA","URXUCR"]); D[f"{cyc}_{'men' if sx==1 else 'women'}"]=[round(wmed(q.URXUMA.values,q.w.values),2),round(wmed(q.URXUCR.values,q.w.values),1)]
O["D_median_alb_cr"]=D
json.dump(O,open("res_round4.json","w"),indent=1); print(json.dumps(O,indent=0))
