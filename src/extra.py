import pandas as pd, numpy as np, json, warnings; warnings.filterwarnings("ignore")
exec(open("metrics.py").read().split("cols={m")[0])
te=pd.read_pickle("test_scored.pkl"); y=te.y.values; w=te.w.values; M="p_gb_norace"; p=te[M].values; h=te.htn.values
pct=(w*h).sum()/w.sum(); X={}
def stats(sel,yy=y,ww=w):
    tp=(ww*sel*yy).sum(); fp=(ww*sel*(1-yy)).sum(); fn=(ww*(1-sel)*yy).sum(); tn=(ww*(1-sel)*(1-yy)).sum()
    return dict(sens=tp/(tp+fn),spec=tn/(tn+fp),ppv=tp/(tp+fp),nns=(tp+fp)/tp,sel=(sel*ww).sum()/ww.sum())
X["diag"]={f"{q:.3f}":{k:round(float(v),4) for k,v in stats(topk(p,w,q)).items()} for q in [.10,.20,pct,.40]}
X["diag_htn"]={k:round(float(v),4) for k,v in stats(h).items()}
# equity by sex and race (capture + selection), model at matched volume vs htn
sel=topk(p,w,pct); race={1:"Hispanic",2:"Hispanic",3:"White",4:"Black",6:"Asian",7:"Other"}
grp=[("all",np.ones(len(te),bool)),("women",te.female.values==1),("men",te.female.values==0)]+[(n,te.race3.map(race).values==n) for n in ["White","Black","Hispanic","Asian","Other"]]
X["equity"]={}
for n,g in grp:
    yy,ww=y[g],w[g]
    X["equity"][n]=dict(cases=int(yy.sum()),n=int(g.sum()),prev=round(float((ww*yy).sum()/ww.sum()),4),
      model_capture=round(float((ww*sel[g]*yy).sum()/(ww*yy).sum()),4),htn_capture=round(float((ww*h[g]*yy).sum()/(ww*yy).sum()),4),
      model_selected=round(float((ww*sel[g]).sum()/ww.sum()),4),htn_selected=round(float((ww*h[g]).sum()/ww.sum()),4),
      pred=round(float((ww*p[g]).sum()/ww.sum()),4),obs=round(float((ww*yy).sum()/ww.sum()),4),
      full_pred=round(float((ww*te.p_gb_full.values[g]).sum()/ww.sum()),4),full_capture=round(float((ww*topk(te.p_gb_full.values,w,pct)[g]*yy).sum()/(ww*yy).sum()),4))
# profile of caught cases
cases=y==1; u=te.uacr.values; bp=((te.sbp>=130)|(te.dbp>=80)).values; a50=(te.age>=50).values
def prof(s):
    m=cases&(s==1); W=w[m]; return dict(n_unw=int(m.sum()),uacr200=round(float((W*(u[m]>=200)).sum()/W.sum()),4),uacr300=round(float((W*(u[m]>=300)).sum()/W.sum()),4),bp130=round(float((W*bp[m]).sum()/W.sum()),4),age50=round(float((W*a50[m]).sum()/W.sum()),4),htn_dx=round(float((W*h[m]).sum()/W.sum()),4))
X["caught_profile"]={"model":prof(sel),"htn":prof(h),"all_cases":prof(np.ones(len(y))),"model_not_htn":prof(sel*(1-h))}
X["cases_all_uacr200_share"]=X["caught_profile"]["all_cases"]["uacr200"]
# weighted population numbers
X["weighted_pop_millions"]=round(w.sum()/1e6,2); X["weighted_cases_millions"]=round((w*y).sum()/1e6,2)
# calibration
X["calib_overall"]=[round(float((w*p).sum()/w.sum()),4),round(float((w*y).sum()/w.sum()),4)]
dec=pd.qcut(p,10,labels=False); X["calib_deciles"]=[[round(float((w[dec==d]*p[dec==d]).sum()/w[dec==d].sum()),4),round(float((w[dec==d]*y[dec==d]).sum()/w[dec==d].sum()),4)] for d in range(10)]
# decision curve
def nb(sel,t): n=w.sum(); return ((w*sel*y).sum()-(w*sel*(1-y)).sum()*t/(1-t))/n
X["net_benefit_per1000"]={str(t):dict(model=round(1000*nb((p>=t)*1,t),2),htn=round(1000*nb(h,t),2),all=round(1000*nb(np.ones(len(y)),t),2),m1=round(1000*nb((te.p_m1.values>=t)*1,t),2)) for t in [.02,.03,.05,.08,.10,.15,.20]}
# sensitivity: exclude pregnant; exclude non-fasting glucose>=200 (none) / >=126
for nm,keep in [("no_preg",te.preg.values==0),("glu_lt126",~(te.glucose.values>=126))]:
    yy,ww,pp,hh=y[keep],w[keep],p[keep],h[keep]; q=(ww*hh).sum()/ww.sum(); s=topk(pp,ww,q)
    X["sens_"+nm]=dict(n=int(keep.sum()),auc=round(float(wauc(yy,pp,ww)),4),model=round(float((ww*s*yy).sum()/(ww*yy).sum()),4),htn=round(float((ww*hh*yy).sum()/(ww*yy).sum()),4))
X["n_preg"]=int(te.preg.sum()); X["n_glu126"]=int((te.glucose>=126).sum())
# SHAP
exec(open("core.py").read()); NORACE=[f for f in FULL if f not in RACE]
m=fit_gb(NORACE); sh=m.predict(te[NORACE],pred_contrib=True)[:,:-1]; ms=np.abs(sh).mean(0); X["shap_share"]={c:round(float(v/ms.sum()),4) for c,v in sorted(zip(NORACE,ms),key=lambda t:-t[1])}
m.booster_.save_model("model_main_racefree.txt")
# risk-factor rule: diagnosed/treated HBP, CVD, BMI>=30, age>=60
rf=((te.htn_dx==1)|(te.cvd==1)|(te.bmi>=30)|(te.age>=60)).values*1
X["riskfactor_rule"]=dict(selected=round(float((w*rf).sum()/w.sum()),6),caught=round(float((w*rf*y).sum()/(w*y).sum()),6),per1000=round(float(1000*(w*rf*y).sum()/(w*rf).sum()),2))
X["all_per1000"]=round(float(1000*(w*y).sum()/w.sum()),2)
o=np.argsort(te.age.values); X["median_age"]=float(te.age.values[o][np.searchsorted(np.cumsum(w[o])/w.sum(),.5)]); X["women_share"]=round(float((w*te.female).sum()/w.sum()),6)
# calibration by fifths within sex
for sx,nm in [(1,"women"),(0,"men")]:
    mm=te.female.values==sx; pp,yy,ww=p[mm],y[mm],w[mm]; d=pd.qcut(pp,5,labels=False)
    X["calib_fifths_"+nm]=[[round(float((ww[d==k]*pp[d==k]).sum()/ww[d==k].sum()),4),round(float((ww[d==k]*yy[d==k]).sum()/ww[d==k].sum()),4),int(yy[d==k].sum())] for k in range(5)]
X["calib_deciles_n_cases"]=[int(y[dec==d].sum()) for d in range(10)]
# overlap of caught cases (unweighted counts and weighted shares)
both=(sel==1)&(h==1)&(y==1); mo=(sel==1)&(h==0)&(y==1); ho=(sel==0)&(h==1)&(y==1)
X["overlap"]=dict(both_unw=int(both.sum()),model_only_unw=int(mo.sum()),rule_only_unw=int(ho.sum()),model_unw=int(((sel==1)&(y==1)).sum()),rule_unw=int(((h==1)&(y==1)).sum()),
  both_w=round(float((w*both).sum()/(w*y).sum()),4),model_only_w=round(float((w*mo).sum()/(w*y).sum()),4),rule_only_w=round(float((w*ho).sum()/(w*y).sum()),4))
X["selected_share_model"]=round(float((w*sel).sum()/w.sum()),6)
json.dump(X,open("res_extra.json","w"),indent=1); print(json.dumps(X,indent=0))
