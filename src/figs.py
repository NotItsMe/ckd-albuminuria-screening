import pandas as pd, numpy as np, json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.spines.top":False,"axes.spines.right":False,
 "axes.edgecolor":"#9a9990","axes.labelcolor":"#3b3a36","xtick.color":"#6b6a63","ytick.color":"#6b6a63",
 "axes.grid":True,"grid.color":"#e7e6e1","grid.linewidth":.8,"figure.facecolor":"#ffffff","axes.facecolor":"#ffffff"})
BLUE,ORANGE,AQUA,YELLOW,GRAY,PURP="#2a78d6","#eb6834","#1baf7a","#eda100","#6b6a63","#8a5cc2"
t=pd.read_pickle("test_scored.pkl"); y,w=t.y.values,t.w.values
R=json.load(open("res_metrics.json")); X=json.load(open("res_extra.json"))
def curve(p):
    o=np.argsort(-p,kind="stable"); cw=np.cumsum(w[o])/w.sum(); cy=np.cumsum((w*y)[o])/(w*y).sum(); return np.r_[0,cw]*100,np.r_[0,cy]*100
def save(f,name): f.tight_layout(); f.savefig(f"figs2/{name}.png",dpi=200); plt.close(f)
pct=R["point"]["pct"]*100; htn=R["point"]["htn"]*100
# Fig 1 flow
f,ax=plt.subplots(figsize=(8,7.6)); ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,11)
def box(x,yb,wd,txt,fc="#eef4fc",ec=BLUE,fs=9.5):
    ax.add_patch(FancyBboxPatch((x-wd/2,yb-.5),wd,1.0,boxstyle="round,pad=0.04,rounding_size=0.12",fc=fc,ec=ec,lw=1.3)); ax.text(x,yb,txt,ha="center",va="center",fontsize=fs,color="#3b3a36")
for col,x,hdr,n in [(0,2.5,"Training: NHANES 2011–March 2020",[26239,24033,20140,19090,1524]),(1,7.5,"Testing: NHANES Aug 2021–Aug 2023",[6337,5551,4664,4377,302])]:
    ax.text(x,10.6,hdr,ha="center",fontsize=10.5,weight="bold",color="#3b3a36")
    box(x,9.6,4.4,f"Examined adults 18+\nn = {n[0]:,}")
    box(x,7.4,4.4,f"With urine albumin, urine creatinine,\nserum creatinine and HbA1c\nn = {n[1]:,}")
    box(x,5.2,4.4,f"No diabetes (self-report or HbA1c ≥6.5%)\nn = {n[2]:,}")
    box(x,3.0,4.4,f"eGFR ≥60 (CKD-EPI 2021)\nn = {n[3]:,}",fc="#dbe8fa")
    box(x,1.0,4.4,f"Albuminuria (UACR ≥30 mg/g)\non a single sample: {n[4]:,}",fc="#fdeee6",ec=ORANGE)
    for a,b,d in [(9.6,7.4,n[0]-n[1]),(7.4,5.2,n[1]-n[2]),(5.2,3.0,n[2]-n[3]),(3.0,1.0,None)]:
        ax.annotate("",xy=(x,b+.55),xytext=(x,a-.55),arrowprops=dict(arrowstyle="->",color=GRAY,lw=1.2))
        if d: ax.text(x+.12,(a+b)/2,f"excluded {d:,}",fontsize=8.5,color=GRAY,va="center")
save(f,"fig1_cohort")
# Fig 2
f,ax=plt.subplots(figsize=(7,5))
for m,c,lab,ls,mk in [("p_gb_norace",BLUE,"Final model (race-free gradient boosting)","-","o"),("p_lr_norace",AQUA,"Logistic regression, same inputs","--","s"),("p_m1",ORANGE,"Bragg-Gresham Model 1",":","^"),("p_muntner",PURP,"Muntner score","-.","v")]:
    x,yy=curve(t[m].values); ax.plot(x,yy,color=c,lw=2,ls=ls,label=lab); i=np.searchsorted(x,pct); ax.plot(x[i],yy[i],marker=mk,color=c,ms=7,mec="#fff",mew=1.5)
ax.plot([0,100],[0,100],color=GRAY,lw=1,ls=(0,(2,3)),label="No ranking")
ax.plot(pct,htn,marker="D",color=YELLOW,ms=9,mec="#3b3a36",mew=1,ls="none",label="Test diagnosed high blood pressure")
ax.axvline(pct,color="#c3c2b7",lw=1); ax.set_xlabel("People tested (%)"); ax.set_ylabel("Albuminuria cases caught (%)"); ax.set_xlim(0,100); ax.set_ylim(0,100)
ax.legend(frameon=False,fontsize=8.5,loc="lower right"); save(f,"fig2_cases_vs_tested")
# Fig 3 AUC forest
rows=[("Muntner score","p_muntner",PURP),("Bragg-Gresham Model 1","p_m1",ORANGE),("Bragg-Gresham predictors, refit","p_bg_refit",ORANGE),("Logistic regression, 5 inputs","p_lr_simple5",AQUA),("Logistic regression, same 18 inputs","p_lr_norace",AQUA),("EHR-only (no waist, no uric acid)","p_gb_ehr",BLUE),("No uric acid","p_gb_norace_nouric",BLUE),("Final model (race-free)","p_gb_norace",BLUE),("Race included (supplement)","p_gb_full",GRAY)]
f,ax=plt.subplots(figsize=(7.5,4.4))
for i,(lab,m,c) in enumerate(rows):
    lo,hi=R["auc_ci"][m]; pt=R["point"][m+"_auc"]; ax.plot([lo,hi],[i,i],color=c,lw=2); ax.plot(pt,i,"o",color=c,ms=8,mec="#fff",mew=1.5); ax.text(hi+.004,i,f"{pt:.3f}",va="center",fontsize=9,color="#3b3a36")
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows],fontsize=9); ax.set_xlabel("AUC, 2021–2023 test data (95% CI)"); ax.set_xlim(.55,.80); ax.grid(axis="y",visible=False); save(f,"fig3_auc")
# Fig 4 calibration overall + by sex
f,axs=plt.subplots(1,2,figsize=(9,4.4)); p=t.p_gb_norace.values
for ax,(nm,mask) in zip(axs,[("All adults",np.ones(len(t),bool)),("By sex",None)]):
    if mask is not None:
        d=pd.qcut(p,10,labels=False); g=[((w[d==k]*p[d==k]).sum()/w[d==k].sum()*100,(w[d==k]*y[d==k]).sum()/w[d==k].sum()*100) for k in range(10)]
        ax.plot(*zip(*g),"o-",color=BLUE,lw=2,ms=7,mec="#fff",mew=1.5,label="Deciles of predicted risk")
    else:
        for sx,c,lab in [(1,ORANGE,"Women"),(0,AQUA,"Men")]:
            mm=t.female.values==sx; pp,yy,ww=p[mm],y[mm],w[mm]; d=pd.qcut(pp,5,labels=False)
            g=[((ww[d==k]*pp[d==k]).sum()/ww[d==k].sum()*100,(ww[d==k]*yy[d==k]).sum()/ww[d==k].sum()*100) for k in range(5)]
            ax.plot(*zip(*g),"o-",color=c,lw=2,ms=7,mec="#fff",mew=1.5,label=lab+" (fifths)")
    ax.plot([0,25],[0,25],color=GRAY,lw=1,ls=(0,(2,3))); ax.set_xlim(0,25); ax.set_ylim(0,25); ax.set_title(nm,fontsize=10.5)
    ax.set_xlabel("Predicted risk (%)"); ax.set_ylabel("Observed albuminuria (%)"); ax.legend(frameon=False,fontsize=8.5,loc="upper left")
save(f,"fig4_calibration")
# Fig 5 DCA
th=np.linspace(.02,.20,37); W=w.sum(); nb=lambda s,tt:((w*s*y).sum()-(w*s*(1-y)).sum()*tt/(1-tt))/W
f,ax=plt.subplots(figsize=(7,5))
for lab,c,ls,fn in [("Final model",BLUE,"-",lambda tt:nb((p>=tt)*1,tt)),("Bragg-Gresham Model 1",ORANGE,":",lambda tt:nb((t.p_m1.values>=tt)*1,tt)),("Test diagnosed high blood pressure",YELLOW,"-.",lambda tt:nb(t.htn.values,tt)),("Test everyone",GRAY,"--",lambda tt:nb(np.ones(len(t)),tt))]:
    ax.plot(th*100,[fn(x)*1000 for x in th],color=c,lw=2,ls=ls,label=lab)
ax.axhline(0,color="#3b3a36",lw=1); ax.set_ylim(-20,45); ax.set_xlabel("Risk threshold for testing (%)"); ax.set_ylabel("Net benefit (true cases per 1,000 people)"); ax.legend(frameon=False,fontsize=9); save(f,"fig5_decision_curve")
# Fig 6 SHAP
nice={"age":"Age","female":"Sex","sbp":"Systolic BP","dbp":"Diastolic BP","htn_dx":"Diagnosed high BP","bmi":"BMI","hba1c":"HbA1c","hdl":"HDL cholesterol","uric":"Uric acid","smoker":"Current smoker","cvd":"Heart disease or stroke","waist":"Waist size","glucose":"Non-fasting glucose","trig":"Triglycerides","chol":"Total cholesterol","alt":"ALT","pulse":"Pulse","chol_dx":"Diagnosed high cholesterol"}
s=pd.Series(X["shap_share"]).sort_values()*100
f,ax=plt.subplots(figsize=(7,5.6)); ax.barh([nice[i] for i in s.index],s.values,color=BLUE,height=.6)
for i,v in enumerate(s.values): ax.text(v+.3,i,f"{v:.0f}%",va="center",fontsize=9,color="#3b3a36")
ax.set_xlabel("Share of mean absolute SHAP value (%)"); ax.grid(axis="y",visible=False); save(f,"fig6_shap")
print("ok")
