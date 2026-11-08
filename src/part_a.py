"""Part A: test existing albuminuria scores vs. testing rules on NHANES Aug 2021-Aug 2023.
Cohort: adults 18+, examined, no diabetes (self-report or HbA1c>=6.5), eGFR>=60 (CKD-EPI 2021).
Outcome: UACR >= 30 mg/g. Scores: Bragg-Gresham 2024 Models 1 & 2 (coefficients from their Appendix 1 calculator)."""
import pandas as pd, numpy as np, sys, json
D = sys.argv[1]
r = lambda f: pd.read_sas(f"{D}/{f}.xpt", format="xport")
d = r("DEMO_L")
for f in ["ALB_CR_L","BIOPRO_L","GHB_L","DIQ_L","BPXO_L","BPQ_L","HDL_L","BMX_L","MCQ_L","SMQ_L"]:
    d = d.merge(r(f), on="SEQN", how="left", suffixes=("", "_"+f))
rx = r("RXQ_RX_L")
a = d[(d.RIDAGEYR>=18)&(d.RIDSTATR==2)].dropna(subset=["URXUMA","URXUCR","LBXSCR","LBXGH"]).copy()
a["UACR"] = 100*a.URXUMA/a.URXUCR
fem = a.RIAGENDR==2; k=np.where(fem,.7,.9); al=np.where(fem,-.241,-.302); s=a.LBXSCR/k
a["eGFR"] = 142*np.minimum(s,1)**al*np.maximum(s,1)**-1.2*.9938**a.RIDAGEYR*np.where(fem,1.012,1)
a = a[~((a.DIQ010==1)|(a.LBXGH>=6.5)) & (a.eGFR>=60)].copy()
a["y"] = (a.UACR>=30).astype(int); a["y_adj"] = (a.UACR>=30*1.062).astype(int)
a["w"] = a.WTMEC2YR
fem = a.RIAGENDR==2
# predictors
age=a.RIDAGEYR
a["age_lt25"]=(age<25)*1; a["age_65_74"]=((age>=65)&(age<75))*1; a["age_75"]=(age>=75)*1
a["male"]=(a.RIAGENDR==1)*1
re=a.RIDRETH3  # 1 MexAm 2 OthHisp 3 White 4 Black 6 Asian 7 Other
a["black"]=(re==4)*1; a["mex"]=(re==1)*1; a["othhisp"]=(re==2)*1; a["othnh"]=re.isin([6,7])*1
a["smoker"]=a.SMQ040.isin([1,2])*1
a["prediab"]=((a.LBXGH>=5.7)&(a.LBXGH<6.5))*1
a["htn"]=((a.BPQ020==1)|(a.BPQ150==1))*1   # told high BP or taking BP meds
cvd_cols=[c for c in ["MCQ160B","MCQ160C","MCQ160D","MCQ160E","MCQ160F"] if c in a]
a["cvd"]=(a[cvd_cols]==1).any(axis=1)*1
ua=a.LBXSUA; a["hi_ua"]=np.where(fem, ua>6.1, ua>7.2)*1
hdl=a.LBDHDD; a["lo_hdl"]=np.where(fem, hdl<40, hdl<30)*1
a["obese"]=(a.BMXBMI>=30)*1
# NOTE: 2021-2023 public RXQ_RX_L has no drug names (only SEQN, RXQ033, RXQ050), so ACEi/ARB/statin are unknown -> set to 0 for Model 2 (sensitivity only)
for m in ["acei","arb","statin"]: a[m]=0
M1=dict(i=-3.3723,age_lt25=.522,age_65_74=.2415,age_75=.9294,male=-.3873,black=.2489,mex=.5301,othhisp=-.1342,othnh=.4185,smoker=.2785,prediab=.1561,htn=1.4134,cvd=.2172)
M2=dict(i=-3.4053,age_lt25=.4956,age_65_74=.2585,age_75=.7934,male=-.3716,black=.2299,mex=.5195,othhisp=-.1417,othnh=.4161,smoker=.254,prediab=.1582,htn=1.4748,cvd=.2751,hi_ua=.3607,lo_hdl=.2314,obese=.000867,acei=-.3529,arb=-.4389,statin=-.313)
def score(m): return 1/(1+np.exp(-(m["i"]+sum(a[k]*v for k,v in m.items() if k!="i"))))
a["p_m1"]=score(M1); a["p_m2"]=score(M2)
a.to_csv(f"{D}/../BigBrain/Projects/CKD Early Detection/analysis/cohort_2021_2023.csv", index=False)
print("n",len(a),"cases",a.y.sum(),"| missing hi_ua",a.LBXSUA.isna().sum(),"lo_hdl",a.LBDHDD.isna().sum(),"smoke",a.SMQ040.isna().sum())
print("htn", a.htn.sum(), "cvd", a.cvd.sum())
