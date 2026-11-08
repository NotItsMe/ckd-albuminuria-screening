"""Add Muntner 2011 (REGARDS) albuminuria score to the 2021-2023 comparison."""
import pandas as pd,numpy as np,json,os
D=os.path.expanduser("~/mnt/Downloads")
c=pd.read_csv("cohort_2021_2023.csv"); h=pd.read_sas(f"{D}/HUQ_L.xpt",format="xport")[["SEQN","HUQ010"]]
c=c.merge(h,on="SEQN",how="left")
hs=c.HUQ010
X=(-5.1062+0.0318*c.RIDAGEYR+0.000994*(c.RIDAGEYR-65)**2+0.3476*(c.RIDRETH3==4)+0.2515*(c.RIAGENDR==1)
   +0.4383*c.SMQ040.isin([1,2])+0.2615*(hs==3)+0.5043*(hs==4)+0.4545*(hs==5)+0.4755*(c.BPQ020==1)+0.3536*(c.MCQ160F==1))
c["p_muntner"]=1/(1+np.exp(-X))
t=pd.read_csv("test_scores_2021_2023.csv").drop(columns=["p_muntner"],errors="ignore").merge(c[["SEQN","p_muntner"]],on="SEQN")
t.to_csv("test_scores_2021_2023.csv",index=False)
print("missing self-rated health:",int(hs.isna().sum()))
