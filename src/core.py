import pandas as pd, numpy as np, json, warnings; warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
P=json.load(open("part_c_cv.json"))["best_params"]; P={**P,"verbose":-1}
tr=pd.read_csv("train_matrix_v2.csv")
c=pd.read_csv("cohort_2021_2023.csv")
def test_X(a):
    X=pd.DataFrame(index=a.index); fem=a.RIAGENDR==2
    X["age"]=a.RIDAGEYR; X["female"]=fem*1
    for nm,k in [("mexam",1),("othhisp",2),("black",4),("asian",6),("other",7)]: X[nm]=(a.RIDRETH3==k)*1
    X["sbp"]=a[["BPXOSY1","BPXOSY2","BPXOSY3"]].mean(axis=1); X["dbp"]=a[["BPXODI1","BPXODI2","BPXODI3"]].replace(0,np.nan).mean(axis=1)
    X["htn_dx"]=((a.BPQ020==1)|(a.BPQ150==1))*1
    X["bmi"]=a.BMXBMI; X["hba1c"]=a.LBXGH; X["hdl"]=a.LBDHDD; X["uric"]=a.LBXSUA
    X["smoker"]=a.SMQ040.isin([1,2])*1
    X["cvd"]=(a[["MCQ160B","MCQ160C","MCQ160D","MCQ160E","MCQ160F"]]==1).any(axis=1)*1
    X["waist"]=a.BMXWAIST; X["glucose"]=a.LBXSGL; X["trig"]=a.LBXSTR; X["chol"]=a.LBXSCH; X["alt"]=a.LBXSATSI
    X["pulse"]=a[["BPXOPLS1","BPXOPLS2","BPXOPLS3"]].bfill(axis=1).iloc[:,0]; X["chol_dx"]=(a.BPQ080==1)*1
    return X
FULL=["age","female","mexam","othhisp","black","asian","other","sbp","dbp","htn_dx","bmi","hba1c","hdl","uric","smoker","cvd","waist","glucose","trig","chol","alt","pulse","chol_dx"]
RACE=["mexam","othhisp","black","asian","other"]
te=test_X(c); te["y"]=(100*c.URXUMA/c.URXUCR>=30)*1; te["w"]=c.WTMEC2YR; te["psu"]=c.SDMVPSU; te["str"]=c.SDMVSTRA
te["uacr"]=100*c.URXUMA/c.URXUCR; te["SEQN"]=c.SEQN; te["race3"]=c.RIDRETH3; te["htn"]=te.htn_dx; te["p_m1"]=c.p_m1
te["preg"]=(c.RIDEXPRG==1)*1
def fit_gb(cols,data=tr): return lgb.LGBMClassifier(**P).fit(data[cols],data.y,sample_weight=data.w)
def fit_lr(cols,data=tr):
    X=data[cols]; med=X.median(); Z=X.fillna(med); mu,sd=Z.mean(),Z.std().replace(0,1)
    m=LogisticRegression(max_iter=5000).fit((Z-mu)/sd,data.y,sample_weight=data.w)
    return lambda D:m.predict_proba((D[cols].fillna(med)-mu)/sd)[:,1]
