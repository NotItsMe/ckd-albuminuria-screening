import pandas as pd,numpy as np
def train_X(a, n_readings=3):
    X=pd.DataFrame(index=a.index); fem=a.RIAGENDR==2
    X["age"]=a.RIDAGEYR; X["female"]=fem*1
    for nm,k in [("mexam",1),("othhisp",2),("black",4),("asian",6),("other",7)]: X[nm]=(a.RIDRETH3==k)*1
    r=range(1,n_readings+1) if n_readings else range(1,5)
    sy=[c for i in r for c in (f"BPXSY{i}",f"BPXOSY{i}") if c in a]; dy=[c for i in r for c in (f"BPXDI{i}",f"BPXODI{i}") if c in a]
    X["sbp"]=a[sy].mean(axis=1); X["dbp"]=a[dy].replace(0,np.nan).mean(axis=1)
    med=a.BPQ150 if "BPQ150" in a else a.BPQ050A
    X["htn_dx"]=((a.BPQ020==1)|(med==1))*1
    X["bmi"]=a.BMXBMI; X["hba1c"]=a.LBXGH; X["hdl"]=a.LBDHDD; X["uric"]=a.LBXSUA
    X["smoker"]=a.SMQ040.isin([1,2])*1
    X["cvd"]=(a[["MCQ160B","MCQ160C","MCQ160D","MCQ160E","MCQ160F"]]==1).any(axis=1)*1
    X["waist"]=a.BMXWAIST; X["glucose"]=a.LBXSGL; X["trig"]=a.LBXSTR; X["chol"]=a.LBXSCH; X["alt"]=a.LBXSATSI
    pc=[c for c in ["BPXPLS","BPXOPLS1","BPXOPLS2","BPXOPLS3"] if c in a]; X["pulse"]=a[pc].bfill(axis=1).iloc[:,0]; X["chol_dx"]=(a.BPQ080==1)*1
    X["y"]=a.y.values; X["w"]=a.w.values; X["cycle"]=a.cycle.astype(int).values; X["sdmvpsu"]=a.SDMVPSU.values; X["sdmvstra"]=a.SDMVSTRA.values
    return X
if __name__=="__main__":
    raw=pd.read_pickle("train_raw.pkl")
    new=train_X(raw,3); new.to_csv("train_matrix_v2.csv",index=False)
    print("training matrix:",new.shape)
