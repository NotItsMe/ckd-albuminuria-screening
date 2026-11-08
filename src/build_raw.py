import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
DN="/mnt/user-data/uploads/Downloads/nhanes"; DL="/mnt/user-data/uploads/Downloads"
rd=lambda p: pd.read_sas(p,format="xport")
def build(files,wcol,wfactor,cycle,egfr_min=60):
    d=rd(files["DEMO"])
    for k,f in files.items():
        if k!="DEMO": d=d.merge(rd(f),on="SEQN",how="left",suffixes=("","_"+k))
    a=d[(d.RIDAGEYR>=18)&(d.RIDSTATR==2)].dropna(subset=["URXUMA","URXUCR","LBXSCR","LBXGH"]).copy()
    fem=a.RIAGENDR==2; k=np.where(fem,.7,.9); al=np.where(fem,-.241,-.302); s=a.LBXSCR/k
    a["eGFR"]=142*np.minimum(s,1)**al*np.maximum(s,1)**-1.2*.9938**a.RIDAGEYR*np.where(fem,1.012,1)
    a=a[~((a.DIQ010==1)|(a.LBXGH>=6.5))&(a.eGFR>=egfr_min)].copy()
    a["UACR"]=100*a.URXUMA/a.URXUCR; a["y"]=(a.UACR>=30)*1
    a["w"]=a[wcol]*wfactor; a["cycle"]=cycle
    sy=[c for c in a.columns if c.startswith("BPXSY") or c.startswith("BPXOSY")]
    dy=[c for c in a.columns if (c.startswith("BPXDI") or c.startswith("BPXODI"))]
    a["sbp"]=a[sy].mean(axis=1); a["dbp"]=a[dy].replace(0,np.nan).mean(axis=1)
    return a
names=["DEMO","ALB_CR","BIOPRO","GHB","DIQ","BPQ","MCQ","SMQ","HDL","BMX"]
def train_raw():
    parts=[]
    for y,sfx in {"2011":"_G","2013":"_H","2015":"_I"}.items():
        f={n:f"{DN}/{n}{sfx}.xpt" for n in names}; f["BPX"]=f"{DN}/BPX{sfx}.xpt"; parts.append(build(f,"WTMEC2YR",2/9.2,y))
    f={n:f"{DN}/P_{n}.xpt" for n in names}; f["BPXO"]=f"{DN}/P_BPXO.xpt"; parts.append(build(f,"WTMECPRP",3.2/9.2,"2017"))
    return pd.concat(parts,ignore_index=True)
def test_raw(egfr_min=60):
    f={n:f"{DL}/{n}_L.xpt" for n in names}; f["BPXO"]=f"{DL}/BPXO_L.xpt"; return build(f,"WTMEC2YR",1,"2021",egfr_min)
if __name__=="__main__":
    tr=train_raw(); tr.to_pickle("train_raw.pkl")
    te0=test_raw(0); te0.to_pickle("test_all_egfr.pkl")
    m=pd.read_csv("train_matrix.csv")
    print(len(tr),len(m), np.allclose(tr.w.values,m.w.values), (tr.y.values==m.y.values).all(), len(te0), (te0.eGFR<60).sum())
