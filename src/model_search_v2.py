"""Overnight search: can routine-data models beat the current final model WITHOUT touching the 2021-23 test set?
Selection metrics (training data only):
  (1) temporal validation: train on 2011-2016 cycles, validate on 2017-Mar 2020 (mimics drift to newer data)
  (2) repeated 5-fold CV on all 2011-2020
Primary metric: weighted % of cases caught at the high-BP-rule testing volume (temporal), then AUC."""
import pandas as pd, numpy as np, json, time, itertools, warnings, sys
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier

D = pd.read_csv(sys.argv[1]); OUT = sys.argv[2]
y = D.y.values; w = D.w.values; cyc = D.cycle.astype(str).values
base_cols = [c for c in D.columns if c not in ("y","w","cycle","sdmvpsu","sdmvstra","mexam","othhisp","black","asian","other")]

def engineer(X):
    X = X.copy()
    X["pulse_pressure"] = X.sbp - X.dbp
    X["map"] = X.dbp + (X.sbp - X.dbp) / 3
    X["trig_hdl"] = X.trig / X.hdl
    X["sbp_x_age"] = X.sbp * X.age / 100
    X["undiag_high_bp"] = ((X.sbp >= 130) & (X.htn_dx == 0)).astype(int)
    return X

def caught(yv, p, wv, ht):
    pct = (wv * ht).sum() / wv.sum()
    o = np.argsort(-p, kind="stable"); cw = np.cumsum(wv[o]) / wv.sum()
    s = np.zeros(len(p)); s[o[cw <= pct]] = 1
    return float((wv * s * yv).sum() / (wv * yv).sum())

tr_t = np.isin(cyc, ["2011", "2013", "2015"]); va_t = cyc == "2017"

def evaluate(make, X, name, log):
    t0 = time.time()
    # temporal
    m = make(); m.fit(X[tr_t], y[tr_t], sample_weight=w[tr_t]); p = m.predict_proba(X[va_t])[:, 1]
    t_auc = roc_auc_score(y[va_t], p, sample_weight=w[va_t]); t_c = caught(y[va_t], p, w[va_t], X.htn_dx.values[va_t])
    # repeated 5-fold
    aucs = []
    for seed in (1, 2, 3):
        oof = np.zeros(len(y))
        for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
            mm = make(); mm.fit(X.iloc[a], y[a], sample_weight=w[a]); oof[b] = mm.predict_proba(X.iloc[b])[:, 1]
        aucs.append(roc_auc_score(y, oof, sample_weight=w))
    r = dict(name=name, temporal_auc=round(t_auc, 4), temporal_caught=round(t_c, 4), cv_auc=round(float(np.mean(aucs)), 4), cv_auc_sd=round(float(np.std(aucs)), 4), secs=round(time.time() - t0, 1))
    log.append(r); json.dump(log, open(OUT, "w"), indent=1); print(r, flush=True)
    return r

class LR:
    def __init__(self, C=1.0): self.C = C
    def fit(self, X, y, sample_weight=None):
        self.med = X.median(); Z = X.fillna(self.med); self.mu, self.sd = Z.mean(), Z.std().replace(0, 1)
        self.m = LogisticRegression(max_iter=4000, C=self.C).fit((Z - self.mu) / self.sd, y, sample_weight=sample_weight); return self
    def predict_proba(self, X): return self.m.predict_proba((X.fillna(self.med) - self.mu) / self.sd)

X0 = D[base_cols]; X1 = engineer(X0)
FINAL = dict(n_estimators=800, learning_rate=0.01, num_leaves=7, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1, verbose=-1)
log = []
evaluate(lambda: lgb.LGBMClassifier(**FINAL), X0, "CURRENT FINAL (lgbm, base features)", log)
evaluate(lambda: lgb.LGBMClassifier(**FINAL), X1, "lgbm + engineered features", log)
evaluate(lambda: LR(1.0), X1, "logistic + engineered", log)
evaluate(lambda: xgb.XGBClassifier(n_estimators=600, learning_rate=0.02, max_depth=3, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, verbosity=0), X1, "xgboost depth3", log)
evaluate(lambda: CatBoostClassifier(iterations=800, learning_rate=0.03, depth=4, verbose=0), X1, "catboost depth4", log)
# monotone: risk should not fall as SBP, age, uric acid, HbA1c rise
mono = [1 if c in ("sbp", "age", "uric", "hba1c", "map", "pulse_pressure") else 0 for c in X1.columns]
evaluate(lambda: lgb.LGBMClassifier(**FINAL, monotone_constraints=mono), X1, "lgbm + engineered + monotone", log)
# random search on lgbm
rng = np.random.default_rng(42)
for i in range(40):
    p = dict(n_estimators=int(rng.choice([400, 800, 1200, 1600])), learning_rate=float(rng.choice([0.005, 0.01, 0.02])), num_leaves=int(rng.choice([4, 7, 11, 15])),
             min_child_samples=int(rng.choice([30, 50, 100, 200])), subsample=float(rng.choice([0.6, 0.8, 1.0])), subsample_freq=1,
             colsample_bytree=float(rng.choice([0.5, 0.7, 0.9])), reg_lambda=float(rng.choice([0, 1, 5, 20])), verbose=-1)
    evaluate(lambda p=p: lgb.LGBMClassifier(**p), X1, "lgbm search " + json.dumps({k: v for k, v in p.items() if k != "verbose"}), log)
print("DONE", flush=True)
