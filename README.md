# Choosing who gets a urine albumin test (NHANES 2011–2023)

**Research use only, not for clinical use.**

Code for the paper *The Urine Test Nobody Orders: Using common clinical measurements to decide who gets a urine albumin test* (Zaid Mohammed, November 2026).

The model ranks U.S. adults without diabetes and with eGFR ≥ 60 by their chance of a urine albumin-to-creatinine ratio (UACR) ≥ 30 mg/g on a single sample, using 18 routine inputs (no kidney-damage markers, no race). It is compared with testing everyone with diagnosed high blood pressure, Bragg-Gresham Model 1 and the Muntner score at the same testing volume.

**Main result (Aug 2021–Aug 2023 test data):** 52.9% of single-sample albuminuria found vs 40.3% for the high blood pressure rule when both test 23.1% of adults (gain 12.6 points, 95% CI 4.4–20.5); AUC 0.726 (0.692–0.760). The model has not been validated on clinical records.

## Later checks
- **Round 4 robustness checks** (`src/round4.py` → `results/res_round4.json`): model vs blood-pressure ranking among people without a hypertension diagnosis, a version with some labs masked, calibration by sex, equity confidence intervals and net benefit.
- **NHANES 2009–2010** (`src/score_2009.py` → `results/res_2009.json`; plan in `PLAN_2009.md`): a one-time, pre-specified check on a cycle never used in development. At the same testing volume, the model found 45.9% of single-sample albuminuria vs 36.1% for the high blood pressure rule (AUC 0.706). For albuminuria confirmed on a second sample, it did about the same as the rule.

## Data
All data are public, de-identified NHANES files from the CDC (not included here). Put these .xpt files in one folder (`data/` at the top of the repo, or any folder named by the `NHANES_DIR` environment variable), keeping the CDC file names:

| Cycle | Files |
| --- | --- |
| 2011–12 (G), 2013–14 (H), 2015–16 (I) | DEMO, ALB_CR, BIOPRO, GHB, DIQ, BPQ, MCQ, SMQ, HDL, BMX, BPX, RXQ_RX (e.g. `DEMO_G.xpt`) |
| 2017–Mar 2020 (P) | the same files with the `P_` prefix and BPXO instead of BPX (e.g. `P_DEMO.xpt`, `P_BPXO.xpt`) |
| Aug 2021–Aug 2023 (L) | DEMO, ALB_CR, BIOPRO, GHB, DIQ, BPQ, MCQ, SMQ, HDL, BMX, BPXO, HUQ (e.g. `DEMO_L.xpt`) |
| 2009–10 (F), used once for the final check | DEMO, ALB_CR, BIOPRO, GHB, DIQ, BPQ, MCQ, SMQ, HDL, BMX, BPX |

## How to run
```
pip install -r requirements.txt
NHANES_DIR=/path/to/xpt/files bash run_all.sh
```
`run_all.sh` runs every script in order from inside `src/` (about 15–20 minutes, most of it bootstrap). Each script writes its `res_*.json` next to it; these match the files in `results/` exactly. A clean run from the raw CDC files was checked against `results/` on 8 November 2026.

| Step | Script | What it does |
| --- | --- | --- |
| 1 | `part_a.py` | Builds the 2021–2023 test cohort and scores Bragg-Gresham Model 1 |
| 2 | `build_raw.py` | Builds the 2011–Mar 2020 training adults (18+, examined, no diabetes, CKD-EPI 2021 eGFR ≥ 60) |
| 3 | `build_features.py` | Makes the 18 inputs (blood pressure = mean of first three readings; pulse = first reading) |
| 4 | `part_d_muntner.py` | Adds the Muntner score |
| 5 | `analysis.py` | Fits the final model, comparators and sensitivity versions; cross-validation and time check |
| 6 | `metrics.py` | Main results with Rao–Wu bootstrap 95% CIs (2,000 replicates) |
| 7 | `extra.py`, `extra2.py` | Bedside table, equity, calibration, decision curves, SHAP, severe albuminuria |
| 8 | `round4.py` | Later robustness checks |
| 9 | `score_2009.py` | One-time check of the frozen model on 2009–2010 |
| 10 | `figs.py` | Figures |

`model_search_v2.py` (run separately: `python model_search_v2.py train_matrix_v2.csv out.json`, several hours) is the training-only model search. `results/earlier_candidates_test_scores.csv` holds test-set scores from earlier development versions, kept because the paper reports them.

`results/` holds the outputs used in the paper. `model/model_main_racefree.txt` is the fitted LightGBM model.

## Not for clinical use
This is a research model built on survey data. It does not diagnose anyone and has not been tested on hospital records.