# Choosing who gets a urine albumin test (NHANES 2011–2023)

**This is a research project and not intended for a diagnosis or for use in a clinic.**

Code for the paper *The Urine Test Nobody Orders: Using common clinical measurements to decide who gets a urine albumin test* (Zaid Mohammed, November 2026).

Most chronic kidney disease is picked up at stage 3. By then, a significant amount of kidney function has already been lost. Around 9 of 10 people living in the US with kidney disease don't realize they have it. Kidney disease in the early stages (1 and 2) can be caught with a cheap urine test called UACR, which looks for protein leaking into the urine. The issue is that this test is not used often unless a patient has diabetes or high blood pressure. So I wanted to answer a simple question: for a clinic that has limited capacity to perform the UACR test, which patients should get tested? 

The goal of this model was to answer this question. It takes data which would normally be measured in a regular check up such as age, sex, blood pressure, BMI, HbA1c, cholesterol, and a handful of other lab inputs (18 total). It then ranks patients according to how likely they are to have protein leaking in their urine. It doesn't use any kidney test results, since the whole point is to find people who haven't been tested yet. I trained the model on data from the CDC national survey (NHANES) from 2011-2020, and tested it on newer data from 2021 to 2023 that it was never trained on.

What I observed: today's rule is effectively "test the people with diagnosed high blood pressure." When both my model and the rule tested the same 23.1% of adults, today's rule detects 40.3% of people with protein in their urine, whereas my model detects 52.9% of the people who have protein. That is about 31% more cases with the same number of tests (95% CI for the difference: 4.4 to 20.5 points; AUC 0.726). When I tested it once on older 2009-10 data that I didn't train it on, it still outperformed "today's rule," 45.9% to 36.1%.

However, all of this information is from survey data, where every person provided only one urine sample. A more valid diagnosis requires that the result remains high on a second follow-up test, and for that group, a small group consisting of 92 people who have a second urine sample, my model and the rule did about the same. It also has not been tested on actual hospital records yet, which is the next planned step.


## Later checks
- **Round 4 robustness checks** (`src/round4.py` → `results/res_round4.json`): model vs blood-pressure ranking among people without a hypertension diagnosis, a version with some labs masked, calibration by sex, equity confidence intervals and net benefit.
- **NHANES 2009–2010** (`src/score_2009.py` → `results/res_2009.json`; plan in `PLAN_2009.md`): a one-time, pre-specified check on a cycle never used in development. At the same testing volume, the model found 45.9% of single-sample albuminuria vs 36.1% for the high blood pressure rule (AUC 0.706). For albuminuria confirmed on a second sample, it did about the same as the rule.
- **Round 5** (`src/round5.py` → `results/res_round5.json`, exploratory, run after the main results): a 16-input model without waist size and uric acid, saved as `model/model_ehr_16inputs.txt` (51.5% found, AUC 0.707 on 2021–2023); ranking men and women separately at the rule's testing rate for each sex (57.2% found on 2021–2023, 46.0% on 2009–2010); and the 92 persistent 2009–2010 cases rescored without pulse, with the 16-input model and with the sex-matched list (none differed clearly from the rule).

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
| 10 | `round5.py` | 16-input model file, sex-matched testing, persistent-case checks |
| 11 | `figs.py` | Figures |

`model_search_v2.py` (run separately: `python model_search_v2.py train_matrix_v2.csv out.json`, several hours) is the training-only model search. `results/earlier_candidates_test_scores.csv` holds test-set scores from earlier development versions, kept because the paper reports them.

`results/` holds the outputs used in the paper. `model/model_main_racefree.txt` is the fitted LightGBM model; `model/model_ehr_16inputs.txt` is the 16-input version without waist size and uric acid.

## Not for clinical use
This is a research model built on survey data. It does not diagnose anyone and has not been tested on hospital records.
