# Choosing who gets a urine albumin test (NHANES 2011–2023)

**Research use only, not for clinical use.**

Code for the paper *The Urine Test Nobody Orders: Using common clinical measurements to decide who gets a urine albumin test* (Zaid Mohammed, November 2026).

The model ranks U.S. adults without diabetes and with eGFR ≥ 60 by their chance of a urine albumin-to-creatinine ratio (UACR) ≥ 30 mg/g on a single sample, using 18 routine inputs (no kidney-damage markers, no race). It is compared with testing everyone with diagnosed high blood pressure, Bragg-Gresham Model 1 and the Muntner score at the same testing volume.

**Main result (Aug 2021–Aug 2023 test data):** 52.9% of single-sample albuminuria found vs 40.3% for the high blood pressure rule when both test 23.1% of adults (gain 12.6 points, 95% CI 4.4–20.5); AUC 0.726 (0.692–0.760). The model has not been validated on clinical records.

## Results
- **Round 4 checks:** extra analyses requested by reviewers are in `src/round4.py`, with outputs in `results/res_round4.json`.
- **One-time check on NHANES 2009–2010:** the analysis plan (`PLAN_2009.md`) was written before any scoring, and the frozen model was scored once (`src/score_2009.py`, `results/res_2009.json`). This cycle was not used in development. With both testing 22.5% of adults, the model found 45.9% of single-sample albuminuria vs 36.1% for the high blood pressure rule (gain 9.8 points, 95% CI 1.9–17.2). For albuminuria confirmed on a second sample, the model did about the same as the rule: 54.4% vs 53.1% (difference 95% CI −17.6 to 16.5; only 92 confirmed cases).

## Data
All data are public, de-identified NHANES files from the CDC (not included here). Download the XPT files for cycles G (2011–12), H (2013–14), I (2015–16), P (2017–Mar 2020) and L (Aug 2021–Aug 2023): DEMO, ALB_CR, BIOPRO, GHB, DIQ, BPQ, MCQ, SMQ, HDL, BMX, BPX/BPXO, RXQ_RX, HUQ_L. Set the folders at the top of `src/build_raw.py`.

## Run order
0. `src/part_a.py` then `src/part_d_muntner.py` — build the 2021–2023 test cohort (`cohort_2021_2023.csv`) with Bragg-Gresham Model 1 and Muntner scores.
1. `src/build_raw.py` — builds the study groups (adults ≥ 18, examined, no diabetes, CKD-EPI 2021 eGFR ≥ 60).
2. `src/build_features.py` — training matrix (blood pressure = mean of first three readings; pulse = first available reading).
3. `src/analysis.py` — fits the final model, comparators and sensitivity versions; cross-validation and time check.
4. `src/metrics.py`, `src/extra.py`, `src/extra2.py` — survey-weighted results with Rao–Wu bootstrap (2,000 replicates), bedside table, calibration, decision curves, equity check, SHAP.
5. `src/figs.py` — figures. `src/model_search_v2.py` — training-only model search.

`results/` holds the outputs used in the paper. `model/model_main_racefree.txt` is the fitted LightGBM model.

## Not for clinical use
This is a research model built on survey data. It does not diagnose anyone and has not been tested on hospital records.