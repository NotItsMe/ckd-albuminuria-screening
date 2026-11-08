#!/usr/bin/env bash
# Rebuilds every result in results/ from the public NHANES files.
# Usage: NHANES_DIR=/path/to/xpt/files bash run_all.sh   (default: ./data)
set -e
export NHANES_DIR="${NHANES_DIR:-$(pwd)/data}"
cd src
python part_a.py            # 2021-2023 test cohort + Bragg-Gresham scores
python build_raw.py         # 2011-Mar 2020 training adults
python build_features.py    # 18 model inputs for training
python part_d_muntner.py    # Muntner score for the test cohort
python analysis.py          # fit final model, comparators, cross-validation, time check
python metrics.py           # main results with Rao-Wu bootstrap CIs
python extra.py             # bedside table, equity, calibration, decision curve, SHAP
python extra2.py            # severe albuminuria, Bragg-Gresham on its own population, trends
python round4.py            # later robustness checks
python score_2009.py        # one-time check on NHANES 2009-2010 with the frozen model
python round5.py           # 16-input model, sex-matched testing, persistent-case checks (exploratory)
python figs.py              # figures into src/figs2/
echo "Done. Compare src/res_*.json with results/."
