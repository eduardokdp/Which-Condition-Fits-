#!/usr/bin/env bash
# Milestone 4: reproduce every number, table and figure of experiments/4_Evaluation_*.pdf.
# Works from any folder; these three commands are equivalent:
#   bash reproduce_m4.sh                     (repository root)
#   bash reproduce_m4.sh                     (proposed-solution/)
#   bash scripts/reproduce_m4.sh             (proposed-solution/)
# Steps: install -> tests -> 5-fold CV grid + single test evaluation on the bundled sample -> hash check
# -> figures -> report tables; then the same on the Reddit Mental Health Dataset if it was downloaded.
set -euo pipefail
# shellcheck source=scripts/_env.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"

echo "== unit, protocol and reproducibility tests"
python -m unittest discover -s tests -t .

run() {  # $1 = config name: m4_sample | m4_rmhd
  local res="evaluation/results/$1"
  echo "== experiment: config/$1.yaml (68 configurations x 5 folds, then one test evaluation)"
  python -m src.main experiment --config "config/$1.yaml"
  check_hash "$res/EXPECTED_SHA256" \
    "$(python -c "import json;print(json.load(open('$res/test_metrics.json'))['test_predictions_sha256'])")" "$1"
  python scripts/make_m4_figures.py --results "$res"
}

run m4_sample
python scripts/make_m4_tables.py --results evaluation/results/m4_sample --out ../experiments/tables

if ls data/raw/*_2019_features_tfidf_256.csv >/dev/null 2>&1; then
  run m4_rmhd
else
  echo "== (optional) Reddit Mental Health Dataset not in data/raw/: bash scripts/download_data.sh, then rerun"
fi
echo "== done. Results: evaluation/results/m4_sample/  Report tables: ../experiments/tables/"
