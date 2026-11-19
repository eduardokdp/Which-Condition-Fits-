#!/usr/bin/env bash
# Milestone 3 quickstart: install -> tests -> pipeline on the bundled Reddit sample -> hash check ->
# figures -> one prediction. Works from any folder:
#   cd Which-Condition-Fits-/proposed-solution && bash quickstart.sh
#   (also works from the repository root: bash quickstart.sh)
set -euo pipefail
# shellcheck source=scripts/_env.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/_env.sh"

echo "== unit, protocol and reproducibility tests"
python -m unittest discover -s tests -t .

echo "== pipeline: prepare -> train -> evaluate -> predict (config/quickstart.yaml)"
python -m src.main all --config config/quickstart.yaml

GOT="$(python -c "import json;print(json.load(open('evaluation/results/quickstart/metrics.json'))['predictions_sha256'])")"
check_hash evaluation/results/quickstart/EXPECTED_SHA256 "$GOT" "Milestone 3 quickstart"

echo "== figures and visual report"
python scripts/make_figures.py --config config/quickstart.yaml

echo "== your own description:"
python -m src.main predict --config config/quickstart.yaml \
  --text "I keep losing track of time, my heart races at work and I feel empty most days."
echo "== done. Results: evaluation/results/quickstart/ (open REPORT.md). Milestone 4: bash reproduce_m4.sh"
