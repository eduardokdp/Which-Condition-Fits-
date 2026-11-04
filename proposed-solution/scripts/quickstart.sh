#!/usr/bin/env bash
# Quickstart: install -> test -> run the full pipeline on the bundled public Reddit sample -> print f(s).
#   git clone https://github.com/eduardokdp/Which-Condition-Fits-.git && cd Which-Condition-Fits-
#   bash quickstart.sh
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

if [ ! -d .venv ]; then "$PY" -m venv .venv; fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip >/dev/null 2>&1 || true
pip install -r requirements.txt

echo "== unit and reproducibility tests"
python -m unittest discover -s tests -t .

echo "== pipeline: prepare -> train -> evaluate -> predict (config/quickstart.yaml)"
python -m src.main all --config config/quickstart.yaml

EXPECTED="$(cat evaluation/results/quickstart/EXPECTED_SHA256)"
GOT="$(python -c "import json;print(json.load(open('evaluation/results/quickstart/metrics.json'))['predictions_sha256'])")"
if [ "$GOT" = "$EXPECTED" ]; then
  echo "== reproducibility check PASSED: predictions match the reported run ($GOT)"
else
  echo "== reproducibility check: hash differs ($GOT vs $EXPECTED); check library versions against requirements.txt"
fi

echo "== figures and visual report (evaluation/results/quickstart/REPORT.md)"
pip install -r requirements-viz.txt
python scripts/make_figures.py --config config/quickstart.yaml

echo "== your own description:"
python -m src.main predict --config config/quickstart.yaml \
  --text "I keep losing track of time, my heart races at work and I feel empty most days."
