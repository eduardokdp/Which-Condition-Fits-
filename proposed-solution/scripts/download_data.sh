#!/usr/bin/env bash
# Download the Reddit Mental Health Dataset files used by config/default.yaml.
# Low, D. M. et al. (2020). Zenodo, doi:10.5281/zenodo.3941387. License: ODC PDDL 1.0 (Reddit API terms apply).
# Cross-platform alternative: python -m src.main download --config config/default.yaml
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw
for c in adhd anxiety depression; do
  f="${c}_2019_features_tfidf_256.csv"
  if [ -s "data/raw/$f" ]; then echo "found data/raw/$f"; continue; fi
  wget -O "data/raw/$f" "https://zenodo.org/records/3941387/files/$f?download=1"
done
