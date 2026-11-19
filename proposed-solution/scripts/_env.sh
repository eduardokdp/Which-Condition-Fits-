# Shared setup for quickstart.sh and reproduce_m4.sh (sourced, not run directly).
# 1. moves to proposed-solution/ (the folder that holds requirements*.txt, src/ and config/);
# 2. picks a Python >= 3.10 (prefers 3.13 or 3.12, which can install the exact versions);
# 3. creates .venv if needed and activates it (Linux, macOS, WSL, or Git Bash on Windows);
# 4. installs requirements-lock.txt (exact, Python 3.12/3.13) or requirements.txt (Python 3.10+).
# Override the interpreter with:  PYTHON=/path/to/python bash quickstart.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
echo "== working folder: $ROOT"

_py_ok() {  # true if "$@" is a Python >= 3.10
  "$@" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1
}

if [ ! -x .venv/bin/python ] && [ ! -x .venv/Scripts/python.exe ]; then
  PYCMD=()
  for c in ${PYTHON:+"$PYTHON"} python3.13 python3.12 python3 python; do
    if command -v "$c" >/dev/null 2>&1 && _py_ok "$c"; then PYCMD=("$c"); break; fi
  done
  if [ ${#PYCMD[@]} -eq 0 ] && command -v py >/dev/null 2>&1 && _py_ok py -3; then PYCMD=(py -3); fi
  if [ ${#PYCMD[@]} -eq 0 ]; then
    echo "ERROR: Python 3.10 or newer was not found. Install Python 3.12 or 3.13 (python.org, or" >&2
    echo "       'sudo apt install python3' on Ubuntu/WSL), or set PYTHON=/path/to/python." >&2
    exit 1
  fi
  echo "== creating .venv with $("${PYCMD[@]}" --version 2>&1)"
  if ! "${PYCMD[@]}" -m venv .venv; then
    rm -rf .venv
    echo "ERROR: could not create the virtual environment. On Ubuntu/Debian/WSL install the venv module" >&2
    echo "       first:  sudo apt install python3-venv   (or python3.X-venv for your version), then rerun." >&2
    exit 1
  fi
fi

# shellcheck disable=SC1091
if [ -f .venv/Scripts/activate ]; then source .venv/Scripts/activate; else source .venv/bin/activate; fi
PYV="$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
echo "== using $(python --version 2>&1) from .venv"
python -m pip install --upgrade pip >/dev/null 2>&1 || true

EXACT=0
case "$PYV" in
  3.12|3.13)
    echo "== installing the exact versions (requirements-lock.txt)"
    if python -m pip install -r requirements-lock.txt; then EXACT=1; else
      echo "== exact versions not installable here; installing compatible versions instead"
    fi ;;
  *)
    echo "== Python $PYV: installing compatible versions (requirements.txt); the exact versions need 3.12 or 3.13" ;;
esac
if [ "$EXACT" -eq 0 ]; then
  python -m pip install -r requirements.txt -r requirements-viz.txt
fi
export CORE_EXACT_VERSIONS="$EXACT"

check_hash() {  # $1 = expected-hash file, $2 = observed hash, $3 = label
  local expected
  expected="$(cut -d' ' -f1 "$1" | tr -d '\r\n')"
  if [ "$2" = "$expected" ]; then
    echo "== reproducibility check PASSED ($3): identical to the reported run ($2)"
  elif [ "$CORE_EXACT_VERSIONS" = "1" ]; then
    echo "== reproducibility check ($3): hash $2 differs from $expected."
    echo "   Compare 'python -m pip list' with requirements-lock.txt."
  else
    echo "== reproducibility check ($3): hash differs, as expected with versions other than"
    echo "   requirements-lock.txt (Python $PYV). The run completed; numbers may differ in the last decimals."
  fi
}
