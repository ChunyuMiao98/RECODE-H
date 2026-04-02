#!/bin/bash
set -euo pipefail

# ==== READ ENV NAME PARAMETER ====
if [ $# -lt 1 ]; then
  echo "Usage: $0 <env_name> [python_version] [requirements_file]"
  echo "  env_name         = name of the new conda environment"
  echo "  python_version   = (optional) default 3.10"
  echo "  requirements_file= (optional) default requirements.txt"
  exit 1
fi

ENV_NAME="$1"
PY_VER="${2:-3.10}"
REQ_FILE="${3:-requirements.txt}"

LOG_DIR="install_logs"
SUCCESS_LOG="$LOG_DIR/success.log"
FAIL_LOG="$LOG_DIR/failures.log"
FULL_LOG="$LOG_DIR/full_output.log"

mkdir -p "$LOG_DIR"
: > "$SUCCESS_LOG"
: > "$FAIL_LOG"
: > "$FULL_LOG"
source ~/miniconda3/etc/profile.d/conda.sh
# ==== LOAD CONDA (Miniconda) ====
if ! command -v conda >/dev/null 2>&1; then
  if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
  else
    echo "❌ Could not find conda.sh. Ensure Miniconda is installed and try again."
    exit 1
  fi
fi

# ==== CREATE ENV ====
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "ℹ️  Conda env '$ENV_NAME' already exists. Skipping creation."
else
  echo "🚀 Creating conda env '$ENV_NAME' with Python $PY_VER ..."
  conda create -n "$ENV_NAME" "python=$PY_VER" -y | tee -a "$FULL_LOG"
fi

# ==== ACTIVATE ENV ====
echo "🔧 Activating env '$ENV_NAME' ..."
conda activate "$ENV_NAME"

# Ensure pip is fresh
python -m pip install --upgrade pip setuptools wheel | tee -a "$FULL_LOG"

# Keep setuptools pinned for test_common1 to ensure pkg_resources works.
if [[ "$ENV_NAME" == "test_common1" ]]; then
  python -m pip install --force-reinstall --no-cache-dir "setuptools==73.0.1" | tee -a "$FULL_LOG"
fi

pip_install_one() {
  local pkg="$1"

  # PyTorch wheels with local version tags (+cpu / +cuXXX) are not on the default PyPI index.
  # Use the official PyTorch wheel index as an additional source for those packages only.
  case "$pkg" in
    torch==*+cpu|torchvision==*+cpu|torchaudio==*+cpu)
      python -m pip install "$pkg" --extra-index-url https://download.pytorch.org/whl/cpu
      return $?
      ;;
    torch==*+cu*|torchvision==*+cu*|torchaudio==*+cu*)
      local cu="${pkg##*+cu}"
      python -m pip install "$pkg" --extra-index-url "https://download.pytorch.org/whl/cu${cu}"
      return $?
      ;;
    *)
      python -m pip install "$pkg"
      return $?
      ;;
  esac
}
# ==== INSTALL ONE BY ONE ====
if [ ! -f "$REQ_FILE" ]; then
  echo "❌ Requirements file '$REQ_FILE' not found!"
  exit 1
fi

echo "📦 Installing packages from '$REQ_FILE' one by one..."
while IFS= read -r pkg || [[ -n "${pkg:-}" ]]; do
  pkg="$(echo "$pkg" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
  [[ -z "$pkg" || "$pkg" =~ ^# ]] && continue

  echo "====================================" | tee -a "$FULL_LOG"
  echo "Installing: $pkg" | tee -a "$FULL_LOG"
  echo "------------------------------------" | tee -a "$FULL_LOG"

  if pip_install_one "$pkg" 2>&1 | tee -a "$FULL_LOG"; then
    echo "✅ OK: $pkg" | tee -a "$SUCCESS_LOG"
  else
    echo "❌ FAIL: $pkg" | tee -a "$FAIL_LOG"
  fi

  echo "" | tee -a "$FULL_LOG"
done < "$REQ_FILE"

# ==== OPTIONAL: EXTRA WHEELS FOR SPECIFIC ENVS ====
# NOTE:
# The DGL wheels-test index below is tied to a specific PyTorch+CUDA combo.
# To avoid breaking the env, we gate it behind an explicit flag and a torch version check.
if [[ "$ENV_NAME" == "test_981" ]]; then
  if [[ "${INSTALL_DGL_WHEELS_TEST:-0}" == "1" ]]; then
    TORCH_VER="$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || true)"
    echo "====================================" | tee -a "$FULL_LOG"
    echo "Optional install (test_981): DGL pre-release wheels-test" | tee -a "$FULL_LOG"
    echo "torch version: ${TORCH_VER:-<unknown>}" | tee -a "$FULL_LOG"
    echo "------------------------------------" | tee -a "$FULL_LOG"
    if [[ "$TORCH_VER" == 2.4.* ]]; then
      python -m pip install --pre dgl -f https://data.dgl.ai/wheels-test/torch-2.4/cu124/repo.html 2>&1 | tee -a "$FULL_LOG"
    else
      echo "⚠️  Skipping DGL wheels-test install: requires torch 2.4.*, got '${TORCH_VER:-unknown}'." | tee -a "$FULL_LOG"
      echo "    If you really want this, align your torch version (e.g. requirements) to 2.4.* first." | tee -a "$FULL_LOG"
    fi
    echo "" | tee -a "$FULL_LOG"
  else
    echo "ℹ️  (test_981) DGL wheels-test step is disabled. To enable: export INSTALL_DGL_WHEELS_TEST=1" | tee -a "$FULL_LOG"
  fi
fi

echo
echo "🎉 Done."
echo "   Successes: $SUCCESS_LOG"
echo "   Failures : $FAIL_LOG"
echo "   Full log : $FULL_LOG"
