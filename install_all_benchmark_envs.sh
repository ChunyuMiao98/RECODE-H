#!/usr/bin/env bash
set -euo pipefail

# One-shot installer for all benchmark conda environments.
# - Reuses existing dataset requirements + install scripts where possible.
# - Applies known post-install fixes (non-PyPI deps, PyG CUDA wheels, extra utilities).
#
# Usage:
#   bash install_all_benchmark_envs.sh
#   RECREATE=1 bash install_all_benchmark_envs.sh
#
# Optional flags (env vars):
#   RECREATE=1        Remove envs before installing
#   ONLY_TEST_981=1   Only (re)install test_981

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="$ROOT_DIR/work_dir/dataset"

if [[ ! -d "$DATASET_DIR" ]]; then
  echo "[fatal] dataset dir not found: $DATASET_DIR" >&2
  exit 2
fi

# shellcheck disable=SC1091
source "$HOME/miniconda3/etc/profile.d/conda.sh"

timestamp="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT_DIR/install_all_logs/$timestamp"
mkdir -p "$LOG_DIR"

log() {
  echo "[$(date +%H:%M:%S)] $*"
}

env_exists() {
  conda env list | awk '{print $1}' | grep -qx "$1"
}

maybe_recreate_env() {
  local env_name="$1"
  if [[ "${RECREATE:-0}" == "1" ]] && env_exists "$env_name"; then
    log "[recreate] removing $env_name"
    conda env remove -n "$env_name" -y >/dev/null 2>&1 || conda remove -n "$env_name" --all -y >/dev/null 2>&1 || true
  fi
}

run_and_log() {
  local log_file="$1"
  shift
  # shellcheck disable=SC2068
  ( "$@" ) 2>&1 | tee -a "$log_file"
}

install_via_script() {
  local env_name="$1"
  local py_ver="$2"
  local req_file="$3"
  local script="$4"
  local log_file="$LOG_DIR/${env_name}.install.log"

  maybe_recreate_env "$env_name"

  log "[install] $env_name via $(basename "$script") req=$(basename "$req_file")"
  run_and_log "$log_file" bash "$script" "$env_name" "$py_ver" "$req_file"
}

post_install_common_specials() {
  # Some requirement sets include packages that are not available on PyPI as pinned.
  # We install known replacements.
  local env_name="$1"
  local log_file="$LOG_DIR/${env_name}.postfix.log"

  log "[postfix] $env_name (special packages)"
  run_and_log "$log_file" bash -lc "
    set -euo pipefail
    source \"$HOME/miniconda3/etc/profile.d/conda.sh\"
    conda activate \"$env_name\"
    # Keep setuptools pinned for test_common1 because some tasks rely on
    # `from pkg_resources import packaging` which newer setuptools can break.
    if [[ \"$env_name\" == \"test_common1\" ]]; then
      python -m pip install -U pip wheel
      python -m pip install --force-reinstall --no-cache-dir \"setuptools==73.0.1\"
    else
      python -m pip install -U pip setuptools wheel
    fi
    # clip==1.0 -> OpenAI CLIP
    python -m pip install \"git+https://github.com/openai/CLIP.git\"
    # mhnreact==1.0 -> ml-jku/mhn-react
    python -m pip install \"git+https://github.com/ml-jku/mhn-react.git\"
    conda deactivate
  "
}

post_install_common1_extras() {
  # Additional special cases found in test_common1 requirements.
  local env_name="test_common1"
  local log_file="$LOG_DIR/${env_name}.postfix.log"

  log "[postfix] $env_name (dassl/setuptools/dgl)"
  run_and_log "$log_file" bash -lc "
    set -euo pipefail
    source \"$HOME/miniconda3/etc/profile.d/conda.sh\"
    conda activate \"$env_name\"
    # dassl==0.6.3 -> Dassl.pytorch (needs numpy in build env; disable isolation)
    python -m pip install --no-build-isolation \"git+https://github.com/KaiyangZhou/Dassl.pytorch.git\"
    # Ensure pkg_resources and pkg_resources.packaging work (some tasks import them directly).
    python -m pip install --force-reinstall --no-cache-dir \"setuptools==73.0.1\"
    python -c \"import pkg_resources; from pkg_resources import packaging; import packaging as p; print('pkg_resources.packaging ok', p.__version__)\" >/dev/null
    # DGL is required by some tasks using this env; use 1.x to avoid GraphBolt dependency chain.
    python -m pip install --no-deps --force-reinstall --no-cache-dir \"dgl==1.1.3\"
    python -c \"import dgl; print('dgl', dgl.__version__)\" >/dev/null
    conda deactivate
  "
}

install_test_981_from_requirements_with_overrides() {
  # requirements_98.txt uses +cpu pins for torch/vision, which are often not on default PyPI.
  # If you have CUDA available, it's usually better to install the non-+cpu versions.
  local env_name="test_981"
  local py_ver="3.10"
  local req_file="$DATASET_DIR/requirements_98.txt"
  local log_file="$LOG_DIR/${env_name}.install.log"

  maybe_recreate_env "$env_name"

  log "[install] $env_name from $(basename "$req_file") with overrides"
  run_and_log "$log_file" bash -lc "
    set -euo pipefail
    source \"$HOME/miniconda3/etc/profile.d/conda.sh\"
    conda create -n \"$env_name\" \"python=$py_ver\" -y
    conda activate \"$env_name\"
    python -m pip install -U pip setuptools wheel

    # Build an install list from requirements, normalizing torch local tags (+cpu/+cuXXX).
    python - <<'PY' \"$req_file\" > /tmp/_req_981_normalized.txt
import re, sys
p = sys.argv[1]
out=[]
for raw in open(p,'r',encoding='utf-8',errors='replace'):
    s = raw.strip()
    if not s or s.startswith('#'):
        continue
    # Normalize torch/vision/audio local version tags to base version.
    s = re.sub(r'^(torch|torchvision|torchaudio)==([0-9]+(?:\\.[0-9]+)*)\\+(cpu|cu[0-9]+)$', r'\\1==\\2', s)
    out.append(s)
print('\\n'.join(out))
PY

    while IFS= read -r pkg; do
      [[ -z \"\${pkg:-}\" ]] && continue
      echo \"[pip] \$pkg\"
      python -m pip install \"\$pkg\"
    done < /tmp/_req_981_normalized.txt

    # Hard check: torch must be importable before we proceed.
    python -c \"import torch; print('torch', torch.__version__)\"

    # Extra utilities you said you rely on in test_981
    python -m pip install -U scikit-learn line-profiler matplotlib pyyaml

    # DGL is required by some benchmark tasks (e.g. annotation_41).
    # Use DGL 1.x to avoid pulling in GraphBolt/torchdata/pandas/pydantic/yaml dependency chain.
    python -m pip install --no-deps --force-reinstall --no-cache-dir \"dgl==1.1.3\"
    python -c \"import dgl; print('dgl', dgl.__version__)\" >/dev/null

    # Ensure PyG extensions match torch CUDA build (fixes 'Not compiled with CUDA support').
    python - <<'PY'
import os, re, subprocess, sys
import torch
ver = torch.__version__
m = re.match(r'^(\\d+\\.\\d+\\.\\d+)(?:\\+([a-z0-9]+))?$', ver)
if not m:
    print('[warn] cannot parse torch version:', ver)
    raise SystemExit(0)
base = m.group(1)
tag = m.group(2) or ''
if not tag.startswith('cu'):
    print('[info] torch is not a CUDA build:', ver, '-> skip PyG CUDA wheels')
    raise SystemExit(0)
index = f'https://data.pyg.org/whl/torch-{base}+{tag}.html'
print('[info] installing PyG CUDA wheels from', index)
pkgs = [
    'torch_scatter==2.1.2+pt28' + tag,
    'torch_sparse==0.6.18+pt28' + tag,
]
cmd = [sys.executable, '-m', 'pip', 'install', '--no-deps', '--force-reinstall', '-f', index] + pkgs
print('[cmd]', ' '.join(cmd))
subprocess.check_call(cmd)
PY

    conda deactivate
  "
}

sanity_check_env() {
  local env_name="$1"
  local log_file="$LOG_DIR/${env_name}.sanity.log"
  log "[sanity] $env_name"
  run_and_log "$log_file" bash -lc "
    set -euo pipefail
    source \"$HOME/miniconda3/etc/profile.d/conda.sh\"
    conda run -n \"$env_name\" python -c \"import sys; print(sys.version)\"
  "
}

main() {
  log "logs: $LOG_DIR"
  cd "$DATASET_DIR"

  # Fast path: only (re)install test_981
  if [[ "${ONLY_TEST_981:-0}" == "1" ]]; then
    install_test_981_from_requirements_with_overrides
    sanity_check_env "test_981"
    log "[done] ONLY_TEST_981=1"
    return 0
  fi

  # Keep the same set as work_dir/dataset/setup_envs.sh.
  install_via_script "test_newtransformer1" "3.10" "$DATASET_DIR/requirements_newtransformer.txt" "$DATASET_DIR/install_env_gpu.sh"
  install_via_script "test_281"            "3.10" "$DATASET_DIR/requirements_28.txt"            "$DATASET_DIR/install_env_gpu.sh"
  install_via_script "test_191"            "3.10" "$DATASET_DIR/requirements_19.txt"            "$DATASET_DIR/install_env_gpu.sh"
  install_via_script "test_common1"        "3.10" "$DATASET_DIR/requirements_test_common.txt"   "$DATASET_DIR/install_env_gpu.sh"
  install_via_script "test_151"            "3.10" "$DATASET_DIR/requirements_15.txt"            "$DATASET_DIR/install_env_gpu.sh"
  install_via_script "test_511"            "3.10" "$DATASET_DIR/requirements_51.txt"            "$DATASET_DIR/install_env_gpu.sh"
  install_via_script "test_631"            "3.10" "$DATASET_DIR/requirements_63.txt"            "$DATASET_DIR/install_env_gpu.sh"

  # test_981 needs overrides + extra tooling + PyG CUDA wheel alignment
  install_test_981_from_requirements_with_overrides

  # Post-install special packages for envs whose requirements often fail on PyPI.
  post_install_common_specials "test_newtransformer1"
  post_install_common_specials "test_common1"
  post_install_common1_extras

  # Sanity checks
  for e in test_newtransformer1 test_281 test_191 test_common1 test_151 test_511 test_631 test_981; do
    sanity_check_env "$e"
  done

  log "[done] all requested benchmark envs installed"
}

main "$@"

