#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_YML="${ENV_YML:-${ROOT}/env/environment-release.yml}"
ENV_ROOT="${ENV_ROOT:-._envs}"
ENV_PREFIX="${ENV_PREFIX:-${ENV_ROOT}/robojudge-iclr2027-release}"
CONDA_ROOT="${CONDA_ROOT:-${ENV_ROOT}/miniforge3-24.11.3-2}"
CONDA_EXE="${CONDA_EXE:-}"
MINIFORGE_URL="${MINIFORGE_URL:-https://github.com/conda-forge/miniforge/releases/download/24.11.3-2/Miniforge3-24.11.3-2-Linux-x86_64.sh}"
MINIFORGE_SHA256="${MINIFORGE_SHA256:-}"
INSTALLER="${ENV_ROOT}/$(basename "${MINIFORGE_URL}")"

mkdir -p "${ENV_ROOT}"

if [[ -z "${CONDA_EXE}" ]]; then
  for candidate in conda mamba micromamba /opt/conda/bin/conda /usr/local/conda/bin/conda "${HOME}/miniconda3/bin/conda" "${HOME}/miniforge3/bin/conda" "${CONDA_ROOT}/bin/conda"; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      CONDA_EXE="$(command -v "${candidate}")"
      break
    elif [[ -x "${candidate}" ]]; then
      CONDA_EXE="${candidate}"
      break
    fi
  done
fi

if [[ -z "${CONDA_EXE}" ]]; then
  if [[ ! -x "${CONDA_ROOT}/bin/conda" ]]; then
    curl -L "${MINIFORGE_URL}" -o "${INSTALLER}"
    actual_sha="$(sha256sum "${INSTALLER}" | awk '{print $1}')"
    if [[ -n "${MINIFORGE_SHA256}" && "${actual_sha}" != "${MINIFORGE_SHA256}" ]]; then
      echo "ERROR: Miniforge sha256 mismatch: got ${actual_sha}, expected ${MINIFORGE_SHA256}" >&2
      exit 2
    fi
    bash "${INSTALLER}" -b -p "${CONDA_ROOT}"
    printf '%s  %s\n' "${actual_sha}" "${INSTALLER}" > "${ENV_ROOT}/miniforge_installer.sha256"
  fi
  CONDA_EXE="${CONDA_ROOT}/bin/conda"
fi

"${CONDA_EXE}" env remove -p "${ENV_PREFIX}" -y >/dev/null 2>&1 || true
"${CONDA_EXE}" env create -p "${ENV_PREFIX}" -f "${ENV_YML}"
"${CONDA_EXE}" run -p "${ENV_PREFIX}" python -m pip freeze | sort > "${ENV_ROOT}/robojudge-iclr2027-release.pip-freeze.txt"
"${CONDA_EXE}" run -p "${ENV_PREFIX}" python - <<'PY'
import json, platform, sys
mods = ["torch", "torchvision", "transformers", "qwen_vl_utils", "decord", "datasets", "trl", "deepspeed", "modelscope"]
versions = {"python": sys.version, "platform": platform.platform()}
for name in mods:
    try:
        mod = __import__(name)
        versions[name] = getattr(mod, "__version__", "unknown")
    except Exception as exc:
        versions[name] = f"IMPORT_ERROR: {exc!r}"
print(json.dumps(versions, indent=2, sort_keys=True))
PY
