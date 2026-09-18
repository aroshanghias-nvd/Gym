#!/usr/bin/env bash
# Reproduce the candidate wheel of the pinned VisGym revision.
#
# Why a wheel instead of the git URL in requirements.txt: VisGym inherits
# Gymnasium's pyproject.toml, which declares both `classic-control` and
# `classic_control` (likewise `mujoco-py`/`mujoco_py`, `toy-text`/`toy_text`).
# PEP 685 normalizes those to one name, so uv refuses to parse the project:
#
#   TOML parse error ... duplicate normalized extra name `classic-control`
#
# NeMo-Gym builds every resource-server venv with `uv pip install`, so the
# source install fails there while pip -- which still tolerates the duplicates
# -- builds it fine. Building a source-pinned candidate wheel sidesteps the parse: wheel
# metadata is already normalized, and uv installs the result happily.
# Delete this script once VisGym drops the duplicate extras upstream and
# requirements.txt can name the git revision directly.
set -euo pipefail

VISGYM_REV="927271d107ad0196ad6aa597095ca57d01c6ddbb"
VISGYM_URL="https://github.com/visgym/VIsGym.git"
VISGYM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-${VISGYM_ROOT}/vendor_wheels}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BUILD_CONSTRAINTS="${VISGYM_ROOT}/scripts/visgym-wheel-build-constraints.txt"
GYMNASIUM_LICENSE="${VISGYM_ROOT}/vendor_licenses/gymnasium-v1.1.1-LICENSE"
GYMNASIUM_LICENSE_SHA256="7dacaa9772e856aee6943b32ef663d3634d91d72ec7bbc74d136943673f91e18"
EXPECTED_WHEEL_SHA256="8b2339261037f409b17b6487a1a5880cbfe0cf79f41b350c3259cc4b4f46d67e"

# A local checkout is used when given; otherwise the pinned revision is cloned
# into a temporary directory.
SRC_DIR="${VISGYM_REPO_ROOT:-}"
CLEANUP_DIR=""
if [[ -z "${SRC_DIR}" ]]; then
  CLEANUP_DIR="$(mktemp -d)"
  trap 'rm -rf "${CLEANUP_DIR}"' EXIT
  SRC_DIR="${CLEANUP_DIR}/VIsGym"
  git clone --quiet "${VISGYM_URL}" "${SRC_DIR}"
  git -C "${SRC_DIR}" checkout --quiet "${VISGYM_REV}"
fi

if ! "${PYTHON_BIN}" -m pip --version >/dev/null 2>&1; then
  echo "pip is required to build the VisGym wheel (uv cannot parse its pyproject)." >&2
  echo "Point PYTHON_BIN at an interpreter that has pip, e.g. PYTHON_BIN=/usr/bin/python3." >&2
  exit 2
fi
if ! "${PYTHON_BIN}" -c 'import sys; raise SystemExit(sys.version_info < (3, 13))'; then
  echo "Python 3.13 or newer is required to reproduce the pinned VisGym wheel." >&2
  exit 2
fi

if ! "${PYTHON_BIN}" -m pip wheel --help | grep -q -- "--build-constraint"; then
  echo "pip with --build-constraint support is required to reproduce the VisGym wheel." >&2
  echo "Point PYTHON_BIN at an interpreter with pip >= 25.3." >&2
  exit 2
fi

if [[ ! -f "${GYMNASIUM_LICENSE}" ]]; then
  echo "Missing the verified Gymnasium v1.1.1 license: ${GYMNASIUM_LICENSE}" >&2
  exit 2
fi
ACTUAL_LICENSE_SHA256="$(
  "${PYTHON_BIN}" -c \
    'import hashlib, pathlib, sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' \
    "${GYMNASIUM_LICENSE}"
)"
if [[ "${ACTUAL_LICENSE_SHA256}" != "${GYMNASIUM_LICENSE_SHA256}" ]]; then
  echo "Gymnasium license digest is ${ACTUAL_LICENSE_SHA256}; expected ${GYMNASIUM_LICENSE_SHA256}." >&2
  exit 2
fi

if ! ACTUAL_REV="$(git -C "${SRC_DIR}" rev-parse HEAD 2>/dev/null)"; then
  echo "VisGym source must be a git checkout so its revision can be verified: ${SRC_DIR}" >&2
  exit 2
fi
if [[ "${ACTUAL_REV}" != "${VISGYM_REV}" ]]; then
  echo "VisGym source is at ${ACTUAL_REV}; expected ${VISGYM_REV}." >&2
  exit 2
fi
if [[ -n "$(git -C "${SRC_DIR}" status --porcelain --untracked-files=all)" ]]; then
  echo "VisGym source checkout is dirty; refusing to build an unreviewed wheel." >&2
  exit 2
fi

# Pin both archive timestamps and the isolated build backend. This makes two
# builds from the same clean commit byte-for-byte reproducible.
SOURCE_DATE_EPOCH="$(git -C "${SRC_DIR}" show -s --format=%ct "${VISGYM_REV}")"
export SOURCE_DATE_EPOCH PYTHONHASHSEED=0 TZ=UTC

BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "${BUILD_DIR}" "${CLEANUP_DIR}"' EXIT
BUILD_SOURCE_DIR="${BUILD_DIR}/source"
mkdir -p "${BUILD_SOURCE_DIR}"

# VisGym's squash-created repository carries an Apache-2.0 LICENSE/NOTICE while
# retaining Gymnasium's MIT package metadata and substantial Gymnasium v1.1.1
# code. It does not retain Gymnasium's MIT notice. Build from an exact archive
# copy and add the exact LICENSE from the official Gymnasium v1.1.1 tag so the
# resulting wheel preserves both sets of notices. This mechanical preservation
# does not resolve the upstream licensing inconsistency; see the provenance.
git -C "${SRC_DIR}" archive "${VISGYM_REV}" | tar -x -C "${BUILD_SOURCE_DIR}"
cp "${GYMNASIUM_LICENSE}" "${BUILD_SOURCE_DIR}/LICENSE-GYMNASIUM-v1.1.1"

"${PYTHON_BIN}" -m pip wheel \
  --no-deps \
  --build-constraint "${BUILD_CONSTRAINTS}" \
  --wheel-dir "${BUILD_DIR}" \
  "${BUILD_SOURCE_DIR}"

WHEEL_PATH="${BUILD_DIR}/gymnasium-1.1.1-py3-none-any.whl"
if [[ ! -f "${WHEEL_PATH}" ]]; then
  echo "Build did not produce the expected wheel: ${WHEEL_PATH}" >&2
  exit 2
fi
ACTUAL_WHEEL_SHA256="$(
  "${PYTHON_BIN}" -c \
    'import hashlib, pathlib, sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' \
    "${WHEEL_PATH}"
)"
if [[ "${ACTUAL_WHEEL_SHA256}" != "${EXPECTED_WHEEL_SHA256}" ]]; then
  echo "Built wheel digest is ${ACTUAL_WHEEL_SHA256}; expected ${EXPECTED_WHEEL_SHA256}." >&2
  echo "Use the Python/pip/setuptools versions recorded in the provenance file." >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"
cp "${WHEEL_PATH}" "${OUT_DIR}/"

echo "Wrote VisGym wheel to ${OUT_DIR}:"
ls -1 "${OUT_DIR}/gymnasium-1.1.1-py3-none-any.whl"
