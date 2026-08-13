#!/usr/bin/env bash
# Run the real multi-GPU correctness worker against an installed wheel.
set -euo pipefail

WHEEL="${1:?usage: $0 <wheel>}"
[[ -f "${WHEEL}" ]] || { echo "no such wheel: ${WHEEL}" >&2; exit 2; }

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
GPUS="${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)}"
WORLD_SIZE="$(awk -F, '{print NF}' <<< "${GPUS}")"
WORK="$(mktemp -d /tmp/fast_ulysses_preflight.XXXXXX)"
trap 'rm -rf "${WORK}"' EXIT

"${PYTHON}" -m venv "${WORK}/venv"
"${WORK}/venv/bin/pip" install "${WHEEL}"
cd "${WORK}"
CUDA_VISIBLE_DEVICES="${GPUS}" "${WORK}/venv/bin/torchrun" \
  --standalone --nproc_per_node="${WORLD_SIZE}" \
  "${REPO}/test/distributed/correctness.py"
