#!/usr/bin/env bash
# Build an isolated environment and run the MiniMax H3 packing A/B/C on RTX PRO 5000.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FAST_ULYSSES_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$(cd -- "${FAST_ULYSSES_ROOT}/.." && pwd)}"
ACTION="${1:-all}"

VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-${WORK_ROOT}/vllm-omni-fast-ulysses}"
VLLM_OMNI_REPO="${VLLM_OMNI_REPO:-https://github.com/lishunyang12/vllm-omni.git}"
VLLM_OMNI_BRANCH="${VLLM_OMNI_BRANCH:-feat/fast-ulysses-transport-v026}"
MODEL_ROOT="${MODEL_ROOT:-${WORK_ROOT}/MiniMax-H3}"
GPU_IDS="${GPU_IDS:-0,2,1,3}"
NUMA_NODE="${NUMA_NODE:-0}"
TP_SIZE="${TP_SIZE:-2}"
ULYSSES_DEGREE="${ULYSSES_DEGREE:-2}"
MICRO_RUNS="${MICRO_RUNS:-5}"
MICRO_ITERS="${MICRO_ITERS:-200}"
MICRO_WARMUP="${MICRO_WARMUP:-50}"
RUN_LEVEL="${RUN_LEVEL:-screen}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULT_ROOT="${RESULT_ROOT:-${WORK_ROOT}/results/h3-packing-${STAMP}}"

export HF_HOME="${HF_HOME:-${WORK_ROOT}/hf-cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${WORK_ROOT}/uv-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${WORK_ROOT}/xdg-cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${WORK_ROOT}/triton-cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${WORK_ROOT}/torchinductor-cache}"
export PATH="${WORK_ROOT}/bin:${WORK_ROOT}/ffmpeg-tools:${WORK_ROOT}/ffmpeg-tools/bin:${WORK_ROOT}/ffmpeg-shared/bin:${PATH}"

if [[ -x "${WORK_ROOT}/bin/uv" ]]; then
  UV="${UV:-${WORK_ROOT}/bin/uv}"
else
  UV="${UV:-$(command -v uv || true)}"
fi

usage() {
  cat <<'EOF'
Usage: run_pro5000_suite.sh [setup|microbench|e2e|all]

Defaults match the validated socket-0 RTX PRO 5000 layout:
  GPU_IDS=0,2,1,3  -> Ulysses pairs are physical (0,1) and (2,3)
  TP_SIZE=2, ULYSSES_DEGREE=2, NUMA_NODE=0

RUN_LEVEL=screen uses 5 denoise steps. RUN_LEVEL=full uses 50 steps.
Override RESULT_ROOT to append later phases to an existing result directory.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

capture_machine() {
  local metadata_dir="${RESULT_ROOT}/metadata"
  mkdir -p "${metadata_dir}"
  nvidia-smi >"${metadata_dir}/nvidia-smi.txt"
  nvidia-smi topo -m >"${metadata_dir}/topology.txt"
  nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,driver_version \
    --format=csv >"${metadata_dir}/gpus.csv"
  lscpu >"${metadata_dir}/lscpu.txt"
  numactl --hardware >"${metadata_dir}/numa.txt"
  git -C "${FAST_ULYSSES_ROOT}" rev-parse HEAD >"${metadata_dir}/fast-ulysses.commit"
  printf '%s\n' "${GPU_IDS}" >"${metadata_dir}/gpu-order.txt"
  printf 'TP_SIZE=%s\nULYSSES_DEGREE=%s\nNUMA_NODE=%s\n' \
    "${TP_SIZE}" "${ULYSSES_DEGREE}" "${NUMA_NODE}" >"${metadata_dir}/parallelism.env"
}

setup_env() {
  [[ -n "${UV}" ]] || die "uv was not found; expected ${WORK_ROOT}/bin/uv or uv on PATH"
  require_command git
  require_command nvidia-smi
  require_command numactl
  require_command curl
  require_command ffmpeg
  require_command ffprobe

  mkdir -p "${RESULT_ROOT}" "${HF_HOME}" "${UV_CACHE_DIR}" "${XDG_CACHE_HOME}" \
    "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
  capture_machine

  if [[ ! -d "${VLLM_OMNI_DIR}/.git" ]]; then
    git clone --branch "${VLLM_OMNI_BRANCH}" --single-branch \
      "${VLLM_OMNI_REPO}" "${VLLM_OMNI_DIR}"
  else
    [[ -z "$(git -C "${VLLM_OMNI_DIR}" status --short)" ]] || \
      die "${VLLM_OMNI_DIR} has local changes; refusing to update it"
    git -C "${VLLM_OMNI_DIR}" fetch origin "${VLLM_OMNI_BRANCH}"
    git -C "${VLLM_OMNI_DIR}" checkout "${VLLM_OMNI_BRANCH}"
    git -C "${VLLM_OMNI_DIR}" merge --ff-only "origin/${VLLM_OMNI_BRANCH}"
  fi

  if [[ ! -x "${VLLM_OMNI_DIR}/.venv/bin/python" ]]; then
    "${UV}" venv --python 3.12 --seed "${VLLM_OMNI_DIR}/.venv"
  fi
  export PATH="${VLLM_OMNI_DIR}/.venv/bin:${PATH}"

  "${UV}" pip install --python "${VLLM_OMNI_DIR}/.venv/bin/python" \
    "vllm==0.26.0" --torch-backend=auto
  "${UV}" pip install --python "${VLLM_OMNI_DIR}/.venv/bin/python" \
    -e "${VLLM_OMNI_DIR}"
  "${UV}" pip install --python "${VLLM_OMNI_DIR}/.venv/bin/python" \
    cmake ninja
  FAST_ULYSSES_CUDA_ARCH=120 "${UV}" pip install \
    --python "${VLLM_OMNI_DIR}/.venv/bin/python" --no-build-isolation \
    -e "${FAST_ULYSSES_ROOT}"

  git -C "${VLLM_OMNI_DIR}" rev-parse HEAD >"${RESULT_ROOT}/metadata/vllm-omni.commit"
  "${VLLM_OMNI_DIR}/.venv/bin/python" - \
    >"${RESULT_ROOT}/metadata/python-environment.txt" <<'PY'
import torch
import vllm
import vllm_omni
import fast_ulysses

print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("vllm", vllm.__version__)
print("vllm_omni", vllm_omni.__file__)
print("fast_ulysses", fast_ulysses.__version__, fast_ulysses.__file__)
print("capability", torch.cuda.get_device_capability())
PY
  cat "${RESULT_ROOT}/metadata/python-environment.txt"
}

run_microbench() {
  local torchrun="${VLLM_OMNI_DIR}/.venv/bin/torchrun"
  [[ -x "${torchrun}" ]] || die "environment missing; run '$0 setup' first"
  local output_dir="${RESULT_ROOT}/microbench"
  mkdir -p "${output_dir}"

  IFS=',' read -r -a ordered_gpus <<<"${GPU_IDS}"
  [[ "${#ordered_gpus[@]}" -eq 4 ]] || die "GPU_IDS must contain four physical GPU IDs"
  local pair0="${ordered_gpus[0]},${ordered_gpus[2]}"
  local pair1="${ordered_gpus[1]},${ordered_gpus[3]}"

  "${FAST_ULYSSES_ROOT}/tools/exclusive.sh" "${GPU_IDS}" -- \
    numactl --cpunodebind="${NUMA_NODE}" --membind="${NUMA_NODE}" \
    "${torchrun}" --standalone --nproc_per_node=4 \
    "${FAST_ULYSSES_ROOT}/benchmark/bench_a2a.py" \
    --mode link --allow-non-nvlink --iters "${MICRO_ITERS}" --warmup "${MICRO_WARMUP}" \
    2>&1 | tee "${output_dir}/link.log"

  for pair in "${pair0}" "${pair1}"; do
    local pair_label="${pair//,/-}"
    "${FAST_ULYSSES_ROOT}/tools/exclusive.sh" "${pair}" -- \
      numactl --cpunodebind="${NUMA_NODE}" --membind="${NUMA_NODE}" \
      "${torchrun}" --standalone --nproc_per_node=2 \
      "${FAST_ULYSSES_ROOT}/benchmark/bench_a2a.py" \
      --mode pcie-pretest --shape h3-t2va-5s --allow-non-nvlink \
      --iters "${MICRO_ITERS}" --warmup "${MICRO_WARMUP}" --host-mib 64 \
      2>&1 | tee "${output_dir}/decomposition-pair-${pair_label}.log"
  done

  for run in $(seq 1 "${MICRO_RUNS}"); do
    "${FAST_ULYSSES_ROOT}/tools/exclusive.sh" "${GPU_IDS}" -- \
      numactl --cpunodebind="${NUMA_NODE}" --membind="${NUMA_NODE}" \
      "${torchrun}" --standalone --nproc_per_node=4 \
      "${FAST_ULYSSES_ROOT}/benchmark/bench_a2a.py" \
      --mode h3-block --shape h3-t2va-5s --tensor-parallel-size "${TP_SIZE}" \
      --allow-non-nvlink --iters "${MICRO_ITERS}" --warmup "${MICRO_WARMUP}" \
      --blocks 50 --steps 50 --json-out "${output_dir}/h3-block-${run}.json" \
      2>&1 | tee "${output_dir}/h3-block-${run}.log"
  done

  "${VLLM_OMNI_DIR}/.venv/bin/python" "${SCRIPT_DIR}/summarize_h3_block.py" \
    "${output_dir}"/h3-block-*.json --output "${output_dir}/h3-block-summary.tsv"
}

run_e2e() {
  [[ -f "${MODEL_ROOT}/FL2VA/model_index.json" ]] || \
    die "MiniMax H3 FL2VA checkpoint not found under ${MODEL_ROOT}/FL2VA"
  local backend_script="${SCRIPT_DIR}/run_h3_e2e_backend.sh"
  local steps=5 warmups=2 runs=3
  if [[ "${RUN_LEVEL}" == "full" ]]; then
    steps=50
  elif [[ "${RUN_LEVEL}" != "screen" ]]; then
    die "RUN_LEVEL must be 'screen' or 'full'"
  fi

  local backends=(nccl pitched-owned pitched-zero packed-owned auto-zero)
  for backend in "${backends[@]}"; do
    "${FAST_ULYSSES_ROOT}/tools/exclusive.sh" "${GPU_IDS}" -- env \
      BACKEND="${backend}" WORK_ROOT="${WORK_ROOT}" MODEL_ROOT="${MODEL_ROOT}" \
      VLLM_OMNI_DIR="${VLLM_OMNI_DIR}" RESULT_ROOT="${RESULT_ROOT}" \
      NUMA_NODE="${NUMA_NODE}" TP_SIZE="${TP_SIZE}" ULYSSES_DEGREE="${ULYSSES_DEGREE}" \
      NUM_INFERENCE_STEPS="${steps}" WARMUPS="${warmups}" MEASURED_RUNS="${runs}" \
      bash "${backend_script}"
  done

  {
    printf 'backend\truns\tmean_seconds\n'
    for backend in "${backends[@]}"; do
      awk -v backend="${backend}" '
        {sum += $1; count += 1}
        END {printf "%s\t%d\t%.3f\n", backend, count, sum / count}
      ' "${RESULT_ROOT}/e2e/${backend}"/run-*.seconds
    done
  } | tee "${RESULT_ROOT}/e2e/summary.tsv"

  local video_checks=() audio_checks=()
  for backend in "${backends[@]}"; do
    video_checks+=("${RESULT_ROOT}/e2e/${backend}/run-1.video.framemd5")
    audio_checks+=("${RESULT_ROOT}/e2e/${backend}/run-1.audio.framemd5")
  done
  sha256sum "${video_checks[@]}" | tee "${RESULT_ROOT}/e2e/video-correctness.sha256"
  sha256sum "${audio_checks[@]}" | tee "${RESULT_ROOT}/e2e/audio-correctness.sha256"

  [[ "$(awk '{print $1}' "${RESULT_ROOT}/e2e/video-correctness.sha256" | sort -u | wc -l)" -eq 1 ]] || \
    die "decoded video FrameMD5 differs across backends"
  [[ "$(awk '{print $1}' "${RESULT_ROOT}/e2e/audio-correctness.sha256" | sort -u | wc -l)" -eq 1 ]] || \
    die "decoded audio FrameMD5 differs across backends"
}

case "${ACTION}" in
  setup)
    setup_env
    ;;
  microbench)
    setup_env
    run_microbench
    ;;
  e2e)
    setup_env
    run_e2e
    ;;
  all)
    setup_env
    run_microbench
    run_e2e
    ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

echo "RESULT_ROOT=${RESULT_ROOT}"
