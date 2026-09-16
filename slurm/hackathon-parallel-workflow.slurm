#!/bin/bash
# @author HCMUT HPC Summer School Hackathon Organizers
# @permission DO_NOT_EDIT
# @note Cluster/run wrapper; teams should configure via environment variables, not by editing this file.
#SBATCH --job-name=hackathon-agents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=logs/hackathon-agents-%j.out
#SBATCH --error=logs/hackathon-agents-%j.err

set -euo pipefail

# Site-provided deployment assets. Set these before sbatch; no private cluster
# paths or container recipes are embedded in the public artifact.
: "${CLIENT_IMAGE:?Set CLIENT_IMAGE to the Python workflow Apptainer image}"
: "${SERVER_IMAGE:?Set SERVER_IMAGE to an Apptainer image containing Ollama}"
: "${OLLAMA_CACHE:?Set OLLAMA_CACHE to a pre-staged Ollama model cache}"
: "${ALLOWED_MODELS_FILE:?Set ALLOWED_MODELS_FILE to the site model allow-list}"

export OLLAMA_PORT_CANDIDATES=${OLLAMA_PORT_CANDIDATES:-"11434 11435 11436 11437"}
if [[ -n "${OLLAMA_PORT:-}" ]]; then
  OLLAMA_PORT_CANDIDATES="${OLLAMA_PORT}"
fi

is_port_available() {
  local port="$1"
  python3 - "${port}" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("0.0.0.0", port))
    except OSError:
        sys.exit(1)
PY
}

choose_ollama_port() {
  local candidates=($OLLAMA_PORT_CANDIDATES)
  local count="${#candidates[@]}"
  local offset=0
  local idx=0
  local port=""

  if [[ "${count}" -eq 0 ]]; then
    echo "No Ollama port candidates were configured." >&2
    return 1
  fi

  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    offset=$((SLURM_JOB_ID % count))
  fi

  for idx in $(seq 0 $((count - 1))); do
    port="${candidates[$(((offset + idx) % count))]}"
    if is_port_available "${port}"; then
      echo "${port}"
      return 0
    fi
  done

  echo "No available Ollama port found in: ${OLLAMA_PORT_CANDIDATES}" >&2
  return 1
}

OLLAMA_PORT=$(choose_ollama_port)
export OLLAMA_PORT
export OLLAMA_BASE_URL=http://127.0.0.1:${OLLAMA_PORT}
export OLLAMA_HOST=0.0.0.0:${OLLAMA_PORT}
export OLLAMA_MODELS=/models/ollama
export OLLAMA_MAX_LOADED_MODELS=${OLLAMA_MAX_LOADED_MODELS:-2}
export OLLAMA_NUM_PARALLEL=${OLLAMA_NUM_PARALLEL:-4}
export OLLAMA_MAX_QUEUE=${OLLAMA_MAX_QUEUE:-64}
export OLLAMA_KEEP_ALIVE=${OLLAMA_KEEP_ALIVE:-30m}
export OLLAMA_CONTEXT_LENGTH=${OLLAMA_CONTEXT_LENGTH:-16384}

export TEAM_ID=${TEAM_ID:-local-team}
export TEAM_NAME=${TEAM_NAME:-Local Team}

mkdir -p logs results

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Ollama port candidates: ${OLLAMA_PORT_CANDIDATES}"
echo "Selected Ollama port: ${OLLAMA_PORT}"
echo "Ollama URL: ${OLLAMA_BASE_URL}"
echo "Max loaded models: ${OLLAMA_MAX_LOADED_MODELS}"
echo "Parallel requests per model: ${OLLAMA_NUM_PARALLEL}"
echo "Context length: ${OLLAMA_CONTEXT_LENGTH}"

if [[ -f "${ALLOWED_MODELS_FILE}" ]]; then
  echo "Allowed models:"
  cat "${ALLOWED_MODELS_FILE}"
fi

nvidia-smi || true

apptainer exec --nv \
  --bind ${OLLAMA_CACHE}:/models/ollama:ro \
  --env OLLAMA_MODELS=${OLLAMA_MODELS} \
  --env OLLAMA_HOST=${OLLAMA_HOST} \
  --env OLLAMA_MAX_LOADED_MODELS=${OLLAMA_MAX_LOADED_MODELS} \
  --env OLLAMA_NUM_PARALLEL=${OLLAMA_NUM_PARALLEL} \
  --env OLLAMA_MAX_QUEUE=${OLLAMA_MAX_QUEUE} \
  --env OLLAMA_KEEP_ALIVE=${OLLAMA_KEEP_ALIVE} \
  --env OLLAMA_CONTEXT_LENGTH=${OLLAMA_CONTEXT_LENGTH} \
  ${SERVER_IMAGE} \
  ollama serve > logs/ollama-server-${SLURM_JOB_ID}.log 2>&1 &

SERVER_PID=$!

cleanup() {
  kill ${SERVER_PID} 2>/dev/null || true
  wait ${SERVER_PID} 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for Ollama server..."
READY=0
for _ in $(seq 1 60); do
  if ! kill -0 ${SERVER_PID} 2>/dev/null; then
    break
  fi
  if apptainer exec ${CLIENT_IMAGE} python3 -c "import urllib.request; urllib.request.urlopen('${OLLAMA_BASE_URL}/api/tags', timeout=2).read()" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 2
done

if [[ "${READY}" != "1" ]]; then
  echo "Ollama server did not become ready. Last server log lines:"
  tail -n 80 logs/ollama-server-${SLURM_JOB_ID}.log || true
  exit 1
fi

WORKFLOW_ARGS=(
  --source-dir .
  --config configs/hackathon_workflow.json
  --output-dir results
  --team-id "${TEAM_ID}"
  --team-name "${TEAM_NAME}"
)

if [[ -n "${BACKEND_URL:-}" ]]; then
  WORKFLOW_ARGS+=(--submit-url "${BACKEND_URL}")
fi

apptainer exec --nv \
  --env OLLAMA_BASE_URL=${OLLAMA_BASE_URL} \
  --env TEAM_ID=${TEAM_ID} \
  --env TEAM_NAME="${TEAM_NAME}" \
  --env TEAM_TOKEN="${TEAM_TOKEN:-}" \
  --env BACKEND_URL="${BACKEND_URL:-}" \
  ${CLIENT_IMAGE} \
  python3 src/hackathon_workflow.py "${WORKFLOW_ARGS[@]}"

echo "Generated hackathon files:"
ls -lh results/*${SLURM_JOB_ID}* || true
