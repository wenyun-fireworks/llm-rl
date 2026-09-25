#!/usr/bin/env bash
# Run one algorithm end to end on a single GPU.
#
# Trainer and vLLM engine are colocated on the same device because CUDA IPC weight
# transfer resolves handles by GPU UUID -- they must share a physical GPU.
#
#   ./scripts/launch_run.sh NAME CONFIG DEVICE PORT MAX_HOURS [overrides...]
set -uo pipefail

NAME="$1"; CONFIG="$2"; DEVICE="$3"; PORT="$4"; MAX_HOURS="$5"; shift 5
OVERRIDES=("$@")

cd "$(dirname "$0")/.."
mkdir -p logs runs
export PATH="$HOME/.local/bin:$PATH"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
# Do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here. It would help
# with the fragmentation that ragged batches cause, but expandable segments back
# allocations with virtual memory whose IPC export goes through pidfd_getfd, and
# the engine process is not permitted to call that on the trainer. The weight
# transfer fails with "pidfd_getfd: Operation not permitted" at the first sync.

SERVER_LOG="logs/${NAME}_server.log"
TRAIN_LOG="logs/${NAME}_train.log"

echo "[$NAME] starting vLLM on GPU $DEVICE port $PORT" | tee "$TRAIN_LOG"
DEVICE="$DEVICE" PORT="$PORT" MODEL="${MODEL:-Qwen/Qwen3.5-4B-Base}" \
  MAX_LEN="${MAX_LEN:-8192}" GPU_UTIL="${GPU_UTIL:-0.25}" \
  setsid nohup ./scripts/serve.sh > "$SERVER_LOG" 2>&1 < /dev/null &
SERVER_PGID=$!

# Engine startup includes a torch.compile pass, so allow several minutes.
for i in $(seq 1 120); do
    if [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/health" 2>/dev/null)" = "200" ]; then
        echo "[$NAME] server ready after ${i}0s" | tee -a "$TRAIN_LOG"
        break
    fi
    sleep 10
done

if [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/health" 2>/dev/null)" != "200" ]; then
    echo "[$NAME] SERVER FAILED TO START, see $SERVER_LOG" | tee -a "$TRAIN_LOG"
    exit 1
fi

echo "[$NAME] training (max ${MAX_HOURS}h)" | tee -a "$TRAIN_LOG"
CUDA_VISIBLE_DEVICES="$DEVICE" uv run python scripts/train.py "$CONFIG" \
    --max-hours "$MAX_HOURS" \
    --override "rollout.server_url=http://127.0.0.1:${PORT}" "output_dir=runs/${NAME}" \
    "${OVERRIDES[@]}" >> "$TRAIN_LOG" 2>&1

echo "[$NAME] finished with code $?" | tee -a "$TRAIN_LOG"

# Free the GPU for anything queued behind this run.
#
# Kill the exact process group we started, NOT `pkill -f "port $PORT"`. The pattern
# form bit three times: when a run failed and exited, its cleanup matched -- and
# killed -- a freshly launched server that happened to reuse the same port, so the
# next attempt died with "connection refused" for no visible reason. setsid makes
# our child a process-group leader, so negating the PID reaches it and its workers.
if [ -n "${SERVER_PGID:-}" ]; then
    kill -- "-${SERVER_PGID}" 2>/dev/null || kill "${SERVER_PGID}" 2>/dev/null || true
fi
