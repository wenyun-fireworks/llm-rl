#!/usr/bin/env bash
# Evaluate saved best checkpoints with DAPO's protocol: avg@32, temp 1.0, top_p 0.7.
#
#   ./scripts/eval_best.sh dapo_full dapo_lr2 ...
#
# Each run gets its own GPU and its own engine, evaluated in parallel. Checkpoints
# are bf16 (saved by train.py when held-out eval improves), so the engine can use
# most of the GPU -- no trainer is colocated here.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
export HF_HUB_DISABLE_PROGRESS_BARS=1

K="${K:-32}"
TEMP="${TEMP:-1.0}"
TOPP="${TOPP:-0.7}"
MAXNEW="${MAXNEW:-6144}"
DATASETS="${DATASETS:-HuggingFaceH4/aime_2024 yentinglin/aime_2025}"

names=("$@")
if [ ${#names[@]} -eq 0 ]; then
    echo "usage: $0 RUN_NAME [RUN_NAME ...]"; exit 1
fi

pids=()
for i in "${!names[@]}"; do
    name="${names[$i]}"
    ckpt="runs/${name}/best"
    if [ ! -d "$ckpt" ]; then
        echo "[$name] no best checkpoint at $ckpt, skipping"
        continue
    fi
    device=$((i % 8))
    port=$((8300 + i))
    (
        echo "[$name] serving $ckpt on GPU $device port $port"
        DEVICE="$device" PORT="$port" MODEL="$ckpt" MAX_LEN=7168 GPU_UTIL=0.85 \
            MAX_NUM_SEQS=512 setsid nohup ./scripts/serve.sh \
            > "logs/eval_${name}_server.log" 2>&1 < /dev/null &

        for _ in $(seq 1 90); do
            [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${port}/health")" = "200" ] && break
            sleep 10
        done
        if [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${port}/health")" != "200" ]; then
            echo "[$name] server failed, see logs/eval_${name}_server.log"; exit 1
        fi

        uv run python scripts/eval_baseline.py \
            --url "http://127.0.0.1:${port}" --model "$ckpt" \
            --k "$K" --temperature "$TEMP" --top-p "$TOPP" --max-new-tokens "$MAXNEW" \
            --max-per-dataset 30 --out "runs/${name}/final_eval.json" \
            --datasets $DATASETS > "logs/eval_${name}.log" 2>&1

        echo "[$name] done:"
        grep -E "avg@k" "logs/eval_${name}.log" | sed "s/^/  [$name] /"
        pkill -f "port ${port}" 2>/dev/null || true
    ) &
    pids+=($!)
    sleep 15
done

for p in "${pids[@]}"; do wait "$p"; done
echo
echo "final evals written to runs/<name>/final_eval.json"
