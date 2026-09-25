#!/usr/bin/env bash
# GSPO variants on Qwen3.5-9B-Base, GPUs 0-3.
#
# configs/gspo.yaml inherits dapo.yaml, so model / data / reward shaping / rollout
# budget are identical to the DAPO runs that reached 55.0% AIME 2024 avg@32. The
# only difference is the policy objective: sequence-level importance ratio and
# sequence-level clipping instead of token-level.
#
#   GPU 0  gspo_std      the paper's config: eps 3e-4/4e-4, minibatch 32, micro 4
#   GPU 1  gspo_eps10x   10x looser clip; measured clip_frac was only 12.5% at 3e-4
#   GPU 2  gspo_mb8      minibatch 8 -> 16 updates/batch, DAPO's update granularity
#   GPU 3  gspo_cosine   + cosine LR decay, our best fix for peak-then-decay
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs runs

MAX_HOURS="${MAX_HOURS:-6.0}"
export MAX_LEN="${MAX_LEN:-7168}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
export MODEL="${MODEL:-Qwen/Qwen3.5-9B-Base}"

COMMON=(
    "total_steps=1000"
    "eval.every_steps=20"
    "eval.avg_at_k=8"
    "save_every_steps=1000000"
)

MANIFEST=logs/runs.manifest
: > "$MANIFEST"

launch () {
    local name="$1" config="$2" device="$3" port="$4"; shift 4
    echo "$name $config $device $port $*" >> "$MANIFEST"
    setsid nohup ./scripts/launch_run.sh "$name" "$config" "$device" "$port" "$MAX_HOURS" \
        "${COMMON[@]}" "$@" > "logs/${name}_launch.log" 2>&1 < /dev/null &
    echo "launched $name on GPU $device (port $port)"
    sleep 20
}

launch gspo_std    configs/gspo.yaml 0 8260 "seed=0"
launch gspo_eps10x configs/gspo.yaml 1 8261 "seed=0" \
    "algo.clip_ratio_low=3.0e-3" "algo.clip_ratio_high=4.0e-3"
launch gspo_mb8    configs/gspo.yaml 2 8262 "seed=0" "algo.minibatch_size=8"
launch gspo_cosine configs/gspo.yaml 3 8263 "seed=0" "optim.scheduler=cosine" "total_steps=100"

echo
echo "4 GSPO runs launched on GPUs 0-3. GPUs 4-7 left free for 35B-A3B."
