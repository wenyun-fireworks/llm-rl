#!/usr/bin/env bash
# Eight parallel DAPO experiments on Qwen3.5-9B-Base, one per GPU.
#
# Each GPU hosts a colocated vLLM engine + trainer (CUDA IPC weight transfer
# resolves handles by GPU UUID, so they must share a device). Measured ~215 s/step,
# so a 6h budget gives roughly 100 steps -- which is where every 4B run peaked.
#
#   GPU 0  dapo_full     the recipe as configured; the reference
#   GPU 1  dapo_kl       + kl_coef 0.001, against the peak-then-decay we measured
#   GPU 2  dapo_dyn      + dynamic-sampling resampling (3x generation cost)
#   GPU 3  dapo_g16      group 16 x 8 prompts, same rollout count, better baselines
#   GPU 4  dapo_lr2      lr 2e-6; the paper's lr may be too slow for ~100 steps
#   GPU 5  dapo_entropy  small entropy bonus to delay collapse
#   GPU 6  dapo_noshape  overlong shaping off, mask instead: isolates its effect
#   GPU 7  grpo_9b       our previous GRPO recipe at 9B, as the control
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs runs

MAX_HOURS="${MAX_HOURS:-6.0}"
export MAX_LEN="${MAX_LEN:-7168}"      # 6144 generation + prompt headroom
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
export MODEL="${MODEL:-Qwen/Qwen3.5-9B-Base}"

COMMON=(
    "total_steps=1000"          # the wall clock, not this, is the real limit
    "eval.every_steps=20"
    "eval.avg_at_k=8"
    "save_every_steps=1000000"  # only the best-eval checkpoint is kept
)

MANIFEST=logs/runs.manifest
: > "$MANIFEST"

launch () {
    local name="$1" config="$2" device="$3" port="$4"; shift 4
    # Overrides go in the manifest too, or a watchdog restart would silently drop
    # the very thing that makes each run a distinct experiment.
    echo "$name $config $device $port $*" >> "$MANIFEST"
    setsid nohup ./scripts/launch_run.sh "$name" "$config" "$device" "$port" "$MAX_HOURS" \
        "${COMMON[@]}" "$@" > "logs/${name}_launch.log" 2>&1 < /dev/null &
    echo "launched $name on GPU $device (port $port)"
    sleep 20   # stagger: eight simultaneous engine inits thrash the page cache
}

launch dapo_full    configs/dapo.yaml 0 8200 "seed=0"
launch dapo_kl      configs/dapo.yaml 1 8201 "seed=0" "algo.kl_coef=0.001"
launch dapo_dyn     configs/dapo.yaml 2 8202 "seed=0" "algo.gen_batch_prompts=32"
launch dapo_g16     configs/dapo.yaml 3 8203 "seed=0" \
    "rollout.group_size=16" "rollout.prompts_per_step=8"
launch dapo_lr2     configs/dapo.yaml 4 8204 "seed=0" "optim.lr=2.0e-6"
launch dapo_entropy configs/dapo.yaml 5 8205 "seed=0" "algo.entropy_coef=0.003"

# Ablation: turn overlong shaping off and go back to masking truncated rollouts,
# which is what the 4B runs did. Isolates what the soft length penalty buys.
launch dapo_noshape configs/dapo.yaml 6 8206 "seed=0" \
    "reward.overlong_max=0" "algo.mask_truncated=true"

# Control: the GRPO recipe from the previous experiment, at 9B. No dual-clip, no
# overlong shaping, weight decay 0, so the DAPO-specific pieces are all absent.
launch grpo_9b      configs/dapo.yaml 7 8207 "seed=0" \
    "algo.clip_ratio_c=null" "reward.overlong_max=0" "algo.mask_truncated=true" \
    "optim.weight_decay=0.0" "optim.warmup_steps=10"

echo
echo "8 runs launched. metrics -> runs/<name>/metrics.jsonl, best weights -> runs/<name>/best"
echo "watch: python scripts/report.py --watch"
