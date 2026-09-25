#!/usr/bin/env bash
# Launch the full overnight comparison: 8 runs, one per GPU.
#
# Layout (each GPU hosts a colocated vLLM engine + trainer):
#
#   GPU 0  reinforce  seed 0     RLOO baseline, no ratio, no critic
#   GPU 1  grpo       seed 0     group baseline + clipped ratio (Dr.GRPO/DAPO flavour)
#   GPU 2  ppo        seed 0     learned critic + token-level GAE
#   GPU 3  reinforce  seed 1     seed repeat, to separate signal from noise
#   GPU 4  grpo       seed 1
#   GPU 5  ppo        seed 1
#   GPU 6  grpo_orig            ablation: GRPO exactly as published (std norm + per-sequence loss)
#   GPU 7  nobaseline           ablation: raw reward, no baseline at all
#
# The two ablations are the teaching payload: 6 vs 1 isolates the loss-normalization
# choice, and 7 vs 0 shows what a baseline is actually worth.
set -uo pipefail
cd "$(dirname "$0")/.."

MAX_HOURS="${MAX_HOURS:-6.5}"
STEPS="${STEPS:-400}"
PROMPTS="${PROMPTS:-16}"
GROUP="${GROUP:-8}"
NEW_TOKENS="${NEW_TOKENS:-4096}"
MINIBATCH="${MINIBATCH:-8}"
EVAL_EVERY="${EVAL_EVERY:-25}"

COMMON=(
    "total_steps=${STEPS}"
    "rollout.prompts_per_step=${PROMPTS}"
    "rollout.group_size=${GROUP}"
    "rollout.max_new_tokens=${NEW_TOKENS}"
    "algo.minibatch_size=${MINIBATCH}"
    "eval.every_steps=${EVAL_EVERY}"
    "eval.avg_at_k=4"
    "eval.max_new_tokens=${NEW_TOKENS}"
    "save_every_steps=100000"   # checkpoints are 17GB in fp32; skip for the tutorial
)

launch () {
    local name="$1" config="$2" device="$3" port="$4"; shift 4
    setsid nohup ./scripts/launch_run.sh "$name" "$config" "$device" "$port" "$MAX_HOURS" \
        "${COMMON[@]}" "$@" > "logs/${name}_launch.log" 2>&1 < /dev/null &
    echo "launched $name on GPU $device (port $port)"
}

mkdir -p logs runs

launch reinforce_s0 configs/reinforce.yaml 0 8100 "seed=0"
launch grpo_s0      configs/grpo.yaml      1 8101 "seed=0"
launch ppo_s0       configs/ppo.yaml       2 8102 "seed=0"
launch reinforce_s1 configs/reinforce.yaml 3 8103 "seed=1"
launch grpo_s1      configs/grpo.yaml      4 8104 "seed=1"
launch ppo_s1       configs/ppo.yaml       5 8105 "seed=1"

# Ablation: GRPO as originally published -- divide by group std, average per
# sequence. Compare against grpo_s0 to see how much the Dr.GRPO/DAPO corrections
# actually matter.
launch grpo_orig configs/grpo.yaml 6 8106 \
    "seed=0" "algo.normalize_advantage_std=true" "algo.loss_normalization=sequence"

# Ablation: no baseline at all. Same loop, advantage = raw reward. Expect higher
# gradient variance and worse/noisier learning than reinforce_s0.
launch nobaseline configs/reinforce.yaml 7 8107 \
    "seed=0" "algo.advantage=raw"

echo
echo "all 8 runs launched; metrics stream to runs/<name>/metrics.jsonl"
echo "watch with: python scripts/report.py --watch"
