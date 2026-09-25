#!/usr/bin/env bash
# Launch a vLLM rollout server configured for IPC weight sync from the trainer.
#
# FlashInfer is bypassed deliberately: its trtllm-gen decode kernel JIT-compiles
# against CUDA driver symbols (CU_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE and friends)
# that do not exist in CUDA 13.3, which is what this host has.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.5-0.8B-Base}"
PORT="${PORT:-8001}"
DEVICE="${DEVICE:-1}"
GPU_UTIL="${GPU_UTIL:-0.35}"
MAX_LEN="${MAX_LEN:-4096}"
BACKEND="${BACKEND:-FLASH_ATTN}"
# Qwen3.5's Gated DeltaNet layers need one recurrent-state cache block per decode
# sequence. vLLM defaults max_num_seqs to 1024, which exceeds the blocks available
# at low gpu_memory_utilization and fails CUDA graph capture at startup. We never
# need more than gen_batch_prompts x group_size concurrent sequences anyway.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-512}"
# Tensor parallel. Needed for models too large for one GPU (Qwen3.5-122B-A10B is
# 250GB, so TP=4 puts ~63GB on each of four devices). DEVICE then takes a
# comma-separated list, e.g. DEVICE=4,5,6,7 TP=4.
TP="${TP:-1}"
# Caps how many tokens enter one forward pass. Critical when serving a teacher for
# `prompt_logprobs`: vLLM computes log_softmax over the full 248,320-token vocab in
# one shot, so a 6,300-token prompt needs 6.3 GB of fp32 logits and a few concurrent
# requests OOM a 178 GiB GPU even with the weights already resident. 2048 tokens
# bounds that buffer to ~2 GB. Empty leaves vLLM's default.
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-}"
EXTRA_ARGS=()
[ -n "$MAX_BATCHED_TOKENS" ] && EXTRA_ARGS+=(--max-num-batched-tokens "$MAX_BATCHED_TOKENS")

export VLLM_ALLOW_INSECURE_SERIALIZATION=1
# The weight-transfer control plane (/init_weight_transfer_engine, /update_weights,
# ...) lives on vLLM's dev router, which is only mounted in dev mode.
export VLLM_SERVER_DEV_MODE=1
export CUDA_VISIBLE_DEVICES="$DEVICE"

exec uv run vllm serve "$MODEL" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len "$MAX_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --tensor-parallel-size "$TP" \
    --attention-backend "$BACKEND" \
    "${EXTRA_ARGS[@]}" \
    --weight-transfer-config '{"backend": "ipc"}'
