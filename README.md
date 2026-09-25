# llm-rl

REINFORCE, GRPO and PPO implemented from scratch over one shared rollout stack, and
trained on competition math with a verifiable reward.

No TRL, no verl, no external RL trainer. The advantage estimators, losses, batching,
weight sync and training loop are all here.

**Start with [TUTORIAL.md](TUTORIAL.md)** -- it explains the algorithms, the
tradeoffs between them, and the five subtle failure modes that this code exists to
avoid. **Measured results** from the builds and runs are in
[FINDINGS.md](FINDINGS.md).

## What it does

Samples a group of solutions per problem from a vLLM engine, checks each final
answer against ground truth, converts those rewards into per-token advantages, and
takes a policy-gradient step. Updated weights are pushed straight back into the
inference engine over CUDA IPC, so no checkpoint is ever written between steps.

- **Model**: Qwen3.5-4B-Base (hybrid Gated DeltaNet + full attention, 248k vocab)
- **Train**: DAPO-Math-17k
- **Eval**: AIME 2024 / 2025 / 2026, MATH-500, GSM8K, reported as avg@k and pass@k
- **Reward**: extract `\boxed{...}`, integer match, `math-verify` as fallback

## Setup

```bash
uv sync --extra fla --group dev
pytest tests/          # 96 tests, CPU only
```

Requires a CUDA GPU with enough memory to host the trainer and an inference engine
side by side (~150 GB at the default batch size; tuned for B200).

## Running

The engine and trainer must share a physical GPU, because CUDA IPC weight transfer
resolves handles by GPU UUID.

```bash
# inference engine on GPU 0
DEVICE=0 PORT=8100 ./scripts/serve.sh

# trainer on the same GPU
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train.py configs/grpo.yaml \
    --override rollout.server_url=http://127.0.0.1:8100
```

Full comparison across 8 GPUs (3 algorithms x 2 seeds + 2 ablations):

```bash
MAX_HOURS=6.5 ./scripts/run_all.sh
python scripts/report.py --plot
```

## Configs

`configs/base.yaml` holds everything shared; the per-algorithm files change only
their `algo` block, so runs are directly comparable. Any field can be overridden:

```bash
--override algo.clip_ratio_high=0.28 rollout.group_size=16 optim.lr=2e-6
```

| config | advantage | loss | critic |
|---|---|---|---|
| `reinforce.yaml` | leave-one-out baseline | plain policy gradient | no |
| `grpo.yaml` | group mean | clipped ratio | no |
| `ppo.yaml` | GAE(lambda) | clipped ratio + value loss | yes |

## Diagnostic scripts

Small standalone checks, each verifying one property that is otherwise invisible:

| script | question it answers |
|---|---|
| `_check_parity.py` | do vLLM and HF agree on tokens and logprobs? |
| `_check_weight_sync.py` | does IPC sync actually change the engine's weights, reversibly? |
| `_check_update_precision.py` | does an lr=1e-6 update survive the parameter dtype? |
| `_check_learning.py` | do real training steps move real weights? |
| `_bench_logprobs.py` | what does chunked log-softmax save? |

## Repo layout

See [TUTORIAL.md section 7](TUTORIAL.md#7-layout). Short version: the algorithm
differences live entirely in `advantages.py` and `losses.py`; everything else is
shared.
