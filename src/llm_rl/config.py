"""Config dataclasses shared by every algorithm.

Anything that materially changes results is a field here rather than a constant in
the code, so that REINFORCE / GRPO / PPO runs can be compared under a matched budget.
"""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen3.5-4B-Base"
    # Master weights are fp32 and this is NOT a performance oversight. AdamW keeps
    # its state in the parameter dtype, and bf16 has an 8-bit mantissa, so near a
    # typical weight of ~0.02 the smallest representable change is ~1.6e-4. An
    # lr=1e-6 Adam step moves a weight by ~1e-6 and is rounded away: measured, only
    # 2% of bf16 weights move at all and the mean update is 36x too small.
    # Compute still runs in bf16 via autocast, so the speed cost is small.
    dtype: str = "float32"
    autocast_dtype: str | None = "bfloat16"
    attn_implementation: str | None = None
    gradient_checkpointing: bool = True
    # Qwen3.5 base ships a vision tower and an MTP head. We train text-only, so both
    # are frozen and excluded from the optimizer; they still live in the state dict
    # because the rollout engine loads the same class.
    freeze_vision: bool = True
    freeze_mtp: bool = True
    value_head: bool = False
    # Critic trains on detached features; see Policy.detach_value_head.
    detach_value_head: bool = True
    # Chunk size for the log-softmax over a 248k vocab. Full logits for one 4k
    # sequence are ~2GB in bf16, so this is a correctness-critical memory bound.
    logprob_chunk_tokens: int = 512
    # Pad every batch to this width instead of to the longest sequence present.
    # Constant shapes stop allocator fragmentation, which at 20k context cost ~50
    # GiB of reserved-but-unused memory and OOMed a run whose allocated peak was
    # only 90 GiB. 0 disables (pad to the batch max).
    pad_to_width: int = 0


@dataclass
class DataConfig:
    train_dataset: str = "BytedTsinghua-SIA/DAPO-Math-17k"
    train_split: str = "train"
    eval_datasets: list[str] = field(
        default_factory=lambda: [
            "HuggingFaceH4/aime_2024",
            "yentinglin/aime_2025",
            "MathArena/aime_2026",
        ]
    )
    max_prompt_tokens: int = 1024
    max_train_samples: int | None = None
    shuffle_seed: int = 0
    # DAPO-Math-17k ships pre-expanded (1.79M rows, ~17.4k distinct problems).
    deduplicate: bool = True
    exclude_cjk: bool = False


@dataclass
class RolloutConfig:
    group_size: int = 8
    prompts_per_step: int = 32
    max_new_tokens: int = 8192
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    seed: int | None = None
    gpu_memory_utilization: float = 0.35
    enforce_eager: bool = False
    # When set, talk to `vllm serve` over HTTP instead of an in-process engine.
    server_url: str | None = None


@dataclass
class RewardConfig:
    correct: float = 1.0
    incorrect: float = 0.0
    # Small bonus for emitting a parseable \boxed{} at all, even if wrong. Helps a
    # base model that has not learned the output format yet.
    format_bonus: float = 0.0
    use_math_verify: bool = True

    # DAPO overlong reward shaping. Responses longer than
    # (overlong_max - overlong_cache) accrue a linear penalty reaching
    # -overlong_penalty at overlong_max. This is the soft alternative to simply
    # masking truncated rollouts out of the loss: masking throws away the evidence,
    # whereas a length penalty actively teaches the model to finish. Measured at 4B,
    # masking alone let truncation climb to 100% until the gradient died.
    # 0 disables shaping. DAPO uses max=20480, cache=4096.
    overlong_max: int = 0
    overlong_cache: int = 4096
    overlong_penalty: float = 1.0


@dataclass
class AlgoConfig:
    name: str = "grpo"  # reinforce | grpo | ppo
    advantage: str = "grpo"  # rloo | grpo | gae

    # GRPO original divides by the group std; Dr.GRPO / DAPO argue this biases
    # toward low-variance groups and drop it.
    normalize_advantage_std: bool = True
    # "sequence": mean over tokens within a sequence, then mean over sequences
    #   (GRPO as published; length-biased).
    # "token": sum over all tokens / total token count in the batch (Dr.GRPO, DAPO).
    loss_normalization: str = "token"

    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2  # DAPO "clip-higher" sets this above the low side
    # Dual-clip: cap the surrogate at |A| * clip_ratio_c for negative advantages,
    # which is otherwise unbounded above. None disables. verl's DAPO uses 10.0.
    clip_ratio_c: float | None = None
    ppo_epochs: int = 1
    # Sequences per optimizer step. This is an *algorithm* knob: it sets how far the
    # policy drifts from pi_old within one rollout batch, which is what the clipped
    # ratio is defined against.
    minibatch_size: int = 8
    # Sequences per forward/backward. This is a *memory* knob. When it is smaller
    # than minibatch_size, gradients accumulate across micro-batches and one
    # optimizer step covers the whole minibatch. Needed for GSPO, whose clip range
    # is ~1e-4: with 32 sequential updates per batch almost every sequence would be
    # clipped (hence zero-gradient) after the first few. 0 means "same as
    # minibatch_size" (no accumulation).
    micro_batch_size: int = 0

    kl_coef: float = 0.0
    kl_estimator: str = "k3"
    entropy_coef: float = 0.0

    value_coef: float = 0.5
    gamma: float = 1.0
    gae_lambda: float = 1.0
    # Normalize GAE advantages to zero mean / unit variance across the batch.
    # Standard for PPO and effectively mandatory here: the scale of R - V is set
    # by the critic's error, so without whitening the policy step size drifts with
    # critic quality instead of staying fixed. The group estimators do not need
    # this -- centering on the group mean already fixes their scale.
    whiten_advantages: bool = False

    # Sequences cut off by max_new_tokens have no terminal answer, so their reward
    # is uninformative; DAPO masks them out of the loss.
    mask_truncated: bool = True
    # Drop groups where every sample got the same reward: their advantage is
    # identically zero and they only dilute the gradient.
    dynamic_sampling: bool = False
    # DAPO's full version of the above: rather than just dropping degenerate groups
    # and training on a smaller batch, keep generating until enough informative
    # groups have been collected. `gen_batch_prompts` is how many prompts to sample
    # per attempt (DAPO uses 3x the target), and `max_gen_batches` bounds the work
    # when almost every group is degenerate. 0 keeps the cheap drop-only behaviour.
    gen_batch_prompts: int = 0
    max_gen_batches: int = 10


@dataclass
class OptimConfig:
    # "adamw" or "adamw8bit". 8-bit quantizes only the optimizer *moments*, not the
    # parameters, cutting Adam state from 8 to 2 bytes per parameter. That is what
    # lets a 9B model train beside a colocated vLLM engine on one 183GB GPU.
    # Parameters stay fp32 regardless -- see ModelConfig.dtype for why that matters.
    optimizer: str = "adamw"
    lr: float = 1e-6
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    warmup_steps: int = 0
    scheduler: str = "constant"  # constant | cosine


@dataclass
class TeacherConfig:
    """On-policy distillation teacher, served separately by vLLM.

    The teacher MUST share the student's tokenizer, since the whole method rests on
    per-token alignment. All Qwen3.5 checkpoints share a 248,320-token vocabulary,
    so any of them can teach any other.
    """

    url: str = "http://127.0.0.1:8270"
    model: str = "Qwen/Qwen3.5-122B-A10B"
    batch_size: int = 16
    timeout: float = 600.0
    # Bound |log pi_teacher - log pi_old| per token. A token the teacher finds
    # nearly impossible (logprob -20, common on formatting noise) would otherwise
    # dominate the batch gradient on its own.
    advantage_clip: float | None = 10.0
    # Mix in the verifiable correctness reward alongside the distillation signal.
    # 0.0 is pure distillation; the reward is still logged either way.
    reward_mix: float = 0.0
    # "score" = single-token score-function estimator (cheap, but drops the entropy
    # term of the reverse KL, so nothing prevents mode collapse).
    # "topk"  = differentiable KL over the teacher's top-k support (keeps it).
    objective: str = "score"
    topk: int = 20


@dataclass
class EvalConfig:
    every_steps: int = 50
    avg_at_k: int = 16
    temperature: float = 0.6
    top_p: float = 0.95
    max_new_tokens: int = 8192


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    algo: AlgoConfig = field(default_factory=AlgoConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)

    total_steps: int = 500
    seed: int = 0
    output_dir: str = "runs/default"
    save_every_steps: int = 100
    log_every_steps: int = 1

    @classmethod
    def from_yaml(cls, *paths: str | Path) -> Config:
        """Load and deep-merge YAML files left to right; later files win.

        A `defaults:` key naming another YAML (relative to the same directory) is
        loaded first, which is how configs/{reinforce,grpo,ppo}.yaml extend base.yaml.
        """
        merged: dict[str, Any] = {}
        for path in paths:
            path = Path(path)
            raw = yaml.safe_load(path.read_text()) or {}
            for parent in _pop_defaults(raw):
                parent_raw = yaml.safe_load((path.parent / parent).read_text()) or {}
                _pop_defaults(parent_raw)
                merged = _deep_merge(merged, parent_raw)
            merged = _deep_merge(merged, raw)
        return cls.from_dict(merged)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        return _build(cls, data)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def apply_overrides(self, overrides: list[str]) -> Config:
        """Apply CLI overrides of the form `algo.clip_ratio_high=0.28`."""
        patch: dict[str, Any] = {}
        for item in overrides:
            key, _, value = item.partition("=")
            if not _:
                raise ValueError(f"override must be key=value, got {item!r}")
            cursor = patch
            *parents, leaf = key.split(".")
            for part in parents:
                cursor = cursor.setdefault(part, {})
            cursor[leaf] = yaml.safe_load(value)
        return Config.from_dict(_deep_merge(self.to_dict(), patch))


def _pop_defaults(raw: dict[str, Any]) -> list[str]:
    value = raw.pop("defaults", [])
    return [value] if isinstance(value, str) else list(value)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _build(cls: type, data: dict[str, Any]) -> Any:
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(fields)
    if unknown:
        raise ValueError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")
    # `from __future__ import annotations` makes every field.type a string, so the
    # nested dataclasses have to be resolved through get_type_hints or sub-configs
    # silently stay plain dicts.
    hints = typing.get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        ftype = hints.get(name, fields[name].type)
        if dataclasses.is_dataclass(ftype) and isinstance(value, dict):
            kwargs[name] = _build(ftype, value)
        elif name == "betas" and isinstance(value, list):
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = _coerce(value, ftype)
    return cls(**kwargs)


def _coerce(value: Any, ftype: Any) -> Any:
    """Coerce a scalar to its declared field type.

    YAML 1.1 does not recognize `2e-6` as a float (it wants `2.0e-6`), so a perfectly
    reasonable `--override optim.lr=2e-6` would otherwise arrive as a string and
    reach the optimizer intact.
    """
    # Only scalars are coerced. Containers carry their own element types
    # (tuple[float, float] would otherwise look like a plain float target).
    if value is None or isinstance(value, (bool, list, tuple, dict)):
        return value
    options = typing.get_args(ftype) or (ftype,)
    target = next((t for t in options if t is not type(None)), None)
    try:
        if target is float and not isinstance(value, float):
            return float(value)
        if target is int and not isinstance(value, int):
            return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"cannot interpret {value!r} as {getattr(target, '__name__', target)}")
    return value
