"""Evaluation harness: avg@k and pass@k on math benchmarks.

AIME has only 30 problems per year, so a single greedy sample is far too noisy to
tell whether RL helped: one problem is 3.3 percentage points. We therefore sample k
completions per problem and report avg@k (the mean per-problem accuracy, an unbiased
estimate of single-sample accuracy) alongside pass@k (solved at least once), which
shows whether the ability is present but unreliable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import EvalConfig, RewardConfig
from .data import Example, load_examples
from .rewards import score
from .rollout import Rollout


@dataclass
class EvalResult:
    dataset: str
    num_problems: int
    k: int
    avg_at_k: float  # mean fraction of samples correct == unbiased pass@1
    pass_at_k: float  # fraction of problems solved at least once
    format_rate: float  # fraction of samples containing a parseable \boxed{}
    truncated_rate: float
    mean_response_tokens: float
    per_problem: list[float] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.dataset:32s} n={self.num_problems:4d} k={self.k:2d}  "
            f"avg@k={self.avg_at_k:6.2%}  pass@k={self.pass_at_k:6.2%}  "
            f"boxed={self.format_rate:6.2%}  trunc={self.truncated_rate:6.2%}  "
            f"len={self.mean_response_tokens:6.0f}"
        )


def evaluate(
    rollout: Rollout,
    examples: list[Example],
    cfg: EvalConfig,
    reward_cfg: RewardConfig | None = None,
    dataset_name: str = "",
    batch_size: int = 32,
) -> EvalResult:
    reward_cfg = reward_cfg or RewardConfig()
    per_problem: list[float] = []
    n_boxed = n_trunc = n_samples = 0
    total_tokens = 0

    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        groups = rollout.generate(
            [e.prompt for e in chunk],
            [e.answer for e in chunk],
            group_size=cfg.avg_at_k,
            temperature=cfg.temperature,
            max_new_tokens=cfg.max_new_tokens,
            top_p=cfg.top_p,
        )
        for group in groups:
            correct = 0
            for sample in group.samples:
                outcome = score(sample.text, group.answer, reward_cfg, truncated=sample.truncated)
                correct += outcome.correct
                n_boxed += outcome.has_boxed
                n_trunc += sample.truncated
                total_tokens += len(sample.response_token_ids)
                n_samples += 1
            per_problem.append(correct / len(group.samples))

    return EvalResult(
        dataset=dataset_name,
        num_problems=len(per_problem),
        k=cfg.avg_at_k,
        avg_at_k=sum(per_problem) / max(len(per_problem), 1),
        pass_at_k=sum(p > 0 for p in per_problem) / max(len(per_problem), 1),
        format_rate=n_boxed / max(n_samples, 1),
        truncated_rate=n_trunc / max(n_samples, 1),
        mean_response_tokens=total_tokens / max(n_samples, 1),
        per_problem=per_problem,
    )


# Secondary benchmarks matter here: a 0.8B base model may sit at a hard zero on AIME,
# in which case every rollout group has identical reward and there is no gradient.
# GSM8K and MATH-500 are dense enough to show whether the algorithms work at all.
DEFAULT_EVAL_SPECS: list[tuple[str, str, str | None]] = [
    ("HuggingFaceH4/aime_2024", "train", None),
    ("yentinglin/aime_2025", "train", None),
    ("MathArena/aime_2026", "train", None),
    ("HuggingFaceH4/MATH-500", "test", None),
    ("openai/gsm8k", "test", "main"),
]


def load_eval_suite(
    specs: list[tuple[str, str, str | None]] | None = None,
    max_per_dataset: int | None = None,
) -> dict[str, list[Example]]:
    suite: dict[str, list[Example]] = {}
    for name, split, config in specs or DEFAULT_EVAL_SPECS:
        suite[name] = load_examples(
            name,
            split=split,
            config=config,
            max_samples=max_per_dataset,
            deduplicate=False,
        )
    return suite
