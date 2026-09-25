"""Dataset loading and prompt construction.

Qwen3.5-0.8B-Base has no chat template, so we use an R1-Zero style plain-text
prompt and rely on the reward to teach the format.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterator

PROMPT_TEMPLATE = """A conversation between User and Assistant. The User asks a math question, \
and the Assistant solves it. The Assistant first reasons step by step, then gives the final answer.
User: {problem}
Please reason step by step, and put your final answer within \\boxed{{}}.
Assistant:"""

# Instruction boilerplate that upstream datasets bake into the problem text. DAPO-Math-17k
# wraps every problem in an "Answer: $Answer" format instruction, which contradicts the
# \boxed{} format our reward grades, so both ends have to come off.
_CJK = re.compile(r"[\u4e00-\u9fff]")
_LEADING_INSTRUCTIONS = re.compile(
    r"^\s*Solve the following math problem step by step\..*?(?:\n\n|\. )",
    re.IGNORECASE | re.DOTALL,
)
_TRAILING_INSTRUCTIONS = re.compile(
    r"\s*(?:Please reason step by step.*|Remember to put your answer.*|Let's think step by step.*"
    r"|Present the answer in LaTex format.*)\s*$",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class Example:
    problem: str
    answer: str
    source: str = ""

    @property
    def prompt(self) -> str:
        return PROMPT_TEMPLATE.format(problem=self.problem)


def _first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _as_text(value: Any) -> str:
    """Flatten a problem field that may be a string or a chat-style message list."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                # Ignore the system turn; only the user's problem statement matters.
                if item.get("role") == "system":
                    continue
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    if isinstance(value, dict):
        return str(value.get("content", value))
    return str(value)


def parse_row(row: dict[str, Any], source: str = "") -> Example | None:
    """Map a dataset row onto an Example, tolerating the schema differences
    between DAPO-Math-17k, the AIME sets, MATH-500 and GSM8K."""
    problem = _as_text(_first_present(row, ("problem", "question", "prompt", "Problem")) or "")
    problem = _LEADING_INSTRUCTIONS.sub("", problem)
    problem = _TRAILING_INSTRUCTIONS.sub("", problem).strip()

    answer = _first_present(row, ("answer", "solution", "Answer", "final_answer"))
    reward_model = row.get("reward_model")
    if isinstance(reward_model, dict):
        answer = reward_model.get("ground_truth", answer)
    extra = row.get("extra_info")
    if answer is None and isinstance(extra, dict):
        answer = extra.get("answer")

    if isinstance(answer, (list, tuple)):
        answer = answer[0] if answer else None
    if not problem or answer is None:
        return None
    answer = str(answer).strip()

    # GSM8K puts the final answer after a #### marker at the end of a worked solution.
    if "####" in answer:
        answer = answer.split("####")[-1].strip()
    if not answer:
        return None
    return Example(problem=problem, answer=answer, source=source)


def load_examples(
    name: str,
    split: str = "train",
    max_samples: int | None = None,
    shuffle_seed: int | None = None,
    deduplicate: bool = True,
    config: str | None = None,
    exclude_cjk: bool = False,
) -> list[Example]:
    """Load a dataset and map it to Examples.

    Deduplication is on by default and matters: BytedTsinghua-SIA/DAPO-Math-17k is
    published pre-expanded for rollout workers, so it has 1.79M rows covering only
    ~17.4k distinct problems. Without dedup a "step" would repeatedly draw the same
    prompt and an "epoch" would mean nothing.
    """
    from datasets import load_dataset

    dataset = load_dataset(name, config, split=split) if config else load_dataset(name, split=split)
    if shuffle_seed is not None:
        dataset = dataset.shuffle(seed=shuffle_seed)

    examples: list[Example] = []
    seen: set[str] = set()
    for row in dataset:
        example = parse_row(row, source=name)
        if example is None:
            continue
        # ~20% of DAPO-Math-17k is Chinese. Kept by default (Qwen is bilingual and it
        # is free extra data), but exposed because it changes what the policy drifts
        # toward and AIME is English-only.
        if exclude_cjk and _CJK.search(example.problem):
            continue
        if deduplicate:
            if example.problem in seen:
                continue
            seen.add(example.problem)
        examples.append(example)
        if max_samples is not None and len(examples) >= max_samples:
            break
    if not examples:
        raise ValueError(f"no usable examples parsed from {name}:{split}")
    return examples


def infinite_batches(
    examples: list[Example], batch_size: int, seed: int = 0
) -> Iterator[list[Example]]:
    """Yield shuffled batches forever, reshuffling each time the data is exhausted."""
    import random

    rng = random.Random(seed)
    order: list[int] = []
    while True:
        batch = []
        while len(batch) < batch_size:
            if not order:
                order = list(range(len(examples)))
                rng.shuffle(order)
            batch.append(examples[order.pop()])
        yield batch
