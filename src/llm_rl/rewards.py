"""Verifiable reward for competition math.

The training signal is a single scalar per rollout: did the model box the right
integer. AIME answers are integers in [0, 999] and every DAPO-Math-17k ground truth
is an integer, so exact integer match is the primary check. math-verify is only a
fallback for answers written in a non-canonical form (fractions that reduce to an
integer, `1{,}000`, `10^3`, and so on).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from .config import RewardConfig

_BOXED = "\\boxed"

# Wrappers that carry no numeric meaning and appear constantly in model output.
_STRIP_WRAPPERS = re.compile(
    r"\\(?:text|mathrm|mathbf|textbf|mbox|displaystyle|left|right)\s*(?=\{|\b)"
)
_LATEX_SPACE = re.compile(r"\\[,;:!]|\\quad|\\qquad|~|\\ ")
_INTEGER = re.compile(r"^[+-]?\d+$")


def extract_boxed(text: str) -> str | None:
    r"""Return the contents of the last ``\boxed{...}``, or None.

    Uses brace counting rather than a regex because the payload frequently contains
    nested braces (``\boxed{\frac{1}{2}}``). Also accepts the brace-less
    ``\boxed 5`` form that models sometimes emit.
    """
    start = text.rfind(_BOXED)
    if start == -1:
        return None
    cursor = start + len(_BOXED)
    while cursor < len(text) and text[cursor] == " ":
        cursor += 1
    if cursor >= len(text):
        return None

    if text[cursor] != "{":
        match = re.match(r"[+-]?[\w.]+", text[cursor:])
        return match.group(0) if match else None

    depth = 0
    for index in range(cursor, len(text)):
        char = text[index]
        if char == "{" and (index == cursor or text[index - 1] != "\\"):
            depth += 1
        elif char == "}" and text[index - 1] != "\\":
            depth -= 1
            if depth == 0:
                return text[cursor + 1 : index].strip()
    # Unbalanced: the generation was probably truncated mid-answer.
    return None


def normalize_numeric(answer: str) -> str:
    """Canonicalize a boxed payload so that string equality is meaningful."""
    text = answer.strip()
    text = _STRIP_WRAPPERS.sub("", text)
    text = _LATEX_SPACE.sub("", text)
    text = text.replace("$", "").replace("\\%", "").replace("%", "")
    text = text.replace("{,}", "").replace("\\,", "")
    text = text.strip().strip("{}").strip()
    text = text.rstrip(".")
    # Thousands separators, but only between digits, so 1,000 -> 1000 while a
    # coordinate pair like (1,2) is left alone and simply fails to parse as an int.
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    text = text.lstrip("+")
    if _INTEGER.match(text):
        # Drops leading zeros and unifies -0 with 0.
        return str(int(text))
    return text


def as_integer(answer: str) -> int | None:
    text = normalize_numeric(answer)
    if _INTEGER.match(text):
        return int(text)
    # A float that is exactly integral (42.0) still counts as that integer.
    try:
        value = float(text)
    except ValueError:
        return None
    return int(value) if value.is_integer() else None


@lru_cache(maxsize=1)
def _math_verify():
    try:
        from math_verify import parse, verify

        return parse, verify
    except ImportError:  # pragma: no cover - exercised only without the extra
        return None


def answers_match(prediction: str, gold: str, use_math_verify: bool = True) -> bool:
    """True if `prediction` denotes the same value as `gold`."""
    pred_int, gold_int = as_integer(prediction), as_integer(gold)
    if pred_int is not None and gold_int is not None:
        return pred_int == gold_int
    if normalize_numeric(prediction) == normalize_numeric(gold):
        return True
    if not use_math_verify:
        return False
    tools = _math_verify()
    if tools is None:
        return False
    parse, verify = tools
    try:
        return bool(verify(parse(f"${gold}$"), parse(f"${prediction}$")))
    except Exception:
        # math-verify raises on plenty of malformed model output; that is a miss,
        # not a crash.
        return False


@dataclass
class RewardOutput:
    reward: float
    correct: bool
    has_boxed: bool
    extracted: str | None
    length_penalty: float = 0.0


def overlong_penalty(response_len: int, cfg: RewardConfig) -> float:
    r"""DAPO's soft length punishment, in [-overlong_penalty, 0].

    Zero until the response passes ``overlong_max - overlong_cache``, then falls
    linearly across the cache window, reaching ``-overlong_penalty`` at
    ``overlong_max``. The gradient this creates points toward finishing sooner,
    which is the behaviour a hard truncation mask cannot teach.
    """
    if not cfg.overlong_max or cfg.overlong_cache <= 0:
        return 0.0
    expected = cfg.overlong_max - cfg.overlong_cache
    if response_len <= expected:
        return 0.0
    if response_len >= cfg.overlong_max:
        return -cfg.overlong_penalty
    return (expected - response_len) / cfg.overlong_cache * cfg.overlong_penalty


def score(
    completion: str,
    gold: str,
    cfg: RewardConfig | None = None,
    truncated: bool = False,
    response_len: int | None = None,
) -> RewardOutput:
    """Score one rollout against its ground truth."""
    cfg = cfg or RewardConfig()
    extracted = extract_boxed(completion)
    has_boxed = extracted is not None
    # A truncated generation may contain a \boxed from earlier scratch work that is
    # not the model's final answer, so it never counts as correct.
    correct = bool(has_boxed and not truncated and answers_match(extracted, gold, cfg.use_math_verify))

    reward = cfg.correct if correct else cfg.incorrect
    if cfg.format_bonus and has_boxed and not correct:
        reward += cfg.format_bonus

    penalty = overlong_penalty(response_len, cfg) if response_len is not None else 0.0
    return RewardOutput(
        reward=reward + penalty,
        correct=correct,
        has_boxed=has_boxed,
        extracted=extracted,
        length_penalty=penalty,
    )


def score_batch(
    completions: list[str],
    golds: list[str],
    cfg: RewardConfig | None = None,
    truncated: list[bool] | None = None,
    response_lens: list[int] | None = None,
) -> list[RewardOutput]:
    truncated = truncated or [False] * len(completions)
    lens: list[int | None] = list(response_lens) if response_lens else [None] * len(completions)
    if not len(completions) == len(golds) == len(truncated) == len(lens):
        raise ValueError("completions, golds, truncated and response_lens must be the same length")
    return [score(c, g, cfg, t, n) for c, g, t, n in zip(completions, golds, truncated, lens)]
