import pytest

from llm_rl.config import RewardConfig
from llm_rl.rewards import answers_match, as_integer, extract_boxed, score, score_batch


@pytest.mark.parametrize(
    "text,expected",
    [
        (r"so the answer is \boxed{42}.", "42"),
        (r"\boxed{\frac{1}{2}}", r"\frac{1}{2}"),
        (r"\boxed{\text{answer is } 7}", r"\text{answer is } 7"),
        # Nested braces must be matched, not regex-ed.
        (r"\boxed{\frac{\sqrt{3}}{2}}", r"\frac{\sqrt{3}}{2}"),
        # Only the last box counts: earlier ones are scratch work.
        (r"first \boxed{1} then actually \boxed{2}", "2"),
        (r"\boxed 5", "5"),
        (r"\boxed  {  99  }", "99"),
        ("no box here", None),
        # Truncated mid-answer.
        (r"the answer is \boxed{12", None),
    ],
)
def test_extract_boxed(text, expected):
    assert extract_boxed(text) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("42", 42),
        ("  042 ", 42),
        ("+7", 7),
        ("-13", -13),
        ("1,000", 1000),
        (r"1{,}000", 1000),
        (r"\text{42}", 42),
        ("42.0", 42),
        ("42.5", None),
        (r"\frac{1}{2}", None),
        ("(1,2)", None),
    ],
)
def test_as_integer(raw, expected):
    assert as_integer(raw) == expected


@pytest.mark.parametrize(
    "pred,gold,expected",
    [
        ("42", "42", True),
        ("042", "42", True),
        (" 42 ", "42", True),
        (r"\text{42}", "42", True),
        ("1,000", "1000", True),
        ("43", "42", False),
        ("", "42", False),
        # math-verify fallback: reduces to the gold integer.
        (r"\frac{84}{2}", "42", True),
    ],
)
def test_answers_match(pred, gold, expected):
    assert answers_match(pred, gold) is expected


def test_score_correct_and_incorrect():
    cfg = RewardConfig(correct=1.0, incorrect=0.0)
    hit = score(r"... so \boxed{42}", "42", cfg)
    assert (hit.reward, hit.correct, hit.has_boxed) == (1.0, True, True)

    miss = score(r"... so \boxed{41}", "42", cfg)
    assert (miss.reward, miss.correct, miss.has_boxed) == (0.0, False, True)

    nothing = score("I give up", "42", cfg)
    assert (nothing.reward, nothing.correct, nothing.has_boxed) == (0.0, False, False)


def test_format_bonus_only_applies_to_wrong_but_formatted():
    cfg = RewardConfig(correct=1.0, incorrect=0.0, format_bonus=0.1)
    assert score(r"\boxed{41}", "42", cfg).reward == pytest.approx(0.1)
    assert score(r"\boxed{42}", "42", cfg).reward == pytest.approx(1.0)
    assert score("no box", "42", cfg).reward == pytest.approx(0.0)


def test_truncated_completion_is_never_correct():
    cfg = RewardConfig()
    # A box from earlier scratch work must not be rewarded when the rollout was cut off.
    out = score(r"maybe \boxed{42} but let me check", "42", cfg, truncated=True)
    assert out.correct is False
    assert out.reward == cfg.incorrect


def test_score_batch_length_mismatch():
    with pytest.raises(ValueError):
        score_batch([r"\boxed{1}"], ["1", "2"])


def test_score_batch():
    outs = score_batch([r"\boxed{1}", r"\boxed{9}"], ["1", "2"])
    assert [o.correct for o in outs] == [True, False]
