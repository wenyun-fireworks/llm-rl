"""Tests for the four DAPO techniques.

Clip-higher and token-level normalization are covered in test_losses.py; this file
covers overlong reward shaping, dual-clip, and dynamic sampling with resampling.
"""

import pytest
import torch

from llm_rl.config import Config, RewardConfig
from llm_rl.losses import ppo_clipped_loss
from llm_rl.rewards import overlong_penalty, score, score_batch

SHAPE = RewardConfig(overlong_max=8192, overlong_cache=2048, overlong_penalty=1.0)
# expected length = 8192 - 2048 = 6144


@pytest.mark.parametrize(
    "length,expected",
    [
        (100, 0.0),
        (6144, 0.0),          # exactly at the threshold, still free
        (7168, -0.5),         # halfway through the cache window
        (8192, -1.0),         # at the cap
        (9999, -1.0),         # clamped past the cap
    ],
)
def test_overlong_penalty_is_linear_across_the_cache_window(length, expected):
    assert overlong_penalty(length, SHAPE) == pytest.approx(expected)


def test_overlong_penalty_disabled_by_default():
    assert overlong_penalty(999_999, RewardConfig()) == 0.0


def test_overlong_penalty_is_monotone():
    values = [overlong_penalty(n, SHAPE) for n in range(0, 9000, 250)]
    assert all(b <= a + 1e-9 for a, b in zip(values, values[1:]))


def test_shaping_reduces_reward_of_a_correct_but_overlong_answer():
    short = score(r"\boxed{42}", "42", SHAPE, response_len=1000)
    long = score(r"\boxed{42}", "42", SHAPE, response_len=7168)
    assert short.reward == pytest.approx(1.0)
    assert long.reward == pytest.approx(0.5)
    # Correctness is a property of the answer, not its length.
    assert short.correct and long.correct


def test_shaping_penalizes_a_wrong_overlong_answer_below_zero():
    out = score("no answer here", "42", SHAPE, response_len=8192)
    assert out.reward == pytest.approx(-1.0)
    assert out.length_penalty == pytest.approx(-1.0)


def test_score_batch_threads_response_lengths():
    outs = score_batch(
        [r"\boxed{1}", r"\boxed{1}"],
        ["1", "1"],
        SHAPE,
        response_lens=[100, 8192],
    )
    assert outs[0].reward == pytest.approx(1.0)
    assert outs[1].reward == pytest.approx(0.0)


def test_score_batch_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        score_batch([r"\boxed{1}"], ["1"], SHAPE, response_lens=[1, 2])


# ---- dual-clip ----


def test_dual_clip_caps_the_loss_for_negative_advantages():
    # log_ratio = 5 -> ratio ~148, far outside the trust region, A < 0.
    new = torch.full((1, 1), 5.0)
    old = torch.zeros(1, 1)
    adv = torch.tensor([[-1.0]])
    mask = torch.ones(1, 1)

    plain = ppo_clipped_loss(new, old, adv, mask)
    dual = ppo_clipped_loss(new, old, adv, mask, clip_c=10.0)
    # Without dual-clip the surrogate scales with the ratio; with it the loss is
    # capped at |A| * c = 10.
    assert plain.loss.item() > 100
    assert dual.loss.item() == pytest.approx(10.0)


def test_dual_clip_leaves_positive_advantages_alone():
    new = torch.full((1, 1), 5.0)
    old = torch.zeros(1, 1)
    adv = torch.tensor([[1.0]])
    mask = torch.ones(1, 1)
    plain = ppo_clipped_loss(new, old, adv, mask)
    dual = ppo_clipped_loss(new, old, adv, mask, clip_c=10.0)
    assert dual.loss.item() == pytest.approx(plain.loss.item())


def test_dual_clip_inactive_inside_the_trust_region():
    new = torch.full((1, 1), 0.05)
    old = torch.zeros(1, 1)
    for adv in (torch.tensor([[1.0]]), torch.tensor([[-1.0]])):
        plain = ppo_clipped_loss(new, old, adv, torch.ones(1, 1))
        dual = ppo_clipped_loss(new, old, adv, torch.ones(1, 1), clip_c=10.0)
        assert dual.loss.item() == pytest.approx(plain.loss.item())


# ---- config ----


def test_dapo_config_enables_all_four_techniques():
    cfg = Config.from_yaml("configs/dapo.yaml")
    assert cfg.algo.clip_ratio_high > cfg.algo.clip_ratio_low   # clip-higher
    assert cfg.algo.loss_normalization == "token"                # token-level loss
    assert cfg.algo.dynamic_sampling                             # dynamic sampling
    assert cfg.reward.overlong_max > 0                           # overlong shaping
    assert cfg.algo.clip_ratio_c == 10.0                         # dual-clip
    assert cfg.algo.kl_coef == 0.0                               # DAPO drops KL
    assert cfg.model.dtype == "float32"


def test_resampling_is_opt_in_not_the_default():
    """Group filtering is on; refilling the batch by regenerating is not.

    DAPO's own ablation reports 50% AIME without dynamic sampling, so resampling is
    not what produces the headline number -- but it triples generation cost. It is
    exercised by the dapo_dyn variant instead of being the default.
    """
    cfg = Config.from_yaml("configs/dapo.yaml")
    assert cfg.algo.dynamic_sampling
    assert cfg.algo.gen_batch_prompts == 0
    # And when it is enabled, it must oversample relative to the target.
    on = cfg.apply_overrides(["algo.gen_batch_prompts=32"])
    assert on.algo.gen_batch_prompts > on.rollout.prompts_per_step


def test_dapo_does_not_both_mask_and_penalize_overlong():
    """Masking truncated rollouts *and* penalizing their length would punish the
    same behaviour twice while also discarding the gradient."""
    cfg = Config.from_yaml("configs/dapo.yaml")
    assert not (cfg.algo.mask_truncated and cfg.reward.overlong_max > 0)
