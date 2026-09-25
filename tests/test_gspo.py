"""Tests for GSPO's sequence-level importance ratio."""

import math

import pytest
import torch

from llm_rl.algos import gspo as gspo_algo
from llm_rl.config import Config
from llm_rl.losses import gspo_loss, ppo_clipped_loss


def test_sequence_ratio_is_the_geometric_mean_of_token_ratios():
    """s_i = exp(mean_t log ratio_t), so a sequence whose token ratios are 2x and
    0.5x has a sequence ratio of exactly 1."""
    old = torch.zeros(1, 2)
    new = torch.tensor([[math.log(2.0), math.log(0.5)]])
    out = gspo_loss(new, old, torch.ones(1, 2), torch.ones(1, 2))
    assert out.metrics["policy/seq_ratio_mean"] == pytest.approx(1.0, abs=1e-6)
    assert out.metrics["policy/clip_frac"] == pytest.approx(0.0)


def test_length_normalization_makes_ratio_independent_of_length():
    """Two sequences with the same per-token log ratio get the same s_i even at
    very different lengths. Without the 1/|y| exponent the longer one would be
    astronomically further from 1."""
    delta = 0.001
    ratios = []
    for length in (4, 400):
        old = torch.zeros(1, length)
        new = torch.full((1, length), delta)
        out = gspo_loss(new, old, torch.ones(1, length), torch.ones(1, length))
        ratios.append(out.metrics["policy/seq_ratio_mean"])
    assert ratios[0] == pytest.approx(ratios[1], abs=1e-6)
    assert ratios[0] == pytest.approx(math.exp(delta), abs=1e-6)


def test_at_ratio_one_gradient_equals_plain_policy_gradient_direction():
    old = torch.zeros(1, 4)
    new = torch.zeros(1, 4, requires_grad=True)
    adv = torch.full((1, 4), 1.0)
    out = gspo_loss(new, old, adv, torch.ones(1, 4))
    out.loss.backward()
    # Positive advantage: descent must raise logprobs, so the gradient is negative,
    # and length normalization spreads it evenly at 1/|y| per token.
    assert (new.grad < 0).all()
    torch.testing.assert_close(new.grad, torch.full((1, 4), -0.25))


def test_clipping_is_per_sequence_not_per_token():
    """One wildly off-policy token cannot by itself clip the sequence; what matters
    is the mean. This is the whole point of GSPO."""
    old = torch.zeros(1, 100)
    new = torch.zeros(1, 100)
    new[0, 0] = 5.0  # one token at ratio ~148
    adv = torch.ones(1, 100)
    # mean log ratio = 5/100 = 0.05, far outside 4e-4, so it does clip here...
    out = gspo_loss(new, old, adv, torch.ones(1, 100), clip_low=3e-4, clip_high=4e-4)
    assert out.metrics["policy/clip_frac"] == pytest.approx(1.0)
    # ...but with a longer sequence the same single token is diluted below the bound.
    old_long, new_long = torch.zeros(1, 20000), torch.zeros(1, 20000)
    new_long[0, 0] = 5.0  # mean log ratio = 2.5e-4 < 4e-4
    out_long = gspo_loss(
        new_long, old_long, torch.ones(1, 20000), torch.ones(1, 20000),
        clip_low=3e-4, clip_high=4e-4,
    )
    assert out_long.metrics["policy/clip_frac"] == pytest.approx(0.0)


def test_clipped_sequence_gets_no_gradient():
    old = torch.zeros(1, 4)
    new = torch.full((1, 4), 0.01, requires_grad=True)  # ratio ~1.01 >> 1 + 4e-4
    out = gspo_loss(new, old, torch.ones(1, 4), torch.ones(1, 4))
    assert out.metrics["policy/clip_frac"] == pytest.approx(1.0)
    out.loss.backward()
    torch.testing.assert_close(new.grad, torch.zeros(1, 4))


def test_mask_excludes_prompt_and_padding_from_the_ratio():
    old = torch.zeros(1, 4)
    new = torch.tensor([[99.0, 0.001, 0.001, 99.0]])
    mask = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
    out = gspo_loss(new, old, torch.ones(1, 4), mask)
    # Only the two masked-in tokens count: mean log ratio = 0.001.
    assert out.metrics["policy/seq_ratio_mean"] == pytest.approx(math.exp(0.001), abs=1e-6)


def test_fully_masked_sequences_do_not_break_the_mean():
    old = torch.zeros(2, 3)
    new = torch.zeros(2, 3)
    mask = torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    out = gspo_loss(new, old, torch.ones(2, 3), mask)
    assert torch.isfinite(out.loss)


def test_each_sequence_weighs_equally_regardless_of_length():
    """GSPO has no token/sequence normalization choice: a short and a long sequence
    contribute the same amount. Contrast with GRPO's token-level loss."""
    old = torch.zeros(2, 10)
    new = torch.zeros(2, 10, requires_grad=True)
    adv = torch.ones(2, 10)
    mask = torch.zeros(2, 10)
    mask[0, :2] = 1.0    # short
    mask[1, :10] = 1.0   # long
    out = gspo_loss(new, old, adv, mask)
    out.loss.backward()
    # Each sequence's gradient sums to the same total magnitude (1/G each).
    torch.testing.assert_close(
        new.grad[0].sum().abs(), new.grad[1].sum().abs(), atol=1e-6, rtol=0
    )


def test_grpo_clip_range_would_be_a_no_op_and_is_rejected():
    """A sequence ratio essentially never leaves 1 +/- 0.2, so GRPO's range would
    silently disable the trust region."""
    old = torch.zeros(1, 100)
    new = torch.full((1, 100), 0.001)  # ratio ~1.001
    out = gspo_loss(new, old, torch.ones(1, 100), torch.ones(1, 100), 0.2, 0.28)
    assert out.metrics["policy/clip_frac"] == pytest.approx(0.0)

    bad = gspo_algo.preset()
    bad.clip_ratio_low, bad.clip_ratio_high = 0.2, 0.28
    with pytest.raises(ValueError, match="1e-4"):
        gspo_algo.validate(bad)


def test_gspo_differs_from_token_level_grpo_on_the_same_inputs():
    torch.manual_seed(0)
    old = torch.zeros(2, 8)
    new = torch.randn(2, 8) * 0.01
    adv = torch.randn(2, 1).expand(2, 8).contiguous()
    mask = torch.ones(2, 8)
    seq = gspo_loss(new, old, adv, mask, 3e-4, 4e-4)
    tok = ppo_clipped_loss(new, old, adv, mask, 0.2, 0.28)
    assert seq.loss.item() != pytest.approx(tok.loss.item())
    assert "policy/seq_ratio_mean" in seq.metrics
    assert "policy/ratio_mean" in tok.metrics


def test_gspo_preset_and_config():
    gspo_algo.validate(gspo_algo.preset())
    cfg = Config.from_yaml("configs/gspo.yaml")
    assert cfg.algo.name == "gspo"
    assert cfg.algo.normalize_advantage_std is True   # paper eq. 4/9, unlike Dr.GRPO
    assert cfg.algo.clip_ratio_low == pytest.approx(3e-4)
    assert cfg.algo.clip_ratio_high == pytest.approx(4e-4)
    assert cfg.algo.clip_ratio_c is None
    gspo_algo.validate(cfg.algo)
    # Inherited from dapo.yaml, so the comparison against DAPO is controlled.
    assert cfg.model.name == "Qwen/Qwen3.5-9B-Base"
    assert cfg.reward.overlong_max == 6144
