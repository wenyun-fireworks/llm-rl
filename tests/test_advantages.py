import pytest
import torch

from llm_rl.advantages import (
    compute_group_advantages,
    gae_advantages,
    grpo_advantages,
    masked_whiten,
    nonzero_variance_groups,
    rloo_advantages,
)


def test_rloo_matches_closed_form():
    rewards = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    # baseline for the winner is 0, for each loser it is 1/3.
    expected = torch.tensor([[1.0, -1 / 3, -1 / 3, -1 / 3]])
    torch.testing.assert_close(rloo_advantages(rewards), expected)


def test_rloo_is_unbiased_zero_sum_only_in_expectation():
    # Leave-one-out advantages do not sum to zero within a group (that is the point:
    # the baseline is independent of the sample it scores).
    rewards = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    adv = rloo_advantages(rewards)
    torch.testing.assert_close(adv, torch.tensor([[2 / 3, 2 / 3, -2 / 3, -2 / 3]]))


def test_grpo_centers_and_normalizes():
    rewards = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    adv = grpo_advantages(rewards, normalize_std=True)
    assert adv.mean().abs() < 1e-5
    torch.testing.assert_close(adv.abs(), torch.ones_like(adv), atol=1e-3, rtol=1e-3)


def test_grpo_without_std_is_plain_centering():
    rewards = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    torch.testing.assert_close(
        grpo_advantages(rewards, normalize_std=False),
        rewards - 0.25,
    )


def test_grpo_degenerate_group_gives_zero_not_nan():
    rewards = torch.zeros(2, 8)
    adv = grpo_advantages(rewards, normalize_std=True)
    assert torch.isfinite(adv).all()
    torch.testing.assert_close(adv, torch.zeros_like(adv))


def test_group_estimators_are_per_prompt():
    rewards = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    adv = grpo_advantages(rewards, normalize_std=False)
    torch.testing.assert_close(adv[1], torch.zeros(2))
    assert adv[0].tolist() == [0.5, -0.5]


def test_compute_group_advantages_dispatch():
    rewards = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    torch.testing.assert_close(compute_group_advantages(rewards, "rloo"), rloo_advantages(rewards))
    torch.testing.assert_close(
        compute_group_advantages(rewards, "grpo", normalize_std=False),
        grpo_advantages(rewards, normalize_std=False),
    )
    with pytest.raises(ValueError):
        compute_group_advantages(rewards, "nope")


def test_bad_shapes_rejected():
    with pytest.raises(ValueError):
        grpo_advantages(torch.zeros(4))
    with pytest.raises(ValueError):
        rloo_advantages(torch.zeros(2, 1))


def test_gae_with_lambda_one_is_reward_to_go_minus_value():
    # gamma=lambda=1, zero interior reward, terminal reward 1 -> every advantage is
    # 1 - V(s_t).
    values = torch.tensor([[0.2, 0.4, 0.6]])
    rewards = torch.tensor([[0.0, 0.0, 1.0]])
    mask = torch.ones(1, 3)
    adv, returns = gae_advantages(rewards, values, mask, gamma=1.0, lam=1.0)
    torch.testing.assert_close(adv, 1.0 - values)
    torch.testing.assert_close(returns, torch.ones(1, 3))


def test_gae_zero_advantage_when_value_is_perfect():
    values = torch.tensor([[1.0, 1.0, 1.0]])
    rewards = torch.tensor([[0.0, 0.0, 1.0]])
    adv, _ = gae_advantages(rewards, values, torch.ones(1, 3))
    torch.testing.assert_close(adv, torch.zeros(1, 3), atol=1e-6, rtol=0)


def test_gae_respects_mask_as_per_sequence_terminal():
    # Sequence 0 ends at t=1; the padding at t=2 must not bootstrap into it.
    values = torch.tensor([[0.5, 0.5, 99.0]])
    rewards = torch.tensor([[0.0, 1.0, 99.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    adv, returns = gae_advantages(rewards, values, mask, gamma=1.0, lam=1.0)
    assert adv[0, 2] == 0.0 and returns[0, 2] == 0.0
    torch.testing.assert_close(adv[0, :2], torch.tensor([0.5, 0.5]))


def test_gae_discounting():
    values = torch.zeros(1, 3)
    rewards = torch.tensor([[0.0, 0.0, 1.0]])
    adv, _ = gae_advantages(rewards, values, torch.ones(1, 3), gamma=0.5, lam=1.0)
    torch.testing.assert_close(adv, torch.tensor([[0.25, 0.5, 1.0]]))


def test_gae_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        gae_advantages(torch.zeros(1, 3), torch.zeros(1, 4), torch.ones(1, 3))


def test_masked_whiten_ignores_padding():
    values = torch.tensor([[1.0, 2.0, 3.0, 1000.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    out = masked_whiten(values, mask)
    assert out[0, 3] == 0.0
    torch.testing.assert_close(out[0, :3].mean(), torch.tensor(0.0), atol=1e-5, rtol=0)
    torch.testing.assert_close(out[0, :3].std(unbiased=False), torch.tensor(1.0), atol=1e-3, rtol=0)


def test_nonzero_variance_groups():
    rewards = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    assert nonzero_variance_groups(rewards).tolist() == [True, False, False]
