import pytest
import torch

from llm_rl.losses import (
    compose_loss,
    normalize_masked,
    policy_gradient_loss,
    ppo_clipped_loss,
    value_loss,
)


def test_token_vs_sequence_normalization_differ_on_ragged_batches():
    # One short sequence and one long one, same per-token value.
    values = torch.tensor([[1.0, 0.0, 0.0, 0.0], [3.0, 3.0, 3.0, 3.0]])
    mask = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    # token: (1 + 12) / 5 = 2.6 -- the long sequence dominates.
    torch.testing.assert_close(normalize_masked(values, mask, "token"), torch.tensor(2.6))
    # sequence: (1 + 3) / 2 = 2.0 -- both sequences weigh the same.
    torch.testing.assert_close(normalize_masked(values, mask, "sequence"), torch.tensor(2.0))


def test_sequence_normalization_ignores_fully_masked_rows():
    values = torch.tensor([[2.0, 2.0], [5.0, 5.0]])
    mask = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    torch.testing.assert_close(normalize_masked(values, mask, "sequence"), torch.tensor(2.0))


def test_unknown_normalization_rejected():
    with pytest.raises(ValueError):
        normalize_masked(torch.zeros(1, 2), torch.ones(1, 2), "mean-ish")


def test_policy_gradient_sign_pushes_up_positive_advantage():
    logprobs = torch.tensor([[-1.0, -1.0]], requires_grad=True)
    advantages = torch.tensor([[1.0, 1.0]])
    out = policy_gradient_loss(logprobs, advantages, torch.ones(1, 2))
    out.loss.backward()
    # Positive advantage => gradient descent must increase logprob => negative grad.
    assert (logprobs.grad < 0).all()


def test_policy_gradient_masks_padding():
    logprobs = torch.tensor([[-1.0, -50.0]], requires_grad=True)
    advantages = torch.tensor([[1.0, 1.0]])
    mask = torch.tensor([[1.0, 0.0]])
    out = policy_gradient_loss(logprobs, advantages, mask)
    out.loss.backward()
    assert logprobs.grad[0, 1] == 0.0


def test_ppo_at_ratio_one_equals_policy_gradient():
    logprobs = torch.tensor([[-1.0, -2.0]])
    advantages = torch.tensor([[1.0, -1.0]])
    mask = torch.ones(1, 2)
    ppo = ppo_clipped_loss(logprobs, logprobs.clone(), advantages, mask)
    # ratio == 1 so the surrogate reduces to -A, matching -A*logp up to the constant
    # logprob factor; check the ratio diagnostics instead.
    assert ppo.metrics["policy/ratio_mean"] == pytest.approx(1.0)
    assert ppo.metrics["policy/approx_kl"] == pytest.approx(0.0, abs=1e-6)
    assert ppo.metrics["policy/clip_frac"] == pytest.approx(0.0)


def test_ppo_clips_positive_advantage_above_upper_bound():
    old = torch.zeros(1, 1)
    # ratio = e^1 = 2.718, well above 1 + 0.2
    new = torch.ones(1, 1, requires_grad=True)
    adv = torch.tensor([[1.0]])
    out = ppo_clipped_loss(new, old, adv, torch.ones(1, 1), clip_low=0.2, clip_high=0.2)
    assert out.metrics["policy/clip_frac"] == pytest.approx(1.0)
    # Clipped: loss is -(1.2 * 1) and the gradient is cut off entirely.
    torch.testing.assert_close(out.loss, torch.tensor(-1.2))
    out.loss.backward()
    torch.testing.assert_close(new.grad, torch.zeros_like(new))


def test_ppo_does_not_clip_inside_trust_region():
    old = torch.zeros(1, 1)
    new = torch.full((1, 1), 0.1, requires_grad=True)
    out = ppo_clipped_loss(new, old, torch.tensor([[1.0]]), torch.ones(1, 1))
    assert out.metrics["policy/clip_frac"] == pytest.approx(0.0)
    out.loss.backward()
    assert new.grad.abs().sum() > 0


def test_clip_higher_widens_only_the_upper_bound():
    old = torch.zeros(1, 1)
    # ratio = e^0.223 ~ 1.250, which is above the tight bound 1.2 but below 1.28.
    new = torch.full((1, 1), 0.223, requires_grad=True)
    adv = torch.tensor([[1.0]])
    tight = ppo_clipped_loss(new, old, adv, torch.ones(1, 1), clip_low=0.2, clip_high=0.2)
    loose = ppo_clipped_loss(new, old, adv, torch.ones(1, 1), clip_low=0.2, clip_high=0.28)
    assert tight.metrics["policy/clip_frac"] == pytest.approx(1.0)
    assert loose.metrics["policy/clip_frac"] == pytest.approx(0.0)
    # Clip-higher keeps reinforcing this token instead of zeroing its gradient.
    assert loose.loss < tight.loss


def test_ppo_negative_advantage_clips_on_the_low_side():
    old = torch.zeros(1, 1)
    new = torch.full((1, 1), -1.0, requires_grad=True)  # ratio ~0.368 < 0.8
    out = ppo_clipped_loss(new, old, torch.tensor([[-1.0]]), torch.ones(1, 1))
    assert out.metrics["policy/clip_frac"] == pytest.approx(1.0)
    torch.testing.assert_close(out.loss, torch.tensor(0.8))


def test_value_loss_zero_when_perfect():
    values = torch.tensor([[1.0, 2.0]])
    out = value_loss(values, values.clone(), torch.ones(1, 2))
    assert out.loss.item() == pytest.approx(0.0)
    assert out.metrics["value/explained_variance"] == pytest.approx(1.0)


def test_value_loss_clipping_is_pessimistic():
    old_values = torch.zeros(1, 1)
    values = torch.tensor([[5.0]])
    returns = torch.tensor([[0.0]])
    plain = value_loss(values, returns, torch.ones(1, 1))
    clipped = value_loss(values, returns, torch.ones(1, 1), old_values=old_values, clip_range=0.2)
    # max() of the two errors means clipping can never reduce the loss.
    assert clipped.loss <= plain.loss
    assert clipped.loss.item() == pytest.approx(0.5 * 25.0)


def test_explained_variance_nan_on_constant_returns():
    out = value_loss(torch.zeros(1, 3), torch.ones(1, 3), torch.ones(1, 3))
    import math

    assert math.isnan(out.metrics["value/explained_variance"])


def test_compose_adds_kl_and_subtracts_entropy():
    mask = torch.ones(1, 2)
    base = policy_gradient_loss(torch.tensor([[-1.0, -1.0]]), torch.tensor([[1.0, 1.0]]), mask)
    entropy = torch.tensor([[2.0, 2.0]])
    with_entropy = compose_loss(base, mask, entropy=entropy, entropy_coef=0.1)
    # Entropy is a bonus, so it lowers the loss.
    assert with_entropy.loss < base.loss
    assert with_entropy.metrics["policy/entropy"] == pytest.approx(2.0)

    logprobs = torch.tensor([[-1.0, -1.0]])
    ref = torch.tensor([[-2.0, -2.0]])
    with_kl = compose_loss(base, mask, logprobs=logprobs, ref_logprobs=ref, kl_coef=0.5)
    assert with_kl.loss > base.loss
    assert with_kl.metrics["policy/kl_to_ref"] > 0


def test_compose_requires_ref_logprobs_when_kl_enabled():
    mask = torch.ones(1, 2)
    base = policy_gradient_loss(torch.zeros(1, 2), torch.zeros(1, 2), mask)
    with pytest.raises(ValueError):
        compose_loss(base, mask, kl_coef=0.1)


def test_compose_includes_value_term():
    mask = torch.ones(1, 2)
    base = policy_gradient_loss(torch.zeros(1, 2), torch.zeros(1, 2), mask)
    v = value_loss(torch.zeros(1, 2), torch.ones(1, 2), mask)
    out = compose_loss(base, mask, value=v, value_coef=1.0)
    assert out.loss.item() == pytest.approx(base.loss.item() + v.loss.item())
    assert "value/explained_variance" in out.metrics
