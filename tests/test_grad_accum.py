"""Gradient accumulation must be exactly equivalent to one big backward.

The accumulation weight has to match the loss's normalizer. A token-level loss
divides by the batch token count, so micro-batches must be weighted by their token
share; a sequence-level loss (GSPO) divides by the sequence count, so the weight is
a sequence share. Getting this wrong silently rescales the gradient in proportion
to how unevenly tokens are distributed across micro-batches -- which for ragged RL
batches is always.
"""

import pytest
import torch

from llm_rl.losses import gspo_loss, ppo_clipped_loss


def ragged_batch(seed=0):
    """Two sequences of very different length, so token and sequence shares differ."""
    torch.manual_seed(seed)
    # Deliberately uneven *across the halves*, not just within them: the first two
    # sequences hold 12 tokens and the last two hold 3, so token share (12/15 vs
    # 3/15) and sequence share (2/4 vs 2/4) genuinely disagree.
    mask = torch.zeros(4, 12)
    mask[0, :2] = 1.0
    mask[1, :10] = 1.0
    mask[2, :2] = 1.0
    mask[3, :1] = 1.0
    old = torch.zeros(4, 12)
    adv = torch.randn(4, 1).expand(4, 12).contiguous()
    return old, adv, mask


def grads_single(loss_fn, param, old, adv, mask):
    param.grad = None
    loss_fn(param, old, adv, mask).loss.backward()
    return param.grad.clone()


def grads_accumulated(loss_fn, param, old, adv, mask, split, by_sequence):
    param.grad = None
    if by_sequence:
        total = mask.sum(dim=-1).gt(0).float().sum().item()
    else:
        total = mask.sum().item()
    for sl in split:
        sub_mask = mask[sl]
        share = (
            sub_mask.sum(dim=-1).gt(0).float().sum().item() if by_sequence else sub_mask.sum().item()
        ) / total
        out = loss_fn(param[sl], old[sl], adv[sl], sub_mask)
        (out.loss * share).backward(retain_graph=True)
    return param.grad.clone()


@pytest.mark.parametrize("split", [
    [slice(0, 2), slice(2, 4)],
    [slice(0, 1), slice(1, 2), slice(2, 3), slice(3, 4)],
    [slice(0, 3), slice(3, 4)],
])
def test_token_level_accumulation_matches_single_backward(split):
    old, adv, mask = ragged_batch()
    param = (torch.randn(4, 12) * 0.01).requires_grad_(True)

    def loss_fn(new, o, a, m):
        return ppo_clipped_loss(new, o, a, m, normalization="token")

    single = grads_single(loss_fn, param, old, adv, mask)
    accum = grads_accumulated(loss_fn, param, old, adv, mask, split, by_sequence=False)
    torch.testing.assert_close(accum, single, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("split", [
    [slice(0, 2), slice(2, 4)],
    [slice(0, 1), slice(1, 4)],
])
def test_gspo_accumulation_matches_single_backward(split):
    old, adv, mask = ragged_batch()
    param = (torch.randn(4, 12) * 1e-5).requires_grad_(True)

    def loss_fn(new, o, a, m):
        return gspo_loss(new, o, a, m, clip_low=1.0, clip_high=1.0)  # wide: no clipping

    single = grads_single(loss_fn, param, old, adv, mask)
    accum = grads_accumulated(loss_fn, param, old, adv, mask, split, by_sequence=True)
    torch.testing.assert_close(accum, single, atol=1e-6, rtol=1e-5)


def test_using_the_wrong_share_basis_is_detectably_wrong():
    """Guards the bug this test file exists for: weighting a token-level loss by
    sequence share does not reproduce the true gradient on a ragged batch."""
    old, adv, mask = ragged_batch()
    param = (torch.randn(4, 12) * 0.01).requires_grad_(True)

    def loss_fn(new, o, a, m):
        return ppo_clipped_loss(new, o, a, m, normalization="token")

    single = grads_single(loss_fn, param, old, adv, mask)
    wrong = grads_accumulated(
        loss_fn, param, old, adv, mask, [slice(0, 2), slice(2, 4)], by_sequence=True
    )
    assert not torch.allclose(wrong, single, atol=1e-6)


def test_shares_sum_to_one():
    _, _, mask = ragged_batch()
    for by_sequence in (True, False):
        total = (
            mask.sum(dim=-1).gt(0).float().sum() if by_sequence else mask.sum()
        ).item()
        shares = []
        for sl in (slice(0, 2), slice(2, 4)):
            sub = mask[sl]
            shares.append(
                (sub.sum(dim=-1).gt(0).float().sum() if by_sequence else sub.sum()).item() / total
            )
        assert sum(shares) == pytest.approx(1.0)
