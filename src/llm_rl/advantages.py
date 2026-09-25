"""Advantage estimators.

This is one of the two files that differ between REINFORCE, GRPO and PPO; everything
else in the stack is shared. All group estimators take rewards shaped
``[num_prompts, group_size]`` and return advantages of the same shape, which the
trainer then broadcasts across the tokens of each sequence.
"""

from __future__ import annotations

import torch

EPS = 1e-6


def raw_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """No baseline at all: the advantage is the raw reward.

    This is the textbook REINFORCE estimator and it is included as a teaching
    control, not as a serious option. It is unbiased but high variance, and with a
    non-negative reward every gradient *increases* the logprob of every sampled
    sequence -- correct ones just get increased more. Compare its learning curve
    against `rloo` to see what a baseline actually buys.
    """
    _check_group(rewards, min_size=1)
    return rewards.clone()


def rloo_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """REINFORCE leave-one-out: baseline each sample on the mean of its siblings.

    Unbiased, unlike the group-mean baseline, because sample i is excluded from its
    own baseline.
    """
    _check_group(rewards, min_size=2)
    group_size = rewards.shape[-1]
    total = rewards.sum(dim=-1, keepdim=True)
    baseline = (total - rewards) / (group_size - 1)
    return rewards - baseline


def grpo_advantages(rewards: torch.Tensor, normalize_std: bool = True) -> torch.Tensor:
    """Group-relative advantage.

    ``normalize_std=True`` reproduces GRPO as published. ``False`` is the
    Dr.GRPO / DAPO variant, which drops the division because it up-weights groups
    that happen to have low reward variance.
    """
    _check_group(rewards, min_size=1)
    centered = rewards - rewards.mean(dim=-1, keepdim=True)
    if not normalize_std:
        return centered
    std = rewards.std(dim=-1, keepdim=True, unbiased=False)
    return centered / (std + EPS)


def gae_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token-level GAE(lambda) for PPO.

    All tensors are ``[batch, seq]`` over response tokens only. ``rewards`` carries
    the terminal correctness reward on the last valid token and, optionally, a
    per-token KL penalty in the interior. Returns ``(advantages, returns)``, both
    masked to zero on padding.
    """
    if rewards.shape != values.shape or rewards.shape != mask.shape:
        raise ValueError(
            f"shape mismatch: rewards {tuple(rewards.shape)}, "
            f"values {tuple(values.shape)}, mask {tuple(mask.shape)}"
        )
    mask = mask.to(rewards.dtype)
    values = values * mask

    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[:, 0])
    for t in reversed(range(rewards.shape[1])):
        # Past the end of a sequence the mask is 0, which zeroes both the
        # bootstrap value and the carried-over GAE term, so each sequence is
        # treated as terminating at its own final token.
        next_value = values[:, t + 1] if t + 1 < values.shape[1] else torch.zeros_like(running)
        next_mask = mask[:, t + 1] if t + 1 < mask.shape[1] else torch.zeros_like(running)
        delta = rewards[:, t] + gamma * next_value * next_mask - values[:, t]
        running = delta + gamma * lam * next_mask * running
        advantages[:, t] = running
    advantages = advantages * mask
    returns = (advantages + values) * mask
    return advantages, returns


def masked_whiten(values: torch.Tensor, mask: torch.Tensor, shift_mean: bool = True) -> torch.Tensor:
    """Normalize `values` over the masked entries only."""
    mask = mask.to(values.dtype)
    count = mask.sum().clamp(min=1.0)
    mean = (values * mask).sum() / count
    var = ((values - mean) ** 2 * mask).sum() / count
    whitened = (values - mean) * torch.rsqrt(var + EPS)
    if not shift_mean:
        whitened = whitened + mean
    return whitened * mask


def nonzero_variance_groups(rewards: torch.Tensor) -> torch.Tensor:
    """Boolean mask over prompts whose group has at least two distinct rewards.

    Groups that are all-correct or all-wrong produce an identically zero advantage
    under any group baseline; DAPO's dynamic sampling drops them so they do not
    dilute the batch.
    """
    _check_group(rewards, min_size=1)
    return (rewards.amax(dim=-1) - rewards.amin(dim=-1)).abs() > EPS


def compute_group_advantages(
    rewards: torch.Tensor,
    kind: str,
    normalize_std: bool = True,
) -> torch.Tensor:
    if kind == "raw":
        return raw_advantages(rewards)
    if kind == "rloo":
        return rloo_advantages(rewards)
    if kind == "grpo":
        return grpo_advantages(rewards, normalize_std=normalize_std)
    raise ValueError(
        f"unknown group advantage estimator {kind!r} (expected raw, rloo or grpo)"
    )


def _check_group(rewards: torch.Tensor, min_size: int) -> None:
    if rewards.ndim != 2:
        raise ValueError(f"expected rewards shaped [num_prompts, group_size], got {tuple(rewards.shape)}")
    if rewards.shape[-1] < min_size:
        raise ValueError(f"group_size must be >= {min_size}, got {rewards.shape[-1]}")
