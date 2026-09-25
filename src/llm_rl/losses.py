"""Policy and value losses.

The second of the two files that differ between algorithms. REINFORCE uses the plain
policy-gradient term, GRPO and PPO use the clipped ratio; everything else (rollout,
reward, logprobs, optimizer) is shared.

The loss normalization mode is a config flag because it is not a detail: GRPO as
published averages over tokens *within* a sequence and then over sequences, which
weights a 50-token sequence the same as a 2000-token one. Dr.GRPO and DAPO instead
divide by the total token count in the batch. On long-form math reasoning the two
give visibly different length dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .logprobs import kl_penalty, masked_mean


@dataclass
class LossOutput:
    loss: torch.Tensor
    metrics: dict[str, float] = field(default_factory=dict)


def normalize_masked(
    values: torch.Tensor, mask: torch.Tensor, mode: str = "token"
) -> torch.Tensor:
    """Reduce a per-token quantity to a scalar.

    "token":    sum(values) / sum(mask) over the whole batch. Every token carries
                equal weight, so long sequences contribute proportionally more.
    "sequence": mean within each sequence, then mean across sequences. Every
                sequence carries equal weight regardless of length.
    """
    mask = mask.to(values.dtype)
    if mode == "token":
        return (values * mask).sum() / mask.sum().clamp(min=1.0)
    if mode == "sequence":
        per_sequence = (values * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
        # Sequences that are entirely masked out must not drag the mean toward zero.
        alive = (mask.sum(dim=-1) > 0).to(values.dtype)
        return (per_sequence * alive).sum() / alive.sum().clamp(min=1.0)
    raise ValueError(f"unknown loss normalization {mode!r} (expected token or sequence)")


def policy_gradient_loss(
    logprobs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    normalization: str = "token",
) -> LossOutput:
    """Plain REINFORCE / vanilla policy gradient: -A * log pi(a|s).

    Valid only on-policy, i.e. exactly one gradient step per batch of rollouts.
    """
    per_token = -advantages * logprobs
    loss = normalize_masked(per_token, mask, normalization)
    return LossOutput(
        loss=loss,
        metrics={
            "loss/policy": loss.item(),
            "policy/logprob_mean": masked_mean(logprobs, mask).item(),
        },
    )


def ppo_clipped_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_low: float = 0.2,
    clip_high: float = 0.2,
    normalization: str = "token",
    clip_c: float | None = None,
) -> LossOutput:
    """PPO / GRPO surrogate with an asymmetric trust region.

    `clip_high > clip_low` is DAPO's "clip-higher": the upper bound on the ratio is
    loosened so that low-probability tokens can still be reinforced, which measurably
    delays entropy collapse. With clip_low == clip_high this is standard PPO.

    `clip_c` enables dual-clip. The standard surrogate is unbounded above for
    negative advantages: if the ratio blows up on a token we are trying to suppress,
    the loss blows up with it. Dual-clip caps that branch at ``|A| * clip_c``
    (verl's DAPO recipe uses 10.0). It only touches tokens with A < 0.
    """
    log_ratio = logprobs - old_logprobs
    ratio = torch.exp(log_ratio)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high) * advantages
    # min() over the two surrogates makes this a pessimistic (lower) bound.
    per_token = -torch.min(unclipped, clipped)

    if clip_c is not None and clip_c > 1.0:
        ceiling = -advantages * clip_c  # positive exactly where advantages < 0
        per_token = torch.where(advantages < 0, torch.min(per_token, ceiling), per_token)

    loss = normalize_masked(per_token, mask, normalization)

    with torch.no_grad():
        clipped_frac = masked_mean((unclipped > clipped).float(), mask)
        # Schulman's k3 estimator of KL(old || new); the usual PPO early-stop signal.
        approx_kl = masked_mean(torch.expm1(-log_ratio) + log_ratio, mask)
    return LossOutput(
        loss=loss,
        metrics={
            "loss/policy": loss.item(),
            "policy/ratio_mean": masked_mean(ratio, mask).item(),
            "policy/clip_frac": clipped_frac.item(),
            "policy/approx_kl": approx_kl.item(),
        },
    )


def gspo_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_low: float = 3e-4,
    clip_high: float = 4e-4,
) -> LossOutput:
    r"""GSPO: Group Sequence Policy Optimization (arXiv 2507.18071, Qwen team).

    The one idea: the importance ratio should be defined on the *sequence*, not on
    each token, because the reward is a property of the sequence. GRPO reweights
    every token by its own ratio, which has no clean justification when the reward
    was earned by the whole response, and whose variance accumulates over thousands
    of tokens.

    GSPO uses the length-normalized sequence likelihood ratio -- equivalently, the
    geometric mean of the token ratios:

        s_i = (pi_theta(y_i|x) / pi_old(y_i|x)) ^ (1/|y_i|)
            = exp( (1/|y_i|) * sum_t log(pi_theta / pi_old) )

    and clips *that*:

        J = E[ (1/G) sum_i min(s_i * A_i, clip(s_i, 1-eps, 1+eps) * A_i) ]

    Two consequences worth knowing before you touch the knobs:

    * The clipping range is three orders of magnitude tighter than GRPO's. A
      geometric mean over thousands of tokens sits in a far narrower band than any
      single token ratio, so the paper uses ~3e-4 / 4e-4 against GRPO's 0.2 / 0.28.
      Passing GRPO-sized values here silently disables the trust region.
    * The loss is inherently per-sequence, so `loss_normalization` does not apply.
      Differentiating exp of a length-normalized sum gives every token in a sequence
      weight s_i * A_i / |y_i|, i.e. each sequence contributes equally regardless of
      length -- the length bias that DAPO's token-level normalization fixes in GRPO
      simply does not arise here.
    """
    mask = mask.to(logprobs.dtype)
    lengths = mask.sum(dim=-1)
    safe_lengths = lengths.clamp(min=1.0)

    # log of the geometric mean of per-token ratios.
    mean_log_ratio = ((logprobs - old_logprobs) * mask).sum(dim=-1) / safe_lengths
    seq_ratio = torch.exp(mean_log_ratio)

    # Advantages arrive broadcast across tokens; recover the per-sequence scalar.
    seq_advantage = (advantages * mask).sum(dim=-1) / safe_lengths

    unclipped = seq_ratio * seq_advantage
    clipped = torch.clamp(seq_ratio, 1.0 - clip_low, 1.0 + clip_high) * seq_advantage
    per_sequence = -torch.min(unclipped, clipped)

    alive = (lengths > 0).to(per_sequence.dtype)
    n_alive = alive.sum().clamp(min=1.0)
    loss = (per_sequence * alive).sum() / n_alive

    with torch.no_grad():
        clip_frac = ((unclipped > clipped).to(alive.dtype) * alive).sum() / n_alive
        approx_kl = ((torch.expm1(-mean_log_ratio) + mean_log_ratio) * alive).sum() / n_alive
    return LossOutput(
        loss=loss,
        metrics={
            "loss/policy": loss.item(),
            "policy/seq_ratio_mean": ((seq_ratio * alive).sum() / n_alive).item(),
            "policy/clip_frac": clip_frac.item(),
            "policy/approx_kl": approx_kl.item(),
        },
    )


def value_loss(
    values: torch.Tensor,
    returns: torch.Tensor,
    mask: torch.Tensor,
    old_values: torch.Tensor | None = None,
    clip_range: float | None = None,
    normalization: str = "token",
) -> LossOutput:
    """Critic regression, optionally with PPO's clipped value objective."""
    unclipped = (values - returns).square()
    if old_values is not None and clip_range is not None:
        clamped = old_values + (values - old_values).clamp(-clip_range, clip_range)
        per_token = torch.max(unclipped, (clamped - returns).square())
    else:
        per_token = unclipped
    loss = 0.5 * normalize_masked(per_token, mask, normalization)
    with torch.no_grad():
        explained = _explained_variance(values, returns, mask)
    return LossOutput(
        loss=loss,
        metrics={
            "loss/value": loss.item(),
            "value/mean": masked_mean(values, mask).item(),
            "value/explained_variance": explained,
        },
    )


def _explained_variance(values: torch.Tensor, returns: torch.Tensor, mask: torch.Tensor) -> float:
    mask = mask.to(values.dtype)
    count = mask.sum().clamp(min=1.0)
    mean_return = (returns * mask).sum() / count
    var_return = ((returns - mean_return).square() * mask).sum() / count
    var_resid = ((returns - values).square() * mask).sum() / count
    if var_return.item() < 1e-8:
        return float("nan")
    return (1.0 - var_resid / var_return).item()


def compose_loss(
    policy: LossOutput,
    mask: torch.Tensor,
    entropy: torch.Tensor | None = None,
    entropy_coef: float = 0.0,
    logprobs: torch.Tensor | None = None,
    ref_logprobs: torch.Tensor | None = None,
    kl_coef: float = 0.0,
    kl_estimator: str = "k3",
    value: LossOutput | None = None,
    value_coef: float = 0.5,
    normalization: str = "token",
) -> LossOutput:
    """Combine the policy term with optional entropy, KL and value terms."""
    total = policy.loss
    metrics = dict(policy.metrics)

    if entropy is not None and entropy_coef:
        entropy_term = normalize_masked(entropy, mask, normalization)
        total = total - entropy_coef * entropy_term
        metrics["policy/entropy"] = entropy_term.item()

    if kl_coef:
        if logprobs is None or ref_logprobs is None:
            raise ValueError("kl_coef > 0 requires both logprobs and ref_logprobs")
        kl_term = normalize_masked(kl_penalty(logprobs, ref_logprobs, kl_estimator), mask, normalization)
        total = total + kl_coef * kl_term
        metrics["policy/kl_to_ref"] = kl_term.item()

    if value is not None:
        total = total + value_coef * value.loss
        metrics.update(value.metrics)

    metrics["loss/total"] = total.item()
    return LossOutput(loss=total, metrics=metrics)
