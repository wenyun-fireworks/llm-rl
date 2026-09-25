"""REINFORCE with a leave-one-out baseline (RLOO).

The simplest of the three: no critic, no ratio, no clipping. The baseline for each
sample is the mean reward of its siblings, which is unbiased precisely because the
sample is excluded from its own baseline.

Strictly on-policy, so `ppo_epochs` must be 1: with a second pass over the same
batch the sampling distribution no longer matches the policy being updated, and the
plain policy-gradient estimator becomes invalid. Use GRPO or PPO if you want to
reuse rollouts.
"""

from __future__ import annotations

from ..config import AlgoConfig


def preset() -> AlgoConfig:
    return AlgoConfig(
        name="reinforce",
        advantage="rloo",
        normalize_advantage_std=False,
        loss_normalization="token",
        ppo_epochs=1,
        kl_coef=0.0,
    )


def validate(cfg: AlgoConfig) -> None:
    # "raw" is the no-baseline control (advantage = reward). Useful for showing what
    # the leave-one-out baseline buys; not a serious training choice.
    if cfg.advantage not in ("rloo", "raw"):
        raise ValueError(f"reinforce expects advantage=rloo or raw, got {cfg.advantage!r}")
    if cfg.ppo_epochs != 1:
        raise ValueError(
            "reinforce is on-policy only: ppo_epochs must be 1, "
            f"got {cfg.ppo_epochs}. Use grpo or ppo to reuse rollouts."
        )
