"""PPO with a learned critic and token-level GAE.

The only one of the three that estimates a value function. A linear head on the
shared backbone predicts the return at every response token; GAE(lambda) then turns
the single terminal correctness reward into a per-token advantage.

This buys credit assignment within a sequence, which the group methods cannot do
(they broadcast one scalar across every token). It costs a critic that must be
learned from a very sparse signal, which is why PPO is often the weaker choice on
verifiable-reward math tasks despite being the more general algorithm.
"""

from __future__ import annotations

from ..config import AlgoConfig, ModelConfig


def preset() -> AlgoConfig:
    return AlgoConfig(
        name="ppo",
        advantage="gae",
        loss_normalization="token",
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        ppo_epochs=2,
        value_coef=0.5,
        gamma=1.0,
        # lambda=1 makes the advantage the full return minus the baseline. With a
        # single terminal reward and no discounting there is no intermediate signal
        # to trade bias against, so shrinking lambda only adds critic bias.
        gae_lambda=1.0,
    )


def validate(cfg: AlgoConfig, model_cfg: ModelConfig) -> None:
    if cfg.advantage != "gae":
        raise ValueError(f"ppo expects advantage=gae, got {cfg.advantage!r}")
    if not model_cfg.value_head:
        raise ValueError("ppo requires model.value_head=true so the critic exists")
    if not 0.0 <= cfg.gae_lambda <= 1.0:
        raise ValueError(f"gae_lambda must be in [0, 1], got {cfg.gae_lambda}")
