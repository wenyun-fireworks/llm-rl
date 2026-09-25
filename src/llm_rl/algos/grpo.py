"""GRPO: group-relative policy optimization.

Like REINFORCE it avoids a critic by baselining against the group, but it uses the
group mean (not leave-one-out) and adds PPO's clipped ratio so the batch can be
reused for several gradient steps.

Two published variants matter and both are reachable from config:

* Original GRPO divides the centered reward by the group standard deviation and
  normalizes the loss per sequence.
* Dr.GRPO / DAPO drop the std division (it up-weights low-variance groups) and
  normalize by total token count instead (per-sequence normalization makes short
  sequences count as much as long ones, biasing length).

DAPO's clip-higher (`clip_ratio_high` above `clip_ratio_low`) and dynamic sampling
(dropping groups whose rewards are all equal) are also config flags.
"""

from __future__ import annotations

from ..config import AlgoConfig


def preset(dr_grpo: bool = True) -> AlgoConfig:
    return AlgoConfig(
        name="grpo",
        advantage="grpo",
        normalize_advantage_std=not dr_grpo,
        loss_normalization="token" if dr_grpo else "sequence",
        clip_ratio_low=0.2,
        clip_ratio_high=0.28,  # clip-higher, delays entropy collapse
        ppo_epochs=1,
        dynamic_sampling=True,
    )


def validate(cfg: AlgoConfig) -> None:
    if cfg.advantage != "grpo":
        raise ValueError(f"grpo expects advantage=grpo, got {cfg.advantage!r}")
    if cfg.clip_ratio_high < cfg.clip_ratio_low:
        raise ValueError(
            "clip_ratio_high must be >= clip_ratio_low; a tighter upper bound than "
            "lower bound would penalize exactly the tokens GRPO is trying to reinforce"
        )
