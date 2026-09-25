"""GSPO: Group Sequence Policy Optimization (arXiv 2507.18071, Qwen team).

Same group-relative advantage as GRPO, but the importance ratio and the clipping
move from the token to the sequence. The argument is one of matching units: the
reward is earned by the whole response, so the importance weight should be the
whole response's likelihood ratio. Token-level ratios have no clean justification
under a sequence-level reward, and their variance compounds over long generations.

The paper's headline practical claim is stability, especially for MoE models: GSPO
removes the need for "Routing Replay" because it only depends on the sequence
likelihood, not on which experts fired for any individual token.
"""

from __future__ import annotations

from ..config import AlgoConfig

# The paper's values. Three orders of magnitude tighter than GRPO's 0.2/0.28,
# because a geometric mean over thousands of token ratios sits in a much narrower
# band than any single token ratio.
CLIP_LOW = 3e-4
CLIP_HIGH = 4e-4


def preset() -> AlgoConfig:
    return AlgoConfig(
        name="gspo",
        advantage="grpo",
        # GSPO's objective (eq. 5/7) uses the std-normalized group advantage,
        # unlike the Dr.GRPO variant we ran for the DAPO experiments.
        normalize_advantage_std=True,
        clip_ratio_low=CLIP_LOW,
        clip_ratio_high=CLIP_HIGH,
        clip_ratio_c=None,  # dual-clip is a token-level patch; not applicable
        ppo_epochs=1,
        dynamic_sampling=True,
    )


def validate(cfg: AlgoConfig) -> None:
    if cfg.advantage != "grpo":
        raise ValueError(f"gspo expects advantage=grpo, got {cfg.advantage!r}")
    # The most likely mistake is carrying GRPO's clip range over, which would leave
    # the trust region wide open: a sequence ratio essentially never leaves 1 +/- 0.2.
    if cfg.clip_ratio_low > 0.01 or cfg.clip_ratio_high > 0.01:
        raise ValueError(
            "gspo clips a length-normalized sequence ratio, so the range must be "
            f"~1e-4, not GRPO's ~0.2. Got low={cfg.clip_ratio_low}, "
            f"high={cfg.clip_ratio_high}. See algos/gspo.py:CLIP_LOW/CLIP_HIGH."
        )
    if cfg.clip_ratio_c is not None:
        raise ValueError("clip_ratio_c (dual-clip) is token-level; leave it null for gspo")
