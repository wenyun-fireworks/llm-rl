"""Policy (and optional value) model loading for Qwen3.5.

Qwen3.5-0.8B-Base is a Qwen3_5ForConditionalGeneration: a text backbone plus a
vision tower, with tied embeddings over a 248,320-token vocab. We train text-only,
so the vision tower is frozen and excluded from the optimizer. The `mtp.*` tensors
in the checkpoint are not materialized by transformers at all, so they need no
handling beyond being tolerated as unused weights on load.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

from .config import ModelConfig

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


class ValueHead(nn.Module):
    """Scalar value per token, read off the shared backbone's hidden states.

    Kept in fp32: the critic target is an unbounded return and bf16's ~3 decimal
    digits of mantissa are not enough for a stable regression target.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1, dtype=torch.float32)
        # Zero init, so the critic predicts exactly 0 before it has learned
        # anything. This matters more than it looks. With a random init the head
        # emits values of order sqrt(hidden) * std * |h|, which measured out at
        # ~3 against a reward in {0, 1}: the advantage R - V is then dominated by
        # critic noise an order of magnitude larger than the real signal, and the
        # policy is destroyed within ~10 steps. Starting at V=0 makes the first
        # advantages equal the reward, which is merely unbaselined, not wrong.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states.to(torch.float32)).squeeze(-1)


@dataclass
class Policy:
    """A loaded policy: the HF model, its tokenizer, and an optional critic.

    Parameters are fp32 (see ModelConfig.dtype for why) while the forward pass runs
    under bf16 autocast, so activations and matmuls stay cheap but the optimizer
    still has the precision to apply a 1e-6 update.
    """

    model: Qwen3_5ForConditionalGeneration
    tokenizer: object
    value_head: ValueHead | None = None
    autocast_dtype: torch.dtype | None = torch.bfloat16
    # Train the critic on detached hidden states, so the value loss never
    # backpropagates into the shared backbone. Measured without it: the critic's
    # gradient norm reached ~29 against the policy's ~1.5, explained variance went
    # to -9 (far worse than predicting the mean), and the value loss reshaped the
    # representation until the policy degenerated to 4096-token non-answers.
    # The critic becomes a linear probe on the policy's features, which is weaker
    # in principle but does not let a failing critic destroy a working policy.
    detach_value_head: bool = True

    def autocast(self):
        if self.autocast_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast("cuda", dtype=self.autocast_dtype)

    @property
    def backbone(self) -> nn.Module:
        """The text transformer, bypassing the vision path entirely."""
        return self.model.model.language_model

    @property
    def lm_head(self) -> nn.Module:
        return self.model.lm_head

    @property
    def hidden_size(self) -> int:
        return self.model.config.text_config.hidden_size

    @property
    def vocab_size(self) -> int:
        return self.model.config.text_config.vocab_size

    def hidden_states(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Run the text backbone and return last hidden states [batch, seq, hidden].

        Deliberately stops before the LM head: at 248,320 vocab the logits for a
        single 4k sequence are ~2GB in bf16, so the caller chunks the projection
        (see logprobs.py) rather than materializing them.
        """
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        return out.last_hidden_state

    def trainable_parameters(self) -> list[nn.Parameter]:
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.value_head is not None:
            params += list(self.value_head.parameters())
        return params

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())


def freeze_non_text(model: nn.Module, freeze_vision: bool = True, freeze_mtp: bool = True) -> None:
    """Freeze everything we are not training text-only RL on."""
    if freeze_vision and hasattr(model.model, "visual"):
        for param in model.model.visual.parameters():
            param.requires_grad_(False)
    # Present in the checkpoint but usually not instantiated; guard anyway so that a
    # future transformers release that does materialize it does not silently start
    # training a multi-token-prediction head.
    if freeze_mtp and hasattr(model, "mtp"):
        for param in model.mtp.parameters():
            param.requires_grad_(False)


def load_policy(
    cfg: ModelConfig,
    device: str | torch.device = "cuda",
    with_value_head: bool | None = None,
) -> Policy:
    dtype = DTYPES[cfg.dtype]
    kwargs = {"dtype": dtype}
    if cfg.attn_implementation:
        kwargs["attn_implementation"] = cfg.attn_implementation

    model = Qwen3_5ForConditionalGeneration.from_pretrained(cfg.name, **kwargs)
    freeze_non_text(model, freeze_vision=cfg.freeze_vision, freeze_mtp=cfg.freeze_mtp)
    model.to(device)

    # from_pretrained returns an eval-mode model, and HF gates gradient checkpointing
    # on self.training, so without this the checkpointing below is silently inert.
    model.train()

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # Checkpointed segments have no grad-requiring input on the first block
        # otherwise, which silently disables recomputation for the embedding.
        model.enable_input_require_grads()

    tokenizer = AutoTokenizer.from_pretrained(cfg.name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Prompts are left-padded so that every sequence's generation starts at the same
    # index and the response mask is a simple suffix slice.
    tokenizer.padding_side = "left"

    use_value = cfg.value_head if with_value_head is None else with_value_head
    value_head = None
    if use_value:
        value_head = ValueHead(model.config.text_config.hidden_size).to(device)

    return Policy(
        model=model,
        tokenizer=tokenizer,
        value_head=value_head,
        autocast_dtype=DTYPES[cfg.autocast_dtype] if cfg.autocast_dtype else None,
        detach_value_head=cfg.detach_value_head,
    )


def load_reference(cfg: ModelConfig, device: str | torch.device = "cuda") -> Qwen3_5ForConditionalGeneration:
    """Frozen reference policy for the KL penalty.

    Always bf16, regardless of ModelConfig.dtype. The policy needs fp32 parameters
    so that small Adam updates survive rounding, but the reference is never updated
    -- it only produces logprobs. At 9B, fp32 here costs 33 GiB instead of 17 and
    OOMs the trainer.
    """
    model = Qwen3_5ForConditionalGeneration.from_pretrained(cfg.name, dtype=torch.bfloat16)
    model.to(device).eval()
    model.requires_grad_(False)
    return model
