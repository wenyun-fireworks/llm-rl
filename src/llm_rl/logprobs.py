"""Per-token log-probabilities under the current policy.

This is the memory bottleneck of the whole pipeline. Qwen3.5's vocab is 248,320, so
the logits for a single 4096-token sequence are 248320 * 4096 * 2 bytes ~= 2.0 GB in
bf16, and autograd would keep them alive until backward. We therefore never
materialize logits for a whole batch: the LM head is applied to a slice of tokens at
a time, inside a checkpointed region so the slice's logits are recomputed during
backward instead of stored.

Peak logit memory is O(chunk_tokens * vocab) rather than O(batch * seq * vocab).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .model import Policy


@dataclass
class LogprobOutput:
    logprobs: torch.Tensor  # [batch, seq-1] logprob of each next token
    entropy: torch.Tensor | None = None  # [batch, seq-1]
    values: torch.Tensor | None = None  # [batch, seq-1] critic output, PPO only


def _logprob_chunk(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    temperature: float,
    want_entropy: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-softmax + gather for one slice of flattened tokens.

    Runs in fp32: log_softmax over 248k classes in bf16 loses enough precision to
    matter for an importance ratio that is supposed to start at exactly 1.0.
    """
    logits = F.linear(hidden, weight).float()
    if temperature != 1.0:
        logits = logits / temperature
    logprobs = torch.log_softmax(logits, dim=-1)
    token_logprobs = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    if not want_entropy:
        return token_logprobs, token_logprobs.new_zeros(())
    entropy = -(logprobs.exp() * logprobs).sum(-1)
    return token_logprobs, entropy


def gather_token_logprobs(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    temperature: float = 1.0,
    chunk_tokens: int = 512,
    compute_entropy: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Chunked logprobs for `targets` given hidden states.

    hidden:  [n_tokens, hidden]  (already flattened and aligned to targets)
    targets: [n_tokens]
    """
    if hidden.shape[0] != targets.shape[0]:
        raise ValueError(f"hidden {tuple(hidden.shape)} and targets {tuple(targets.shape)} disagree")

    use_checkpoint = torch.is_grad_enabled() and hidden.requires_grad
    logprob_parts, entropy_parts = [], []
    for start in range(0, hidden.shape[0], chunk_tokens):
        h = hidden[start : start + chunk_tokens]
        t = targets[start : start + chunk_tokens]
        if use_checkpoint:
            lp, ent = checkpoint(
                _logprob_chunk, h, weight, t, temperature, compute_entropy, use_reentrant=False
            )
        else:
            lp, ent = _logprob_chunk(h, weight, t, temperature, compute_entropy)
        logprob_parts.append(lp)
        if compute_entropy:
            entropy_parts.append(ent)

    logprobs = torch.cat(logprob_parts, dim=0)
    entropy = torch.cat(entropy_parts, dim=0) if compute_entropy else None
    return logprobs, entropy


def compute_logprobs(
    policy: Policy,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    temperature: float = 1.0,
    chunk_tokens: int = 512,
    compute_entropy: bool = False,
    compute_values: bool = False,
) -> LogprobOutput:
    """Logprob of every next token in `input_ids` under `policy`.

    Returns tensors of width seq-1, aligned so that index t is the logprob of
    ``input_ids[:, t + 1]``. Callers mask this down to response tokens.
    """
    # bf16 autocast for the backbone and the LM head projection; the log-softmax
    # inside gather_token_logprobs still upcasts to fp32 where precision matters.
    with policy.autocast():
        hidden = policy.hidden_states(input_ids, attention_mask)
        # Position t predicts token t+1.
        hidden = hidden[:, :-1, :]
        targets = input_ids[:, 1:]
        batch, seq_minus_one, hidden_size = hidden.shape

        logprobs, entropy = gather_token_logprobs(
            hidden.reshape(-1, hidden_size),
            policy.lm_head.weight,
            targets.reshape(-1),
            temperature=temperature,
            chunk_tokens=chunk_tokens,
            compute_entropy=compute_entropy,
        )
    logprobs = logprobs.view(batch, seq_minus_one)
    if entropy is not None:
        entropy = entropy.view(batch, seq_minus_one)

    values = None
    if compute_values:
        if policy.value_head is None:
            raise ValueError("compute_values=True but the policy has no value head")
        # See Policy.detach_value_head: keeps a badly-fit critic from dragging the
        # shared backbone around.
        values = policy.value_head(hidden.detach() if policy.detach_value_head else hidden)

    return LogprobOutput(logprobs=logprobs, entropy=entropy, values=values)


def _candidate_chunk(
    hidden: torch.Tensor, weight: torch.Tensor, candidates: torch.Tensor, temperature: float
) -> torch.Tensor:
    logits = F.linear(hidden, weight).float()
    if temperature != 1.0:
        logits = logits / temperature
    logprobs = torch.log_softmax(logits, dim=-1)
    return logprobs.gather(-1, candidates)


def gather_candidate_logprobs(
    policy: Policy,
    input_ids: torch.Tensor,
    candidates: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    temperature: float = 1.0,
    chunk_tokens: int = 256,
) -> torch.Tensor:
    """Student logprobs for k candidate tokens at every position.

    `candidates` is [batch, seq-1, k], typically the teacher's top-k token ids.
    Returns [batch, seq-1, k], differentiable, for the direct reverse-KL objective.

    Same chunking and checkpointing as `compute_logprobs`, and for the same reason:
    the intermediate is still (chunk x 248320) logits regardless of how few
    candidates we ultimately keep. The chunk default is smaller here because each
    position now carries k values instead of one.
    """
    with policy.autocast():
        hidden = policy.hidden_states(input_ids, attention_mask)[:, :-1, :]
        batch, width, hidden_size = hidden.shape
        k = candidates.shape[-1]

        flat_hidden = hidden.reshape(-1, hidden_size)
        flat_candidates = candidates.reshape(-1, k)
        use_checkpoint = torch.is_grad_enabled() and flat_hidden.requires_grad

        parts = []
        for start in range(0, flat_hidden.shape[0], chunk_tokens):
            h = flat_hidden[start : start + chunk_tokens]
            c = flat_candidates[start : start + chunk_tokens]
            if use_checkpoint:
                parts.append(
                    checkpoint(
                        _candidate_chunk, h, policy.lm_head.weight, c, temperature,
                        use_reentrant=False,
                    )
                )
            else:
                parts.append(_candidate_chunk(h, policy.lm_head.weight, c, temperature))
    return torch.cat(parts, dim=0).view(batch, width, k)


def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    mask = mask.to(values.dtype)
    if dim is None:
        return (values * mask).sum() / mask.sum().clamp(min=1.0)
    return (values * mask).sum(dim) / mask.sum(dim).clamp(min=1.0)


def kl_penalty(
    logprobs: torch.Tensor, ref_logprobs: torch.Tensor, estimator: str = "k3"
) -> torch.Tensor:
    """Per-token KL(policy || reference) estimate from single samples.

    k3 is Schulman's low-variance unbiased estimator, ``exp(-d) - 1 + d`` where
    ``d = logp - logp_ref``. Unlike the naive ``d``, it is non-negative, which keeps
    the penalty from ever paying the policy to move away from the reference.
    """
    diff = logprobs - ref_logprobs
    if estimator == "k1":
        return diff
    if estimator == "k2":
        return 0.5 * diff.square()
    if estimator == "k3":
        return torch.expm1(-diff) + diff
    raise ValueError(f"unknown kl estimator {estimator!r} (expected k1, k2 or k3)")
