"""The shared RL training loop.

REINFORCE, GRPO and PPO all run through this one loop. They differ only in which
advantage estimator runs (advantages.py) and which policy loss is applied
(losses.py), both selected from AlgoConfig. Keeping a single loop is what makes a
matched-budget comparison meaningful: the rollouts, reward, tokenization, logprob
recomputation and optimizer are literally the same code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch.nn.utils import clip_grad_norm_

from .advantages import (
    compute_group_advantages,
    gae_advantages,
    masked_whiten,
    nonzero_variance_groups,
)
from .config import Config
from .distill import (
    TeacherClient,
    reverse_kl_advantages,
    teacher_scores_to_tensor,
    topk_reverse_kl,
    topk_teacher_to_tensors,
)
from .logprobs import compute_logprobs, gather_candidate_logprobs
from .losses import compose_loss, gspo_loss, policy_gradient_loss, ppo_clipped_loss, value_loss
from .model import Policy
from .rewards import score_batch
from .rollout import Group, Rollout, rollout_stats
from .weight_sync import WeightSync


def _accumulate(metrics: dict[str, float], weight: float, totals: dict, weights: dict) -> None:
    """Weighted metric accumulation.

    NaNs are dropped rather than averaged in: value/explained_variance is NaN
    whenever a micro-batch has constant returns (a group that was all-right or
    all-wrong), which is common, and one NaN would poison the whole metric.
    """
    for key, value in metrics.items():
        if value != value:
            continue
        totals[key] = totals.get(key, 0.0) + value * weight
        weights[key] = weights.get(key, 0.0) + weight


def build_optimizer(params, cfg):
    """AdamW, optionally with 8-bit moment states.

    bitsandbytes quantizes exp_avg and exp_avg_sq to 8 bits each, taking Adam state
    from 8 bytes per parameter to 2. For a 9B model that is 68GB -> 17GB, which is
    the difference between fitting on one GPU beside an inference engine and not.
    The parameters themselves are untouched and stay fp32.
    """
    kwargs = dict(
        lr=cfg.lr,
        betas=tuple(cfg.betas),
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(params, **kwargs)
    if cfg.optimizer == "adamw8bit":
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(params, **kwargs)
    raise ValueError(f"unknown optimizer {cfg.optimizer!r} (expected adamw or adamw8bit)")


@dataclass
class TrainingBatch:
    """Flattened, padded rollouts ready for a forward pass.

    Sequences are right-padded. That is deliberate rather than incidental: the model
    has Gated DeltaNet layers whose recurrent state runs left to right, so left
    padding would feed pad tokens into the state before the real tokens. With right
    padding the contamination lands only after the response, where everything is
    masked out anyway.
    """

    input_ids: torch.Tensor  # [n, seq]
    attention_mask: torch.Tensor  # [n, seq]
    response_mask: torch.Tensor  # [n, seq-1], aligned to logprob positions
    advantages: torch.Tensor  # [n, seq-1]
    rewards: torch.Tensor  # [n]
    metrics: dict[str, float] = field(default_factory=dict)

    def to(self, device) -> TrainingBatch:
        return TrainingBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            response_mask=self.response_mask.to(device),
            advantages=self.advantages.to(device),
            rewards=self.rewards.to(device),
            metrics=self.metrics,
        )

    def __len__(self) -> int:
        return self.input_ids.shape[0]


def build_batch(
    groups: list[Group],
    advantages: torch.Tensor,
    pad_token_id: int,
    mask_truncated: bool = True,
    max_length: int | None = None,
    pad_to: int | None = None,
) -> TrainingBatch:
    """Pad a list of rollout groups into tensors.

    `advantages` is [num_groups, group_size]; it is broadcast across every response
    token of its sequence, which is what makes these sequence-level (outcome
    supervised) methods.
    """
    sequences, prompt_lens, adv_values, keep, rewards_list = [], [], [], [], []
    for g_index, group in enumerate(groups):
        for s_index, sample in enumerate(group.samples):
            ids = sample.prompt_token_ids + sample.response_token_ids
            if max_length is not None:
                ids = ids[:max_length]
            sequences.append(ids)
            prompt_lens.append(len(sample.prompt_token_ids))
            adv_values.append(advantages[g_index, s_index].item())
            rewards_list.append(getattr(sample, "reward_value", 0.0))
            # A truncated rollout never produced a final answer, so its reward says
            # nothing about the tokens that generated it.
            keep.append(not (mask_truncated and sample.truncated))

    n = len(sequences)
    seq_len = max(len(s) for s in sequences)
    if pad_to:
        # Pad every step to the same width. Ragged batches make the allocator see a
        # new shape every step, and the resulting fragmentation cost us ~50 GiB at
        # 20k context -- enough to OOM even though allocated memory was only 90 GiB.
        # The usual remedy, expandable_segments, cannot be used here because it
        # breaks CUDA IPC weight transfer. Constant shapes achieve the same end.
        seq_len = max(seq_len, pad_to)
    input_ids = torch.full((n, seq_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((n, seq_len), dtype=torch.long)
    response_mask = torch.zeros((n, seq_len - 1), dtype=torch.float32)
    adv = torch.zeros((n, seq_len - 1), dtype=torch.float32)

    for i, (ids, prompt_len) in enumerate(zip(sequences, prompt_lens)):
        length = len(ids)
        input_ids[i, :length] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, :length] = 1
        if keep[i] and length > prompt_len:
            # Logprob index t scores token t+1, so response token at absolute index
            # p lands at shifted index p-1.
            response_mask[i, prompt_len - 1 : length - 1] = 1.0
            adv[i, prompt_len - 1 : length - 1] = adv_values[i]

    return TrainingBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        response_mask=response_mask,
        advantages=adv,
        rewards=torch.tensor(rewards_list, dtype=torch.float32),
        metrics={
            "batch/sequences": float(n),
            "batch/padded_len": float(seq_len),
            "batch/masked_out_frac": 1.0 - sum(keep) / max(n, 1),
            "batch/response_tokens": float(response_mask.sum().item()),
        },
    )


class Trainer:
    def __init__(
        self,
        cfg: Config,
        policy: Policy,
        rollout: Rollout,
        weight_sync: WeightSync | None = None,
        reference=None,
        device: str = "cuda",
        example_sampler=None,
    ):
        self.cfg = cfg
        # Callable[[int], list[Example]] supplying fresh prompts for DAPO dynamic
        # sampling, which may need several generation rounds to fill a batch.
        self.example_sampler = example_sampler
        self.policy = policy
        self.rollout = rollout
        self.weight_sync = weight_sync
        self.reference = reference
        self.device = device
        self.step = 0
        self._ppo_returns = None
        self._ppo_old_values = None
        self.teacher = (
            TeacherClient(
                cfg.teacher.url,
                cfg.teacher.model,
                timeout=cfg.teacher.timeout,
                batch_size=cfg.teacher.batch_size,
            )
            if cfg.algo.name == "distill"
            else None
        )

        self.optimizer = build_optimizer(policy.trainable_parameters(), cfg.optim)
        self.scheduler = self._build_scheduler()

    def _build_scheduler(self):
        warmup, total = self.cfg.optim.warmup_steps, self.cfg.total_steps

        def lr_lambda(step: int) -> float:
            if warmup and step < warmup:
                return (step + 1) / warmup
            if self.cfg.optim.scheduler == "cosine":
                progress = (step - warmup) / max(total - warmup, 1)
                return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
            return 1.0

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    # ---- one RL step ----

    def collect(self, examples) -> tuple[list[Group], torch.Tensor, dict[str, float]]:
        """Generate rollouts and score them."""
        groups = self.rollout.generate([e.prompt for e in examples], [e.answer for e in examples])
        metrics = rollout_stats(groups)

        rewards = torch.zeros(len(groups), self.cfg.rollout.group_size)
        n_correct = n_boxed = 0
        penalties: list[float] = []
        for g_index, group in enumerate(groups):
            outcomes = score_batch(
                [s.text for s in group.samples],
                [group.answer] * len(group.samples),
                self.cfg.reward,
                [s.truncated for s in group.samples],
                [len(s.response_token_ids) for s in group.samples],
            )
            for s_index, (sample, outcome) in enumerate(zip(group.samples, outcomes)):
                rewards[g_index, s_index] = outcome.reward
                sample.reward_value = outcome.reward
                n_correct += outcome.correct
                n_boxed += outcome.has_boxed
                penalties.append(outcome.length_penalty)

        total = max(rewards.numel(), 1)
        metrics.update(
            {
                "reward/mean": rewards.mean().item(),
                "reward/accuracy": n_correct / total,
                "reward/format_rate": n_boxed / total,
                "reward/group_std_mean": rewards.std(dim=-1, unbiased=False).mean().item(),
                "reward/length_penalty": sum(penalties) / max(len(penalties), 1),
            }
        )
        return groups, rewards, metrics

    def collect_with_resampling(self, sampler) -> tuple[list[Group], torch.Tensor, dict[str, float]]:
        """DAPO dynamic sampling: keep generating until the batch is full of
        informative groups.

        Dropping degenerate groups (all samples same reward, so zero advantage)
        shrinks the batch and makes the effective step size drift with how hard the
        current prompts happen to be. DAPO instead oversamples and refills, so every
        step trains on exactly `prompts_per_step` groups that actually carry
        gradient. The cost is extra generation, bounded by `max_gen_batches`.
        """
        target = self.cfg.rollout.prompts_per_step
        per_attempt = self.cfg.algo.gen_batch_prompts or target

        kept: list[Group] = []
        kept_rewards: list[torch.Tensor] = []
        merged: dict[str, float] = {}
        attempts = 0
        prompts_used = 0

        for attempts in range(1, self.cfg.algo.max_gen_batches + 1):
            groups, rewards, metrics = self.collect(sampler(per_attempt))
            prompts_used += len(groups)
            # Metrics describe the generated population, not the surviving subset,
            # so that reward/accuracy stays an unbiased view of the policy.
            merged = metrics if not merged else {
                k: (merged.get(k, 0.0) * (attempts - 1) + v) / attempts for k, v in metrics.items()
            }
            keep = nonzero_variance_groups(rewards)
            for group, k, row in zip(groups, keep.tolist(), rewards):
                if k and len(kept) < target:
                    kept.append(group)
                    kept_rewards.append(row)
            if len(kept) >= target:
                break

        merged["sampling/gen_batches"] = float(attempts)
        merged["sampling/prompts_generated"] = float(prompts_used)
        merged["sampling/kept_frac"] = len(kept) / max(prompts_used, 1)
        merged["sampling/batch_filled"] = float(len(kept) >= target)
        if not kept:
            return [], torch.zeros(0, self.cfg.rollout.group_size), merged
        return kept, torch.stack(kept_rewards), merged

    def compute_advantages(self, groups, rewards) -> tuple[list[Group], torch.Tensor, dict]:
        metrics: dict[str, float] = {}
        if self.cfg.algo.dynamic_sampling:
            keep = nonzero_variance_groups(rewards)
            metrics["batch/groups_dropped_frac"] = 1.0 - keep.float().mean().item()
            if not keep.any():
                return [], rewards[:0], metrics
            groups = [g for g, k in zip(groups, keep.tolist()) if k]
            rewards = rewards[keep]
        advantages = compute_group_advantages(
            rewards,
            self.cfg.algo.advantage,
            normalize_std=self.cfg.algo.normalize_advantage_std,
        )
        metrics["advantage/abs_mean"] = advantages.abs().mean().item()
        return groups, advantages, metrics

    def train_step(self, examples) -> dict[str, float]:
        """One full RL iteration. This is the whole algorithm, end to end.

        1. sample `group_size` solutions per problem and grade each one
        2. turn the rewards into advantages (the part that differs per algorithm)
        3. pad into tensors and broadcast each advantage over its response tokens
        4. recompute logprobs, apply the loss, backward, optimizer step
        5. push the updated weights into the inference engine

        Step 5 is what makes the next iteration on-policy. Skip it and you are
        training against rollouts from an ever-staler policy.
        """
        cfg = self.cfg
        if cfg.algo.gen_batch_prompts and self.example_sampler is not None:
            groups, rewards, metrics = self.collect_with_resampling(self.example_sampler)
            if not groups:
                return metrics | {"loss/total": 0.0, "skipped": 1.0}
        else:
            groups, rewards, metrics = self.collect(examples)

        if cfg.algo.name == "distill" and cfg.teacher.objective == "topk":
            return metrics | self._distill_topk_step(groups)
        elif cfg.algo.name == "distill":
            batch, extra = self._prepare_distill_batch(groups, rewards)
        elif cfg.algo.advantage == "gae":
            # PPO needs a critic pass before it can assign per-token advantages.
            batch, extra = self._prepare_ppo_batch(groups, rewards)
        else:
            groups, advantages, extra = self.compute_advantages(groups, rewards)
            if not groups:
                # Dynamic sampling dropped everything: every group had identical
                # rewards, so every advantage is zero and there is no gradient to
                # be had. Skipping is correct, not a failure.
                return metrics | extra | {"loss/total": 0.0, "skipped": 1.0}
            batch = build_batch(
                groups,
                advantages,
                self.policy.tokenizer.pad_token_id,
                mask_truncated=cfg.algo.mask_truncated,
                pad_to=cfg.model.pad_to_width or None,
            ).to(self.device)
        metrics.update(extra)
        metrics.update(batch.metrics)

        # Possible when every rollout in the batch was truncated and masked out.
        if batch.response_mask.sum() == 0:
            return metrics | {"loss/total": 0.0, "skipped": 1.0}

        metrics.update(self._optimize(batch))

        if self.weight_sync is not None:
            self.weight_sync.sync()
        self.step += 1
        metrics["train/lr"] = self.scheduler.get_last_lr()[0]
        return metrics

    def _distill_topk_step(self, groups) -> dict[str, float]:
        """Differentiable reverse KL over the teacher's top-k support.

        Self-contained rather than routed through the advantage/loss machinery: this
        objective is a straight supervised loss on the student's own samples, with
        no advantage, no ratio and no clipping. It keeps the -H(pi_s) term that the
        score-function form drops, which is the term that discourages collapsing
        onto one sharp mode.
        """
        cfg = self.cfg
        batch = build_batch(
            groups,
            torch.zeros(len(groups), cfg.rollout.group_size),
            self.policy.tokenizer.pad_token_id,
            mask_truncated=False,
            pad_to=self.cfg.model.pad_to_width or None,
        ).to(self.device)

        sequences, prompt_lens, total_lens = [], [], []
        for group in groups:
            for sample in group.samples:
                ids = sample.prompt_token_ids + sample.response_token_ids
                sequences.append(ids)
                prompt_lens.append(len(sample.prompt_token_ids))
                total_lens.append(len(ids))

        scored = self.teacher.score_sequences_topk(sequences, topk=cfg.teacher.topk)
        candidates, teacher_logprobs, teacher_mask = topk_teacher_to_tensors(
            scored, prompt_lens, total_lens, batch.response_mask.shape[1],
            cfg.teacher.topk, device=self.device,
        )
        mask = batch.response_mask * teacher_mask

        totals: dict[str, float] = {}
        weights: dict[str, float] = {}
        n_updates = 0
        n = len(batch)
        minibatch = cfg.algo.minibatch_size or n
        micro = min(cfg.algo.micro_batch_size or minibatch, minibatch)

        for mb_start in range(0, n, minibatch):
            mb_stop = min(mb_start + minibatch, n)
            live = mask[mb_start:mb_stop].sum().item()
            if live == 0:
                continue
            self.optimizer.zero_grad(set_to_none=True)
            stepped = False

            for start in range(mb_start, mb_stop, micro):
                sl = slice(start, min(start + micro, mb_stop))
                sub_mask = mask[sl]
                sub_live = sub_mask.sum().item()
                if sub_live == 0:
                    continue
                student = gather_candidate_logprobs(
                    self.policy,
                    batch.input_ids[sl],
                    candidates[sl],
                    batch.attention_mask[sl],
                    temperature=cfg.rollout.temperature,
                    chunk_tokens=max(cfg.model.logprob_chunk_tokens // 2, 64),
                )
                loss, loss_metrics = topk_reverse_kl(student, teacher_logprobs[sl], sub_mask)
                # Token share, since the loss averages over tokens.
                share = sub_live / live
                (loss * share).backward()
                stepped = True
                _accumulate(loss_metrics, share, totals, weights)

            if not stepped:
                continue
            grad_norm = self._clip_and_step()
            n_updates += 1
            _accumulate({"train/grad_norm": grad_norm}, 1.0, totals, weights)

        self.scheduler.step()
        if self.weight_sync is not None:
            self.weight_sync.sync()
        self.step += 1

        out = {k: v / weights[k] for k, v in totals.items() if weights.get(k)}
        out.update(batch.metrics)
        out["train/minibatch_updates"] = float(n_updates)
        out["train/lr"] = self.scheduler.get_last_lr()[0]
        out["loss/total"] = out.get("distill/topk_reverse_kl", 0.0)
        return out

    def _prepare_distill_batch(self, groups, rewards):
        """On-policy distillation: per-token advantage from the teacher.

        Unlike the RL paths there is no group baseline. Every token gets its own
        signal, log pi_teacher(y_t) - log pi_old(y_t), which is dense supervision
        rather than one scalar spread across a whole response.
        """
        tcfg = self.cfg.teacher
        # mask_truncated is irrelevant here: a truncated rollout still contains
        # thousands of tokens the teacher can score, and there is no terminal
        # reward whose validity depends on the response having finished.
        batch = build_batch(
            groups,
            torch.zeros(len(groups), self.cfg.rollout.group_size),
            self.policy.tokenizer.pad_token_id,
            mask_truncated=False,
            pad_to=self.cfg.model.pad_to_width or None,
        ).to(self.device)

        sequences, prompt_lens, total_lens = [], [], []
        for group in groups:
            for sample in group.samples:
                ids = sample.prompt_token_ids + sample.response_token_ids
                sequences.append(ids)
                prompt_lens.append(len(sample.prompt_token_ids))
                total_lens.append(len(ids))

        scored = self.teacher.score_sequences(sequences)
        teacher = teacher_scores_to_tensor(
            scored, prompt_lens, total_lens, batch.response_mask.shape[1], device=self.device
        )

        # Only train where both sides have a value.
        mask = batch.response_mask * teacher.mask
        batch.response_mask = mask
        old_logprobs, _ = self._forward_no_grad_chunked(batch)

        advantages, metrics = reverse_kl_advantages(
            teacher.logprobs, old_logprobs, mask, clip=tcfg.advantage_clip
        )

        if tcfg.reward_mix:
            # Optional: add the group-relative correctness advantage on top, so the
            # student is pulled toward the teacher *and* toward being right.
            group_adv = compute_group_advantages(
                rewards, "grpo", normalize_std=self.cfg.algo.normalize_advantage_std
            ).to(self.device)
            broadcast = torch.zeros_like(advantages)
            for i in range(len(batch)):
                broadcast[i] = group_adv.view(-1)[i]
            advantages = advantages + tcfg.reward_mix * broadcast * mask
            metrics["distill/reward_mix"] = tcfg.reward_mix

        batch.advantages = advantages
        return batch, metrics

    def _prepare_ppo_batch(self, groups, rewards):
        """PPO: token-level GAE needs a value estimate for every response token."""
        placeholder = torch.zeros(len(groups), self.cfg.rollout.group_size)
        batch = build_batch(
            groups,
            placeholder,
            self.policy.tokenizer.pad_token_id,
            mask_truncated=self.cfg.algo.mask_truncated,
            pad_to=self.cfg.model.pad_to_width or None,
        ).to(self.device)

        _, values = self._forward_no_grad_chunked(batch, compute_values=True)

        # The correctness reward lands on the last real response token; interior
        # tokens carry nothing (optionally a KL penalty, added in the loss).
        token_rewards = torch.zeros_like(values)
        lengths = batch.response_mask.sum(dim=-1).long()
        offsets = batch.response_mask.argmax(dim=-1)
        for i in range(len(batch)):
            if lengths[i] > 0:
                token_rewards[i, offsets[i] + lengths[i] - 1] = batch.rewards[i]

        advantages, returns = gae_advantages(
            token_rewards,
            values,
            batch.response_mask,
            gamma=self.cfg.algo.gamma,
            lam=self.cfg.algo.gae_lambda,
        )
        metrics = {"advantage/raw_abs_mean": advantages.abs().mean().item()}
        if self.cfg.algo.whiten_advantages:
            advantages = masked_whiten(advantages, batch.response_mask)

        # Returns are the critic's regression target and must stay on the reward
        # scale, so they are never whitened -- only the policy-side advantages are.
        batch.advantages = advantages
        self._ppo_returns = returns
        self._ppo_old_values = values
        metrics["advantage/abs_mean"] = advantages.abs().mean().item()
        return batch, metrics

    def _forward(self, batch: TrainingBatch, compute_values: bool = False):
        return compute_logprobs(
            self.policy,
            batch.input_ids,
            batch.attention_mask,
            temperature=self.cfg.rollout.temperature,
            chunk_tokens=self.cfg.model.logprob_chunk_tokens,
            compute_entropy=self.cfg.algo.entropy_coef > 0,
            compute_values=compute_values,
        )

    @torch.no_grad()
    def _forward_no_grad_chunked(self, batch: TrainingBatch, compute_values: bool = False):
        """Full-batch forward, evaluated `minibatch_size` sequences at a time.

        The old-logprob and initial-value passes cover every sequence in the step.
        Doing that in one call would allocate activations for the whole batch at
        once (hundreds of thousands of tokens) and OOM, even though no gradients are
        needed.

        Chunk by `micro_batch_size`, not `minibatch_size`: the latter is an
        algorithm knob that GSPO deliberately sets large (32), and using it here
        would put 32 sequences x 6144 tokens through a single forward.
        """
        cfg = self.cfg.algo
        size = cfg.micro_batch_size or cfg.minibatch_size or len(batch)
        logprob_parts, value_parts = [], []
        for start in range(0, len(batch), size):
            sl = slice(start, start + size)
            out = compute_logprobs(
                self.policy,
                batch.input_ids[sl],
                batch.attention_mask[sl],
                temperature=self.cfg.rollout.temperature,
                chunk_tokens=self.cfg.model.logprob_chunk_tokens,
                compute_values=compute_values,
            )
            logprob_parts.append(out.logprobs)
            if compute_values:
                value_parts.append(out.values)
        logprobs = torch.cat(logprob_parts, dim=0)
        values = torch.cat(value_parts, dim=0) if compute_values else None
        return logprobs, values

    def _optimize(self, batch: TrainingBatch) -> dict[str, float]:
        cfg = self.cfg
        needs_values = cfg.algo.advantage == "gae"

        # Recompute old logprobs under the current policy rather than reusing vLLM's.
        # vLLM and HF agree only to ~1e-2 in bf16, and seeding the ratio with that
        # error would bias every clipped update from the very first inner epoch.
        old_logprobs, _ = self._forward_no_grad_chunked(batch)
        ref_logprobs = None
        if cfg.algo.kl_coef and self.reference is not None:
            with torch.no_grad():
                ref_logprobs = self._reference_logprobs(batch)

        totals: dict[str, float] = {}
        weights: dict[str, float] = {}
        n_updates = 0
        n = len(batch)
        minibatch = cfg.algo.minibatch_size or n
        # micro <= minibatch: forward/backward in micro-batches, one optimizer step
        # per minibatch. Equal when no accumulation is configured.
        micro = min(cfg.algo.micro_batch_size or minibatch, minibatch)

        for _ in range(max(cfg.algo.ppo_epochs, 1)):
            for mb_start in range(0, n, minibatch):
                mb_stop = min(mb_start + minibatch, n)
                # The accumulation weight has to match how the loss normalizes.
                # GSPO and sequence-level losses average over sequences, so the
                # share is a count of live sequences; token-level losses divide by
                # the token total, so the share must be a token count. Using the
                # wrong basis biases the gradient toward whichever micro-batch has
                # fewer of the other unit.
                mb_mask = batch.response_mask[mb_start:mb_stop]
                by_sequence = cfg.algo.name == "gspo" or cfg.algo.loss_normalization == "sequence"
                live = (
                    mb_mask.sum(dim=-1).gt(0).float().sum().item()
                    if by_sequence
                    else mb_mask.sum().item()
                )
                if live == 0:
                    continue

                self.optimizer.zero_grad(set_to_none=True)
                stepped = False

                for start in range(mb_start, mb_stop, micro):
                    sl = slice(start, min(start + micro, mb_stop))
                    sub = TrainingBatch(
                        input_ids=batch.input_ids[sl],
                        attention_mask=batch.attention_mask[sl],
                        response_mask=batch.response_mask[sl],
                        advantages=batch.advantages[sl],
                        rewards=batch.rewards[sl],
                    )
                    sub_live = (
                        sub.response_mask.sum(dim=-1).gt(0).float().sum().item()
                        if by_sequence
                        else sub.response_mask.sum().item()
                    )
                    if sub_live == 0:
                        continue
                    # Scale by this micro-batch's share of the minibatch so the
                    # accumulated gradient equals one backward over the whole
                    # minibatch. Without this, accumulation would inflate the
                    # effective learning rate by the number of micro-batches.
                    share = sub_live / live
                    loss_out = self._micro_loss(sub, sl, old_logprobs, ref_logprobs, needs_values)
                    (loss_out.loss * share).backward()
                    stepped = True
                    _accumulate(loss_out.metrics, share, totals, weights)

                if not stepped:
                    continue
                grad_norm = self._clip_and_step()
                n_updates += 1
                _accumulate({"train/grad_norm": grad_norm}, 1.0, totals, weights)

        self.scheduler.step()
        metrics = {k: v / weights[k] for k, v in totals.items() if weights.get(k)}
        metrics["train/minibatch_updates"] = float(n_updates)
        metrics["train/micro_batch_size"] = float(micro)
        return metrics

    def _micro_loss(self, sub, sl, old_logprobs, ref_logprobs, needs_values):
        """Policy (+ optional critic / entropy / KL) loss for one micro-batch."""
        cfg = self.cfg
        out = self._forward(sub, compute_values=needs_values)

        # Plain REINFORCE is only valid strictly on-policy; as soon as we take more
        # than one pass over a batch the ratio must be clipped.
        if cfg.algo.name == "reinforce" and cfg.algo.ppo_epochs == 1:
            policy_loss = policy_gradient_loss(
                out.logprobs, sub.advantages, sub.response_mask, cfg.algo.loss_normalization
            )
        elif cfg.algo.name == "gspo":
            # Sequence-level ratio and sequence-level clipping. No loss_normalization
            # argument: the objective is already one term per sequence.
            policy_loss = gspo_loss(
                out.logprobs,
                old_logprobs[sl],
                sub.advantages,
                sub.response_mask,
                clip_low=cfg.algo.clip_ratio_low,
                clip_high=cfg.algo.clip_ratio_high,
            )
        else:
            policy_loss = ppo_clipped_loss(
                out.logprobs,
                old_logprobs[sl],
                sub.advantages,
                sub.response_mask,
                clip_low=cfg.algo.clip_ratio_low,
                clip_high=cfg.algo.clip_ratio_high,
                normalization=cfg.algo.loss_normalization,
                clip_c=cfg.algo.clip_ratio_c,
            )

        critic = None
        if needs_values:
            critic = value_loss(
                out.values,
                self._ppo_returns[sl],
                sub.response_mask,
                old_values=self._ppo_old_values[sl],
                clip_range=cfg.algo.clip_ratio_low,
                normalization=cfg.algo.loss_normalization,
            )

        return compose_loss(
            policy_loss,
            sub.response_mask,
            entropy=out.entropy,
            entropy_coef=cfg.algo.entropy_coef,
            logprobs=out.logprobs,
            ref_logprobs=ref_logprobs[sl] if ref_logprobs is not None else None,
            kl_coef=cfg.algo.kl_coef,
            kl_estimator=cfg.algo.kl_estimator,
            value=critic,
            value_coef=cfg.algo.value_coef,
            normalization=cfg.algo.loss_normalization,
        )

    def _clip_and_step(self) -> float:
        """Clip the policy and the critic separately, then step.

        A single global clip couples them: the value head sits on hidden states with
        large magnitude, so its gradient norm reached ~25 while the policy's was
        ~1.5, and one global rescale then shrank the real policy update by more than
        an order of magnitude.
        """
        max_norm = self.cfg.optim.max_grad_norm
        policy_params = [p for p in self.policy.model.parameters() if p.requires_grad]
        grad_norm = clip_grad_norm_(policy_params, max_norm)
        if self.policy.value_head is not None:
            clip_grad_norm_(self.policy.value_head.parameters(), max_norm)
        self.optimizer.step()
        return float(grad_norm)


    def _reference_logprobs(self, batch: TrainingBatch) -> torch.Tensor:
        ref_policy = Policy(model=self.reference, tokenizer=self.policy.tokenizer)
        return compute_logprobs(
            ref_policy,
            batch.input_ids,
            batch.attention_mask,
            temperature=self.cfg.rollout.temperature,
            chunk_tokens=self.cfg.model.logprob_chunk_tokens,
        ).logprobs
