r"""On-policy distillation: reverse KL against a teacher, on the student's own rollouts.

This is not RL. There is no reward and no verifier; the supervision is the teacher's
opinion of every token the student produced. That makes it dense -- one signal per
token instead of one per response -- which is why it is far more sample-efficient
than outcome-supervised RL. The price is a hard ceiling: the student cannot exceed
the teacher, whereas RL can in principle exceed any teacher.

The objective is the reverse (mode-seeking) KL, evaluated on trajectories sampled
from the student:

    L = E_{y ~ pi_student} [ sum_t KL( pi_student(.|s_t) || pi_teacher(.|s_t) ) ]

Computing that literally would need the teacher's full 248,320-way distribution at
every position. It does not: the gradient reduces to a policy gradient whose
per-token reward is the teacher's log-probability advantage.

    grad KL = E_{v~pi_s}[ (log pi_s(v) - log pi_t(v)) grad log pi_s(v) ]
            + E_{v~pi_s}[ grad log pi_s(v) ]

The second term vanishes because sum_v pi_s grad log pi_s = grad sum_v pi_s = 0.
So descending the KL is ascending a policy gradient with

    A_t = log pi_teacher(y_t) - log pi_old(y_t)

which needs only the teacher's logprob **at the token the student actually sampled**.
That is one cheap prefill-only call per rollout (vLLM's `prompt_logprobs`), no top-k
approximation, and it plugs straight into the existing per-token advantage path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import requests
import torch


@dataclass
class TeacherScores:
    """Teacher logprob for each response token, padded to [n, seq-1].

    Aligned exactly like `logprobs.compute_logprobs`: index t holds the logprob of
    absolute token t+1, so a response token at absolute index p sits at index p-1.
    """

    logprobs: torch.Tensor
    mask: torch.Tensor  # 1 where the teacher actually returned a value


class TeacherClient:
    """Scores student tokens with a teacher served by vLLM.

    Requires the teacher to share the student's tokenizer. Every Qwen3.5 checkpoint
    uses the same 248,320-token vocabulary, so any of them works as a teacher for
    any other; a Llama or DeepSeek teacher would not, because token boundaries would
    not line up and per-token alignment is the whole premise.
    """

    def __init__(self, url: str, model: str, timeout: float = 600.0, batch_size: int = 4):
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.batch_size = batch_size

    def score_sequences(self, token_id_lists: list[list[int]], attempts: int = 4) -> list[list[float]]:
        """Per-token teacher logprobs for each full sequence (prompt + response).

        Sends token ids rather than text so that no re-tokenization can shift the
        alignment. `max_tokens=0` makes this prefill-only, which is much cheaper for
        the server than generating.
        """
        out: list[list[float]] = []
        for start in range(0, len(token_id_lists), self.batch_size):
            chunk = token_id_lists[start : start + self.batch_size]
            body = self._post(
                {
                    "model": self.model,
                    "prompt": chunk,
                    # max_tokens=0 is rejected on its own ("must be at least 1"),
                    # but is accepted together with echo=True -- which is what we
                    # want anyway: pure prefill, not one wasted generated token.
                    "max_tokens": 0,
                    "echo": True,
                    "temperature": 0.0,
                    "prompt_logprobs": 0,
                },
                attempts,
            )
            # Responses can come back out of order; index carries the position.
            ordered = sorted(body["choices"], key=lambda c: c["index"])
            for choice, ids in zip(ordered, chunk):
                out.append(self._extract(choice, ids))
        return out

    def score_sequences_topk(
        self, token_id_lists: list[list[int]], topk: int = 20, attempts: int = 4
    ) -> list[list[dict[int, float]]]:
        """Teacher's top-k distribution at every position, not just the sampled token.

        Needed for the differentiable form of the reverse KL. The single-token
        version supports only the score-function estimator, which drops the entropy
        term of KL(pi_s || pi_t) and so has nothing preventing mode collapse.
        """
        out: list[list[dict[int, float]]] = []
        for start in range(0, len(token_id_lists), self.batch_size):
            chunk = token_id_lists[start : start + self.batch_size]
            body = self._post(
                {
                    "model": self.model,
                    "prompt": chunk,
                    "max_tokens": 0,
                    "echo": True,
                    "temperature": 0.0,
                    "prompt_logprobs": topk,
                },
                attempts,
            )
            for choice in sorted(body["choices"], key=lambda c: c["index"]):
                raw = choice.get("prompt_logprobs")
                if raw is None:
                    raise RuntimeError("teacher returned no prompt_logprobs")
                positions: list[dict[int, float]] = []
                for entry in raw:
                    if entry is None:
                        positions.append({})
                        continue
                    positions.append(
                        {
                            int(tok): float(v["logprob"] if isinstance(v, dict) else v)
                            for tok, v in entry.items()
                        }
                    )
                out.append(positions)
        return out

    def _post(self, payload: dict, attempts: int) -> dict:
        last: Exception | None = None
        for attempt in range(attempts):
            try:
                response = requests.post(
                    f"{self.url}/v1/completions", json=payload, timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
            except requests.RequestException as error:
                last = error
                if attempt < attempts - 1:
                    time.sleep(min(2**attempt * 5, 60))
        raise RuntimeError(f"teacher scoring failed after {attempts} attempts: {last}")

    @staticmethod
    def _extract(choice: dict, token_ids: list[int]) -> list[float]:
        """Pull out the logprob of each actual token.

        vLLM returns `prompt_logprobs` as one entry per prompt token, each a dict
        keyed by token id. Position 0 is None because the first token has no
        context. We take the entry for the token that is actually there.
        """
        raw = choice.get("prompt_logprobs")
        if raw is None:
            raise RuntimeError(
                "teacher returned no prompt_logprobs; the server must support "
                "`prompt_logprobs` on /v1/completions"
            )
        values: list[float] = []
        for position, entry in enumerate(raw):
            if entry is None or position >= len(token_ids):
                values.append(float("nan"))
                continue
            token = token_ids[position]
            item = entry.get(str(token), entry.get(token))
            if item is None:
                values.append(float("nan"))
            else:
                values.append(float(item["logprob"] if isinstance(item, dict) else item))
        return values


def teacher_scores_to_tensor(
    per_sequence: list[list[float]],
    prompt_lens: list[int],
    total_lens: list[int],
    width: int,
    device="cpu",
) -> TeacherScores:
    """Lay per-sequence teacher logprobs into a [n, width] tensor.

    `width` is seq-1, matching the student's logprob tensor. A response token at
    absolute index p is scored at index p-1, the same shift used for
    `response_mask`, so the two line up element for element.
    """
    n = len(per_sequence)
    logprobs = torch.zeros(n, width, dtype=torch.float32)
    mask = torch.zeros(n, width, dtype=torch.float32)
    for i, (values, prompt_len, total) in enumerate(zip(per_sequence, prompt_lens, total_lens)):
        for absolute in range(prompt_len, min(total, width + 1)):
            if absolute >= len(values):
                break
            value = values[absolute]
            if value != value:  # NaN: teacher gave us nothing here
                continue
            logprobs[i, absolute - 1] = value
            mask[i, absolute - 1] = 1.0
    return TeacherScores(logprobs=logprobs.to(device), mask=mask.to(device))


def topk_teacher_to_tensors(
    per_sequence: list[list[dict[int, float]]],
    prompt_lens: list[int],
    total_lens: list[int],
    width: int,
    topk: int,
    device="cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Lay the teacher's top-k distributions into dense tensors.

    Returns (candidate_ids [n, width, k], teacher_logprobs [n, width, k],
    mask [n, width]). Positions with fewer than k candidates are padded with the
    first candidate and a very negative logprob so they contribute nothing.
    """
    n = len(per_sequence)
    ids = torch.zeros(n, width, topk, dtype=torch.long)
    logprobs = torch.full((n, width, topk), -1e4, dtype=torch.float32)
    mask = torch.zeros(n, width, dtype=torch.float32)

    for i, (positions, prompt_len, total) in enumerate(zip(per_sequence, prompt_lens, total_lens)):
        for absolute in range(prompt_len, min(total, width + 1)):
            if absolute >= len(positions):
                break
            entry = positions[absolute]
            if not entry:
                continue
            items = sorted(entry.items(), key=lambda kv: -kv[1])[:topk]
            index = absolute - 1
            for j, (token, logprob) in enumerate(items):
                ids[i, index, j] = token
                logprobs[i, index, j] = logprob
            mask[i, index] = 1.0
    return ids.to(device), logprobs.to(device), mask.to(device)


def topk_reverse_kl(
    student_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    r"""Differentiable reverse KL over the teacher's top-k support.

    Both sides are renormalized over the k candidates, then

        KL(p || q) = sum_k p (log p - log q)

    Unlike the score-function form this differentiates through the student's
    probabilities directly, so it keeps the -H(pi_s) term. That term is what stops
    the student collapsing onto a single sharp mode that the teacher happens to
    agree with.

    student_logprobs / teacher_logprobs: [n, width, k]; mask: [n, width].
    """
    # Renormalize over the candidate set, since we only see the top k.
    p_log = student_logprobs - student_logprobs.logsumexp(dim=-1, keepdim=True)
    q_log = teacher_logprobs - teacher_logprobs.logsumexp(dim=-1, keepdim=True)
    p = p_log.exp()

    per_token = (p * (p_log - q_log)).sum(dim=-1)
    mask = mask.to(per_token.dtype)
    count = mask.sum().clamp(min=1.0)
    loss = (per_token * mask).sum() / count

    with torch.no_grad():
        entropy = -(p * p_log).sum(dim=-1)
        agreement = (p_log.argmax(dim=-1) == q_log.argmax(dim=-1)).to(mask.dtype)
        metrics = {
            "distill/topk_reverse_kl": loss.item(),
            "distill/student_entropy": ((entropy * mask).sum() / count).item(),
            "distill/argmax_agreement": ((agreement * mask).sum() / count).item(),
            "distill/scored_frac": (mask.sum() / mask.numel()).item(),
        }
    return loss, metrics


def reverse_kl_advantages(
    teacher_logprobs: torch.Tensor,
    student_logprobs: torch.Tensor,
    mask: torch.Tensor,
    clip: float | None = 10.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Per-token advantage for reverse-KL distillation: log pi_teacher - log pi_old.

    Positive where the teacher considers the token more likely than the student did,
    i.e. exactly where the student should move toward the teacher.

    `clip` bounds the magnitude. Without it a single token the teacher finds nearly
    impossible (logprob -20 or worse, which happens on formatting noise) would
    dominate the whole batch's gradient.
    """
    advantages = (teacher_logprobs - student_logprobs) * mask
    if clip is not None:
        advantages = advantages.clamp(-clip, clip)
    advantages = advantages * mask

    count = mask.sum().clamp(min=1.0)
    # Reverse KL per token is -(advantage) in expectation; report it positive.
    kl = -(advantages.sum() / count)
    metrics = {
        "distill/reverse_kl": kl.item(),
        "distill/advantage_abs_mean": (advantages.abs().sum() / count).item(),
        "distill/teacher_logprob_mean": ((teacher_logprobs * mask).sum() / count).item(),
        "distill/teacher_better_frac": (((advantages > 0).float() * mask).sum() / count).item(),
        "distill/scored_frac": (mask.sum() / mask.numel()).item(),
    }
    return advantages, metrics
