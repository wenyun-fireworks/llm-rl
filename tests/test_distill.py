"""Tests for on-policy distillation.

The two things that can silently ruin this method are token misalignment (teacher
logprobs offset by one against the student's) and an unbounded advantage from a
token the teacher considers impossible. Both are covered here.
"""

import math

import pytest
import torch

from llm_rl.config import Config
from llm_rl.distill import TeacherClient, reverse_kl_advantages, teacher_scores_to_tensor


def test_extract_picks_the_logprob_of_the_actual_token():
    ids = [10, 11, 12]
    choice = {
        "prompt_logprobs": [
            None,                                   # first token has no context
            {"11": {"logprob": -0.5}},
            {"12": {"logprob": -1.5}},
        ]
    }
    assert TeacherClient._extract(choice, ids) == pytest.approx([float("nan"), -0.5, -1.5], nan_ok=True)


def test_extract_tolerates_int_keys_and_bare_floats():
    choice = {"prompt_logprobs": [None, {11: -0.25}]}
    values = TeacherClient._extract(choice, [10, 11])
    assert values[1] == pytest.approx(-0.25)


def test_extract_rejects_a_server_that_returns_nothing():
    with pytest.raises(RuntimeError, match="prompt_logprobs"):
        TeacherClient._extract({}, [1, 2])


def test_alignment_matches_the_student_logprob_shift():
    """A response token at absolute index p must land at index p-1, the same shift
    build_batch uses for response_mask. Off by one here and the student is trained
    against the teacher's opinion of the *previous* token."""
    # prompt = [a, b], response = [c, d]; total 4 tokens, width = 3
    per_seq = [[float("nan"), -0.1, -0.2, -0.3]]   # index = absolute position
    scores = teacher_scores_to_tensor(per_seq, prompt_lens=[2], total_lens=[4], width=3)
    # Response tokens are absolute 2 and 3 -> indices 1 and 2.
    assert scores.mask[0].tolist() == [0.0, 1.0, 1.0]
    torch.testing.assert_close(scores.logprobs[0], torch.tensor([0.0, -0.2, -0.3]))


def test_nan_positions_are_masked_out_not_zeroed_in():
    per_seq = [[float("nan"), -0.1, float("nan"), -0.3]]
    scores = teacher_scores_to_tensor(per_seq, prompt_lens=[1], total_lens=[4], width=3)
    # absolute 1,2,3 -> indices 0,1,2; absolute 2 was NaN so index 1 is masked off.
    assert scores.mask[0].tolist() == [1.0, 0.0, 1.0]


def test_ragged_batch_padding():
    per_seq = [
        [float("nan"), -1.0, -1.0, -1.0, -1.0],
        [float("nan"), -2.0],
    ]
    scores = teacher_scores_to_tensor(per_seq, [1, 1], [5, 2], width=4)
    assert scores.mask[0].sum() == 4.0
    assert scores.mask[1].sum() == 1.0
    assert scores.logprobs.shape == (2, 4)


def test_advantage_is_teacher_minus_student():
    teacher = torch.tensor([[-1.0, -3.0]])
    student = torch.tensor([[-2.0, -2.0]])
    mask = torch.ones(1, 2)
    adv, m = reverse_kl_advantages(teacher, student, mask, clip=None)
    # Token 0: teacher likes it more (+1). Token 1: teacher likes it less (-1).
    torch.testing.assert_close(adv, torch.tensor([[1.0, -1.0]]))
    assert m["distill/teacher_better_frac"] == pytest.approx(0.5)
    # Reverse KL is the negative mean advantage, and here they cancel.
    assert m["distill/reverse_kl"] == pytest.approx(0.0)


def test_zero_kl_when_student_matches_teacher():
    lp = torch.tensor([[-1.0, -2.0, -3.0]])
    adv, m = reverse_kl_advantages(lp, lp.clone(), torch.ones(1, 3))
    torch.testing.assert_close(adv, torch.zeros(1, 3))
    assert m["distill/reverse_kl"] == pytest.approx(0.0)


def test_clip_bounds_a_catastrophically_unlikely_token():
    """A token the teacher assigns -60 would otherwise swamp the batch."""
    teacher = torch.tensor([[-60.0, -1.0]])
    student = torch.tensor([[-1.0, -1.0]])
    adv, _ = reverse_kl_advantages(teacher, student, torch.ones(1, 2), clip=10.0)
    assert adv[0, 0].item() == pytest.approx(-10.0)
    unclipped, _ = reverse_kl_advantages(teacher, student, torch.ones(1, 2), clip=None)
    assert unclipped[0, 0].item() == pytest.approx(-59.0)


def test_mask_zeroes_advantage_outside_the_response():
    teacher = torch.tensor([[-1.0, -5.0]])
    student = torch.tensor([[-2.0, -2.0]])
    mask = torch.tensor([[1.0, 0.0]])
    adv, m = reverse_kl_advantages(teacher, student, mask)
    assert adv[0, 1].item() == 0.0
    # The metric averages over masked-in tokens only.
    assert m["distill/reverse_kl"] == pytest.approx(-1.0)
    assert m["distill/scored_frac"] == pytest.approx(0.5)


def test_positive_advantage_pushes_student_toward_teacher():
    """Sanity check on the sign convention against the policy-gradient loss: where
    the teacher likes a token more than the student did, the update must raise the
    student's logprob for it."""
    from llm_rl.losses import policy_gradient_loss

    teacher = torch.tensor([[-1.0]])
    student_old = torch.tensor([[-3.0]])
    adv, _ = reverse_kl_advantages(teacher, student_old, torch.ones(1, 1))
    assert adv.item() > 0

    logp = torch.tensor([[-3.0]], requires_grad=True)
    policy_gradient_loss(logp, adv, torch.ones(1, 1)).loss.backward()
    assert logp.grad.item() < 0   # descent raises logp


def test_distill_config():
    cfg = Config.from_yaml("configs/distill.yaml")
    assert cfg.algo.name == "distill"
    assert cfg.teacher.model.startswith("Qwen/Qwen3.5-")   # tokenizer must match
    assert cfg.teacher.advantage_clip is not None
    # Distillation has no group baseline, so group_size is only about throughput.
    assert cfg.model.name == "Qwen/Qwen3.5-9B-Base"
