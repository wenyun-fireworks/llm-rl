"""Tests for the differentiable top-k reverse KL.

The point of this objective over the score-function form is that it keeps the
-H(pi_s) term, so it must (a) be zero only when the distributions match, and
(b) actually push probability mass toward the teacher rather than just up or down
on the sampled token.
"""

import math

import pytest
import torch

from llm_rl.distill import topk_reverse_kl, topk_teacher_to_tensors


def test_zero_when_distributions_match():
    q = torch.tensor([[[-0.5, -1.0, -2.0]]])
    kl, m = topk_reverse_kl(q.clone(), q, torch.ones(1, 1))
    assert kl.item() == pytest.approx(0.0, abs=1e-6)
    assert m["distill/argmax_agreement"] == pytest.approx(1.0)


def test_kl_is_nonnegative():
    torch.manual_seed(0)
    for _ in range(20):
        p = torch.randn(2, 3, 5)
        q = torch.randn(2, 3, 5)
        kl, _ = topk_reverse_kl(p, q, torch.ones(2, 3))
        assert kl.item() >= -1e-6


def test_matches_closed_form_on_a_known_pair():
    # Two candidates. Student 50/50, teacher 75/25 after renormalization.
    p_log = torch.tensor([[[math.log(0.5), math.log(0.5)]]])
    q_log = torch.tensor([[[math.log(0.75), math.log(0.25)]]])
    kl, _ = topk_reverse_kl(p_log, q_log, torch.ones(1, 1))
    expected = 0.5 * math.log(0.5 / 0.75) + 0.5 * math.log(0.5 / 0.25)
    assert kl.item() == pytest.approx(expected, abs=1e-6)


def test_renormalizes_over_the_candidate_set():
    """Only the top-k support is visible, so both sides must be renormalized;
    otherwise an arbitrary constant offset would change the loss."""
    p = torch.tensor([[[-1.0, -2.0]]])
    q = torch.tensor([[[-3.0, -4.0]]])
    a, _ = topk_reverse_kl(p, q, torch.ones(1, 1))
    b, _ = topk_reverse_kl(p - 7.0, q + 3.0, torch.ones(1, 1))
    assert a.item() == pytest.approx(b.item(), abs=1e-6)


def test_gradient_moves_student_toward_teacher():
    p = torch.tensor([[[0.0, 0.0]]], requires_grad=True)      # 50/50
    q = torch.tensor([[[math.log(0.9), math.log(0.1)]]])      # teacher favours token 0
    kl, _ = topk_reverse_kl(p, q, torch.ones(1, 1))
    kl.backward()
    # Descent must raise the logit of the token the teacher prefers relative to the other.
    assert p.grad[0, 0, 0] < p.grad[0, 0, 1]


def test_entropy_term_is_present():
    """The distinguishing property versus the score-function form: a student that is
    maximally peaked pays for it even when its argmax agrees with the teacher."""
    q = torch.tensor([[[math.log(0.6), math.log(0.4)]]])
    peaked = torch.tensor([[[0.0, -20.0]]])     # ~all mass on token 0
    diffuse = torch.tensor([[[math.log(0.6), math.log(0.4)]]])
    kl_peaked, m_peaked = topk_reverse_kl(peaked, q, torch.ones(1, 1))
    kl_diffuse, _ = topk_reverse_kl(diffuse, q, torch.ones(1, 1))
    assert m_peaked["distill/argmax_agreement"] == pytest.approx(1.0)
    assert kl_peaked.item() > kl_diffuse.item()
    assert m_peaked["distill/student_entropy"] < 0.05


def test_mask_excludes_positions():
    p = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
    q = torch.tensor([[[math.log(0.9), math.log(0.1)], [0.0, 0.0]]])
    both, _ = topk_reverse_kl(p, q, torch.ones(1, 2))
    first_only, _ = topk_reverse_kl(p, q, torch.tensor([[1.0, 0.0]]))
    # Position 1 has zero KL, so masking it out doubles the average.
    assert first_only.item() == pytest.approx(both.item() * 2, abs=1e-6)


def test_teacher_topk_tensors_alignment_and_ordering():
    # prompt = [a], response = [b, c]; width 2. Position index = absolute - 1.
    per_seq = [[{}, {5: -0.5, 7: -1.5, 9: -3.0}, {11: -0.2, 13: -2.0}]]
    ids, lps, mask = topk_teacher_to_tensors(per_seq, [1], [3], width=2, topk=3)
    assert mask[0].tolist() == [1.0, 1.0]
    # Candidates sorted by descending logprob.
    assert ids[0, 0].tolist() == [5, 7, 9]
    torch.testing.assert_close(lps[0, 0], torch.tensor([-0.5, -1.5, -3.0]))
    # Fewer than k candidates: padded slot must be effectively impossible.
    assert ids[0, 1, 0].item() == 11
    assert lps[0, 1, 2].item() < -1000


def test_padded_slots_do_not_affect_the_kl():
    """A position with only 2 real candidates and k=4 must give the same KL as k=2."""
    per_seq = [[{}, {5: -0.5, 7: -1.5}]]
    _, lp4, mask = topk_teacher_to_tensors(per_seq, [1], [2], width=1, topk=4)
    student4 = torch.tensor([[[-0.5, -1.5, -1e4, -1e4]]])
    kl4, _ = topk_reverse_kl(student4, lp4, mask)
    assert kl4.item() == pytest.approx(0.0, abs=1e-5)
