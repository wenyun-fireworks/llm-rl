import pytest
import torch
import torch.nn.functional as F

from llm_rl.logprobs import gather_token_logprobs, kl_penalty, masked_mean


def naive_logprobs(hidden, weight, targets, temperature=1.0):
    logits = F.linear(hidden, weight).float() / temperature
    logprobs = torch.log_softmax(logits, dim=-1)
    return logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


@pytest.fixture
def toy():
    torch.manual_seed(0)
    n_tokens, hidden_size, vocab = 37, 16, 129
    hidden = torch.randn(n_tokens, hidden_size, dtype=torch.float32)
    weight = torch.randn(vocab, hidden_size, dtype=torch.float32) * 0.1
    targets = torch.randint(0, vocab, (n_tokens,))
    return hidden, weight, targets


@pytest.mark.parametrize("chunk", [1, 5, 16, 37, 1000])
def test_chunking_matches_naive_regardless_of_chunk_size(toy, chunk):
    hidden, weight, targets = toy
    got, _ = gather_token_logprobs(hidden, weight, targets, chunk_tokens=chunk)
    torch.testing.assert_close(got, naive_logprobs(hidden, weight, targets))


def test_chunk_size_does_not_change_gradients(toy):
    hidden, weight, targets = toy
    grads = []
    for chunk in (4, 37):
        h = hidden.clone().requires_grad_(True)
        w = weight.clone().requires_grad_(True)
        lp, _ = gather_token_logprobs(h, w, targets, chunk_tokens=chunk)
        lp.sum().backward()
        grads.append((h.grad, w.grad))
    torch.testing.assert_close(grads[0][0], grads[1][0])
    torch.testing.assert_close(grads[0][1], grads[1][1])


def test_gradients_match_naive(toy):
    hidden, weight, targets = toy
    h1 = hidden.clone().requires_grad_(True)
    lp, _ = gather_token_logprobs(h1, weight, targets, chunk_tokens=8)
    lp.sum().backward()

    h2 = hidden.clone().requires_grad_(True)
    naive_logprobs(h2, weight, targets).sum().backward()
    torch.testing.assert_close(h1.grad, h2.grad)


def test_temperature_scales_logits_not_logprobs_directly(toy):
    hidden, weight, targets = toy
    got, _ = gather_token_logprobs(hidden, weight, targets, temperature=0.5, chunk_tokens=8)
    torch.testing.assert_close(got, naive_logprobs(hidden, weight, targets, temperature=0.5))
    # Lower temperature sharpens, so the argmax token's logprob must rise.
    argmax = F.linear(hidden, weight).argmax(-1)
    hot, _ = gather_token_logprobs(hidden, weight, argmax, temperature=1.0)
    cold, _ = gather_token_logprobs(hidden, weight, argmax, temperature=0.5)
    assert (cold >= hot - 1e-6).all()


def test_entropy_matches_full_distribution(toy):
    hidden, weight, targets = toy
    _, entropy = gather_token_logprobs(hidden, weight, targets, chunk_tokens=8, compute_entropy=True)
    logprobs = torch.log_softmax(F.linear(hidden, weight).float(), dim=-1)
    torch.testing.assert_close(entropy, -(logprobs.exp() * logprobs).sum(-1))


def test_entropy_is_none_when_not_requested(toy):
    hidden, weight, targets = toy
    _, entropy = gather_token_logprobs(hidden, weight, targets)
    assert entropy is None


def test_shape_mismatch_rejected(toy):
    hidden, weight, targets = toy
    with pytest.raises(ValueError):
        gather_token_logprobs(hidden, weight, targets[:-1])


def test_kl_k3_is_nonnegative_and_zero_at_identity():
    torch.manual_seed(0)
    logp = torch.randn(64) - 2.0
    ref = torch.randn(64) - 2.0
    assert (kl_penalty(logp, ref, "k3") >= -1e-6).all()
    torch.testing.assert_close(kl_penalty(logp, logp, "k3"), torch.zeros_like(logp), atol=1e-6, rtol=0)
    # The naive k1 estimator is signed, which is exactly why we default to k3.
    assert (kl_penalty(logp, ref, "k1") < 0).any()


def test_kl_unknown_estimator():
    with pytest.raises(ValueError):
        kl_penalty(torch.zeros(3), torch.zeros(3), "k9")


def test_masked_mean():
    values = torch.tensor([[1.0, 2.0, 100.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    torch.testing.assert_close(masked_mean(values, mask), torch.tensor(1.5))
    torch.testing.assert_close(masked_mean(values, mask, dim=-1), torch.tensor([1.5]))
    # All-padding rows must not divide by zero.
    assert torch.isfinite(masked_mean(values, torch.zeros_like(mask))).all()
