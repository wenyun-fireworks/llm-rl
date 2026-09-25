"""Guards the numerical choices that silently break RL fine-tuning.

These run on CPU in a second and encode two facts that are invisible until they
have already ruined a training run.
"""

import torch

from llm_rl.logprobs import gather_token_logprobs


def adam_steps(dtype, lr, steps=10, size=512):
    torch.manual_seed(0)
    param = torch.nn.Parameter(torch.randn(size, size, dtype=dtype) * 0.02)
    before = param.detach().clone().float()
    opt = torch.optim.AdamW([param], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        param.grad = torch.randn_like(param) * 0.001 + 0.01
        opt.step()
    delta = (param.detach().float() - before).abs()
    return (delta > 0).float().mean().item(), delta.mean().item()


def test_bf16_parameters_swallow_small_updates():
    """The reason ModelConfig.dtype is float32.

    bf16 has an 8-bit mantissa, so near a weight of 0.02 the smallest representable
    change is ~1.6e-4, while an lr=1e-6 Adam step moves it by ~1e-6.
    """
    moved, mean_delta = adam_steps(torch.bfloat16, lr=1e-6)
    assert moved < 0.10, f"expected bf16 to lose most 1e-6 updates, {moved:.1%} moved"
    # An order of magnitude short of the 1e-5 that ten steps should produce.
    assert mean_delta < 1e-6


def test_fp32_parameters_apply_small_updates_faithfully():
    moved, mean_delta = adam_steps(torch.float32, lr=1e-6)
    assert moved > 0.99, f"fp32 should move every weight, only {moved:.1%} moved"
    assert mean_delta == torch.tensor(1e-5).item() or abs(mean_delta - 1e-5) < 2e-6


def test_bf16_is_fine_at_a_large_learning_rate():
    """It is the ratio of update to weight that matters, not bf16 as such."""
    moved, _ = adam_steps(torch.bfloat16, lr=1e-4)
    assert moved > 0.80


def test_logprobs_are_computed_in_fp32_even_from_bf16_hidden_states():
    """log_softmax over 248k classes in bf16 costs more precision than an
    importance ratio can afford, so the chunk kernel upcasts."""
    torch.manual_seed(0)
    hidden = (torch.randn(64, 32) * 0.5).to(torch.bfloat16)
    weight = (torch.randn(1000, 32) * 0.1).to(torch.bfloat16)
    targets = torch.randint(0, 1000, (64,))

    logprobs, _ = gather_token_logprobs(hidden, weight, targets, chunk_tokens=16)
    assert logprobs.dtype == torch.float32

    reference = torch.log_softmax(
        torch.nn.functional.linear(hidden, weight).float(), dim=-1
    ).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(logprobs, reference)
