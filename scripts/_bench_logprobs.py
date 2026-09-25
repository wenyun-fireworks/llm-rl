"""Honest chunked-vs-unchunked benchmark: warm up every (shape, chunk) pair first."""

import time

import torch

from llm_rl.config import ModelConfig
from llm_rl.logprobs import compute_logprobs
from llm_rl.model import load_policy

torch.manual_seed(0)
policy = load_policy(ModelConfig(gradient_checkpointing=True), device="cuda")


def measure(batch, seq, chunk, iters=3):
    input_ids = torch.randint(0, 100_000, (batch, seq), device="cuda")
    attention_mask = torch.ones_like(input_ids)

    def once():
        policy.model.zero_grad(set_to_none=True)
        out = compute_logprobs(policy, input_ids, attention_mask, chunk_tokens=chunk)
        out.logprobs.mean().backward()
        torch.cuda.synchronize()

    once()  # warm up kernels for this exact shape
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    for _ in range(iters):
        once()
    elapsed = (time.time() - start) / iters
    return torch.cuda.max_memory_allocated() / 2**30, elapsed


print(f"{'shape':>12s} {'tokens':>7s} | {'chunked GiB':>11s} {'s':>6s} | {'unchunked GiB':>13s} {'s':>6s}")
for batch, seq in [(4, 1024), (8, 2048), (16, 2048)]:
    mem_c, t_c = measure(batch, seq, 512)
    try:
        mem_u, t_u = measure(batch, seq, batch * seq)
        unchunked = f"{mem_u:13.1f} {t_u:6.2f}"
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        unchunked = f"{'OOM':>13s} {'-':>6s}"
    print(f"{f'[{batch},{seq}]':>12s} {batch*seq:7d} | {mem_c:11.1f} {t_c:6.2f} | {unchunked}")
