"""Verify vLLM and HF agree on token ids and per-token logprobs.

A mismatch here would silently corrupt every PPO/GRPO importance ratio, so this runs
before any algorithm is trusted.
"""

import os

import requests
import torch

from llm_rl.config import ModelConfig, RolloutConfig
from llm_rl.data import Example
from llm_rl.logprobs import compute_logprobs
from llm_rl.model import load_policy
from llm_rl.rollout import Rollout, rollout_stats

URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8001")
MODEL = "Qwen/Qwen3.5-0.8B-Base"

examples = [
    Example(problem="What is 17 times 24?", answer="408"),
    Example(problem="If 3x + 7 = 22, what is x?", answer="5"),
]
prompts = [e.prompt for e in examples]
answers = [e.answer for e in examples]

cfg = RolloutConfig(group_size=2, max_new_tokens=64, temperature=1.0, seed=1234)
rollout = Rollout(URL, MODEL, cfg)
groups = rollout.generate(prompts, answers)

print("=== rollout ===")
for key, value in rollout_stats(groups).items():
    print(f"  {key}: {value}")
sample = groups[0].samples[0]
print(f"  prompt tokens: {len(sample.prompt_token_ids)}, response tokens: {len(sample.response_token_ids)}")
print(f"  finish_reason: {sample.finish_reason}")
print(f"  text: {sample.text[:160]!r}")

# Ask vLLM for the logprob of each sampled token so we have something to compare to.
payload = {
    "model": MODEL,
    "prompt": prompts[0],
    "n": 1,
    "max_tokens": 48,
    "temperature": 1.0,
    "seed": 7,
    "logprobs": 0,
    "return_token_ids": True,
}
choice = requests.post(f"{URL}/v1/completions", json=payload, timeout=600).json()["choices"][0]
prompt_ids = choice["prompt_token_ids"]
response_ids = choice["token_ids"]
vllm_logprobs = torch.tensor(choice["logprobs"]["token_logprobs"], dtype=torch.float32)

print("\n=== tokenizer round-trip ===")
policy = load_policy(ModelConfig(gradient_checkpointing=False), device="cuda")
hf_prompt_ids = policy.tokenizer(prompts[0], add_special_tokens=False)["input_ids"]
print(f"  vLLM prompt ids: {len(prompt_ids)}, HF tokenizer ids: {len(hf_prompt_ids)}, equal={prompt_ids == hf_prompt_ids}")

full_ids = torch.tensor([prompt_ids + response_ids], device="cuda")
with torch.no_grad():
    out = compute_logprobs(policy, full_ids, torch.ones_like(full_ids), chunk_tokens=512)

# compute_logprobs index t holds the logprob of token t+1, so response token at
# absolute index P+i is scored at shifted index P+i-1.
start = len(prompt_ids) - 1
hf_logprobs = out.logprobs[0, start : start + len(response_ids)].float().cpu()

diff = (hf_logprobs - vllm_logprobs).abs()
print("\n=== logprob parity (HF recompute vs vLLM sampling) ===")
print(f"  tokens compared: {len(diff)}")
print(f"  max abs diff:  {diff.max():.5f}")
print(f"  mean abs diff: {diff.mean():.5f}")
print(f"  correlation:   {torch.corrcoef(torch.stack([hf_logprobs, vllm_logprobs]))[0,1]:.6f}")
ratio = (hf_logprobs - vllm_logprobs).exp()
print(f"  implied importance ratio: mean={ratio.mean():.4f} min={ratio.min():.4f} max={ratio.max():.4f}")
print("\n  first 8 tokens (HF vs vLLM):")
for i in range(min(8, len(diff))):
    print(f"    {policy.tokenizer.decode([response_ids[i]])!r:>14}  {hf_logprobs[i]:+.4f}  {vllm_logprobs[i]:+.4f}  d={diff[i]:.5f}")
