"""Verify CUDA IPC weight sync from the trainer into a colocated vLLM engine.

Three greedy generations from the same prompt and seed:
  1. baseline, engine still on its checkpoint weights
  2. after syncing the trainer's *unmodified* weights  -> must be identical
  3. after perturbing the trainer's weights and syncing -> must differ

Step 2 catches a sync that silently corrupts or drops parameters; step 3 catches a
sync that is a no-op. The trainer must sit on the same physical GPU as the engine,
since CUDA IPC handles are looked up by GPU UUID.
"""

import os

import requests
import torch

from llm_rl.config import ModelConfig
from llm_rl.data import Example
from llm_rl.model import load_policy
from llm_rl.weight_sync import WeightSync

URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8001")
MODEL = "Qwen/Qwen3.5-0.8B-Base"
PROMPT = Example(problem="What is 17 times 24?", answer="408").prompt


def greedy(label: str) -> str:
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "n": 1,
        "max_tokens": 40,
        "temperature": 0.0,
        "seed": 0,
    }
    text = requests.post(f"{URL}/v1/completions", json=payload, timeout=600).json()["choices"][0]["text"]
    print(f"  [{label}] {text[:110]!r}")
    return text


print("=== step 1: baseline (engine on checkpoint weights) ===")
baseline = greedy("baseline")

print("\n=== loading trainer policy on the same GPU as the engine ===")
policy = load_policy(ModelConfig(gradient_checkpointing=False), device="cuda")
sync = WeightSync(policy.model, URL)
print(f"  parameters in sync source: {sync.num_synced} (vision tower excluded)")

print("\n=== step 2: sync unmodified weights (expect identical output) ===")
sync.sync()
unchanged = greedy("after no-op sync")
print(f"  identical to baseline: {unchanged == baseline}")

print("\n=== step 3: perturb a text layer and sync (expect different output) ===")
target = policy.backbone.layers[0].mlp.gate_proj.weight
original = target.detach().clone()
with torch.no_grad():
    target.add_(torch.randn_like(target, dtype=torch.float32).to(target.dtype) * 0.05)
sync.sync()
perturbed = greedy("after perturbed sync")
print(f"  differs from baseline: {perturbed != baseline}")

# Leave the engine on clean weights. Without this the server keeps serving the
# perturbed policy and every later eval silently measures a broken model.
print("\n=== step 4: restore original weights and sync back ===")
with torch.no_grad():
    target.copy_(original)
sync.sync()
restored = greedy("after restore")
print(f"  restored to baseline: {restored == baseline}")

print("\n=== result ===")
ok = (unchanged == baseline) and (perturbed != baseline) and (restored == baseline)
print("  IPC weight sync is faithful, effective and reversible" if ok else "  FAILED")
