"""Does Qwen3.5-9B-Base train on one GPU beside a vLLM engine?

Checks the two assumptions the 8-run design rests on:
  1. peak memory at the target context leaves room for a colocated engine
  2. 8-bit Adam moments still apply a 1e-6 update to fp32 parameters

No server needed; this is the trainer half only.
"""

import argparse

import torch

from llm_rl.config import ModelConfig, OptimConfig
from llm_rl.logprobs import compute_logprobs
from llm_rl.model import load_policy
from llm_rl.trainer import build_optimizer

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="Qwen/Qwen3.5-9B-Base")
parser.add_argument("--optimizer", default="adamw8bit")
parser.add_argument("--minibatch", type=int, default=8)
parser.add_argument("--contexts", type=int, nargs="*", default=[8192, 12288])
parser.add_argument("--engine-gib", type=float, default=45.0, help="budget to reserve for vLLM")
args = parser.parse_args()

torch.manual_seed(0)
policy = load_policy(ModelConfig(name=args.model, gradient_checkpointing=True), device="cuda")

trainable = policy.num_trainable()
frozen = sum(p.numel() for p in policy.model.parameters() if not p.requires_grad)
print(f"model: {args.model}")
print(f"  trainable {trainable/1e9:.3f}B   frozen(vision) {frozen/1e6:.0f}M")
print(f"  param dtype {next(policy.model.parameters()).dtype}, autocast {policy.autocast_dtype}")
print(f"  weights alone: {trainable*4/2**30:.1f} GiB fp32")

opt = build_optimizer(policy.trainable_parameters(), OptimConfig(optimizer=args.optimizer, lr=1e-6))
print(f"  optimizer: {type(opt).__name__}")

total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"\nGPU total {total_gib:.0f} GiB, reserving {args.engine_gib:.0f} GiB for the engine")
print(f"{'context':>9}{'peak GiB':>11}{'+engine':>10}{'headroom':>10}  verdict")

watched = policy.backbone.layers[0].mlp.gate_proj.weight
before = watched.detach().clone()

for seq in args.contexts:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        ids = torch.randint(0, 100_000, (args.minibatch, seq), device="cuda")
        mask = torch.ones_like(ids)
        opt.zero_grad(set_to_none=True)
        out = compute_logprobs(policy, ids, mask, chunk_tokens=512)
        out.logprobs.mean().backward()
        torch.nn.utils.clip_grad_norm_(policy.trainable_parameters(), 1.0)
        opt.step()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 2**30
        total = peak + args.engine_gib
        head = total_gib - total
        print(f"{seq:>9}{peak:>11.1f}{total:>10.1f}{head:>10.1f}  {'OK' if head > 8 else 'TIGHT' if head > 0 else 'NO FIT'}")
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"{seq:>9}{'OOM':>11}")

delta = (watched.detach() - before).abs()
print(f"\n8-bit Adam update check on layer0.mlp.gate:")
print(f"  weights moved: {(delta > 0).float().mean().item():.1%}   mean |delta|: {delta.mean().item():.3e}")
print("  (expect ~100% moved and ~1e-6 per step; near 0% means updates are being rounded away)")
