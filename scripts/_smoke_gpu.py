"""Stage-3 smoke test: bf16 forward/backward on B200, chunked logprob memory."""

import time

import torch

from llm_rl.config import ModelConfig
from llm_rl.logprobs import compute_logprobs
from llm_rl.model import load_policy

torch.manual_seed(0)
cfg = ModelConfig(gradient_checkpointing=True)
policy = load_policy(cfg, device="cuda")
print(f"trainable params: {policy.num_trainable()/1e9:.3f}B  vocab={policy.vocab_size}")

frozen = sum(p.numel() for p in policy.model.parameters() if not p.requires_grad)
print(f"frozen params:    {frozen/1e6:.0f}M (vision tower)")

BATCH, SEQ = 4, 1024
input_ids = torch.randint(0, 100_000, (BATCH, SEQ), device="cuda")
attention_mask = torch.ones_like(input_ids)


def run(chunk_tokens: int, label: str, keep_grads: bool = False):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    policy.model.zero_grad(set_to_none=True)
    start = time.time()
    out = compute_logprobs(
        policy,
        input_ids,
        attention_mask,
        chunk_tokens=chunk_tokens,
        compute_entropy=True,
    )
    loss = out.logprobs.mean()
    loss.backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"{label:28s} peak={peak:6.1f} GiB  {time.time()-start:5.2f}s  logprobs={tuple(out.logprobs.shape)}")
    return peak


# Warm up the Gated DeltaNet / attention kernels so the timings below are steady state.
run(512, "warmup")
print(f"\nforward+backward on [{BATCH}, {SEQ}] tokens, {BATCH*SEQ} positions")
chunked = run(512, "chunked (512 tok)")
big = run(BATCH * SEQ, "unchunked (all 4096 tok)")
print(f"\nchunking saves {big - chunked:.1f} GiB ({big/max(chunked,1e-9):.1f}x lower peak)")

# Gradients must reach both attention flavours: Gated DeltaNet (linear_attention) on
# most layers and standard full attention on layers 3/7/11/15/19/23.
layers = policy.backbone.layers
checks = {
    "linear_attn layer 0": layers[0],
    "full_attn layer 3": layers[3],
    "embed_tokens (tied lm_head)": policy.backbone.embed_tokens,
}
run(512, "rerun to populate grads")
print()
for label, module in checks.items():
    grads = [(n, p.grad) for n, p in module.named_parameters() if p.requires_grad]
    have = [n for n, g in grads if g is not None and torch.isfinite(g).all() and g.abs().sum() > 0]
    print(f"{label:30s} {len(have)}/{len(grads)} params have finite nonzero grads")

vision_grads = [p.grad for p in policy.model.model.visual.parameters()]
print(f"vision tower grads (want all None): {all(g is None for g in vision_grads)}")

# The chunking win scales with total tokens, since unchunked logit memory is
# O(batch * seq * vocab) while chunked is O(chunk * vocab).
print("\nscaling: peak GiB at realistic RL batch shapes")
print(f"{'shape':>14s} {'tokens':>7s} {'chunked':>9s} {'unchunked':>11s}")
for batch, seq in [(4, 1024), (8, 2048), (16, 2048)]:
    input_ids = torch.randint(0, 100_000, (batch, seq), device="cuda")
    attention_mask = torch.ones_like(input_ids)
    results = {}
    for chunk, key in [(512, "chunked"), (batch * seq, "unchunked")]:
        try:
            results[key] = run(chunk, f"  [{batch},{seq}] {key}")
        except torch.OutOfMemoryError:
            results[key] = None
            torch.cuda.empty_cache()
    fmt = lambda v: f"{v:.1f}" if v is not None else "OOM"
    print(
        f"{f'[{batch},{seq}]':>14s} {batch*seq:7d} "
        f"{fmt(results['chunked']):>9s} {fmt(results['unchunked']):>11s}"
    )
