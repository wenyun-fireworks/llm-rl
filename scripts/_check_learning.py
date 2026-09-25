"""Confirm that a real training step actually moves the weights.

Guards the fp32-master-weight fix: with bf16 parameters an lr=1e-6 Adam update is
rounded away and training silently does nothing.
"""

import os

import torch

from llm_rl.config import Config
from llm_rl.data import load_examples
from llm_rl.model import load_policy
from llm_rl.rollout import Rollout
from llm_rl.trainer import Trainer
from llm_rl.weight_sync import WeightSync

URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8002")

cfg = Config.from_yaml("configs/reinforce.yaml").apply_overrides(
    [
        "rollout.prompts_per_step=8",
        "rollout.group_size=8",
        "rollout.max_new_tokens=1024",
        f"rollout.server_url={URL}",
        "algo.minibatch_size=8",
        "data.max_train_samples=200",
    ]
)

policy = load_policy(cfg.model, device="cuda")
print(f"param dtype: {next(policy.model.parameters()).dtype}, autocast: {policy.autocast_dtype}")
print(f"trainable: {policy.num_trainable() / 1e9:.3f}B")

examples = load_examples(cfg.data.train_dataset, max_samples=200, shuffle_seed=0)
rollout = Rollout(URL, cfg.model.name, cfg.rollout)
sync = WeightSync(policy.model, URL)
trainer = Trainer(cfg, policy, rollout, sync)

watched = {
    "embed_tokens": policy.backbone.embed_tokens.weight,
    "layer0.mlp.gate": policy.backbone.layers[0].mlp.gate_proj.weight,
    "layer3.attn(full)": policy.backbone.layers[3].self_attn.q_proj.weight,
    "layer23.mlp.down": policy.backbone.layers[23].mlp.down_proj.weight,
}
before = {k: v.detach().clone() for k, v in watched.items()}

torch.cuda.reset_peak_memory_stats()
for step in range(3):
    metrics = trainer.train_step(examples[step * 8 : (step + 1) * 8])
    print(
        f"step {step}: reward={metrics.get('reward/mean', 0):.3f} "
        f"loss={metrics.get('loss/total', 0):+.5f} gn={metrics.get('train/grad_norm', 0):.3f} "
        f"updates={metrics.get('train/minibatch_updates', 0):.0f}"
    )

print(f"\npeak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
print(f"\n{'parameter':<22}{'% moved':>10}{'mean |delta|':>15}")
for name, tensor in watched.items():
    delta = (tensor.detach() - before[name]).abs()
    print(f"{name:<22}{(delta > 0).float().mean().item():>9.1%}{delta.mean().item():>15.3e}")
