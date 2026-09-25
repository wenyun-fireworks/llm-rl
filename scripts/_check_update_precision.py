"""Does an lr=1e-6 AdamW update survive in bf16 parameters?

bf16 has an 8-bit mantissa, so its relative resolution is about 2^-8 = 0.4%. A
typical transformer weight is ~1e-2, so the smallest representable change is ~4e-5.
An Adam step at lr=1e-6 moves a weight by ~1e-6. If that is below the rounding
threshold the update is discarded and training is a silent no-op.
"""

import torch

torch.manual_seed(0)


def run(dtype, lr, steps=10):
    param = torch.nn.Parameter(torch.randn(4096, 4096, dtype=dtype) * 0.02)
    before = param.detach().clone().float()
    opt = torch.optim.AdamW([param], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        # Constant-ish gradient, so Adam's normalized step is ~1.0 and the update
        # magnitude is essentially lr.
        param.grad = torch.randn_like(param) * 0.001 + 0.01
        opt.step()
    after = param.detach().float()
    changed = (after != before).float().mean().item()
    delta = (after - before).abs()
    state = opt.state[param]
    return {
        "dtype": str(dtype),
        "lr": lr,
        "frac_changed": changed,
        "mean_abs_delta": delta.mean().item(),
        "expected_delta": lr * steps,
        "adam_state_dtype": str(state["exp_avg"].dtype),
    }


print(f"{'dtype':<16}{'lr':<10}{'state':<12}{'% weights moved':>16}{'actual Δ':>12}{'expected Δ':>12}")
for dtype in (torch.bfloat16, torch.float32):
    for lr in (1e-6, 1e-5, 1e-4):
        r = run(dtype, lr)
        print(
            f"{r['dtype']:<16}{r['lr']:<10.0e}{r['adam_state_dtype'].replace('torch.',''):<12}"
            f"{r['frac_changed']:>15.1%}{r['mean_abs_delta']:>12.2e}{r['expected_delta']:>12.2e}"
        )

print("\nSmallest representable relative step:")
for dtype in (torch.bfloat16, torch.float32):
    one = torch.tensor(0.02, dtype=dtype)
    eps = torch.finfo(dtype).eps
    print(f"  {str(dtype):<16} eps={eps:.2e}  abs resolution near 0.02 = {0.02 * eps:.2e}")
