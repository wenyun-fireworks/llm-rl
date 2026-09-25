# Findings from the llm-rl experiments

A running summary of what we learned while building and training this stack
(Aug 31 – Sep 7, 2026). Numbers below are from runs on this codebase; local
`runs/` and `logs/` were cleaned before publishing, so treat this file as the
record of those results.

For algorithms and failure modes encoded in the code, start with
[TUTORIAL.md](TUTORIAL.md). This document is about **empirical outcomes**.

---

## 1. Setup in one paragraph

Shared rollout stack (vLLM generate → verifiable `\boxed{}` reward → trainer
recomputes logprobs → policy-gradient update → CUDA IPC weight sync). No TRL /
verl. Train on DAPO-Math-17k; headline eval is AIME 2024 (also MATH-500 / GSM8K /
AIME 2025–2026). Hardware: B200-class GPUs, colocated engine + trainer per GPU.

---

## 2. Phase A — 4B algorithm comparison

**Model:** Qwen3.5-4B-Base  
**Eval:** AIME 2024 avg@4, temp 0.6 (baseline **17.9%**)  
**Algos:** RLOO / REINFORCE, GRPO (Dr.GRPO-style), PPO, plus `nobaseline` and
`grpo_orig` ablations

| run | peak | at step | final | peak gain |
|---|---|---|---|---|
| `grpo_orig` | **41.7%** | 49 | 15.0% | +23.7 pp |
| `grpo_s1` | 38.3% | 249 | **24.2%** | +20.4 pp |
| `nobaseline` | 38.3% | 49 | 22.5% | +20.4 pp |
| `grpo_s0` | 37.5% | 99 | 15.8% | +19.6 pp |
| `reinforce_s1` | 34.2% | 49 | 20.0% | +16.2 pp |
| `reinforce_s0` | 30.8% | 49 | 14.2% | +12.9 pp |
| `ppo_s0` / `ppo_s1` | ≤2.5% | — | **0.0%** | collapse |

### Takeaways

1. **RL roughly doubles AIME, and does it early.** Non-PPO runs peaked in the
   first ~50–100 steps. Gains of ~+13 to +24 pp over the base.
2. **More steps hurt without a KL anchor.** With `kl_coef: 0`, every healthy run
   peaked then decayed; training reward kept rising while held-out accuracy fell
   (over-optimization on DAPO-Math). MATH-500 showed the same monotone decline more
   cleanly than AIME’s 30-problem noise.
3. **Seed noise ≥ algorithm gaps.** Two GRPO seeds finished 8.4 pp apart — wider
   than most between-algorithm differences. Do not crown a winner from one seed.
4. **`grpo_orig` vs `grpo_s0` is confounded.** Sequence-level loss norm produced
   median grad norms ~7 vs ~1.6; clipping then silently gave them different
   effective learning rates.

### Why PPO died (structural, not a one-off bug)

`value/explained_variance` stayed in **−2 … −9** across four attempted fixes
(zero-init + whitening, detached critic, separate clip budgets, …). Advantages
became noise → responses lengthened → hit the 4096 cap → truncated rollouts masked
out of the loss → effective batch → 0 → gradient exactly zero. Dead by ~step 24.

Group baselines (RLOO / GRPO) are safer here: when a group has no reward variance,
advantages are **exactly zero** (no-op). A bad critic always has an opinion.

This does **not** prove PPO cannot work on math RL. It shows a thin linear critic
on a shared backbone, with sparse binary rewards over long sequences, is fragile
in a way the group methods are not.

---

## 3. Phase B — DAPO-style 9B

**Model:** Qwen3.5-9B-Base (no Qwen3.5-27B/32B-Base exists; paper-scale DAPO on
128×H800 is ~2.5 orders of magnitude beyond 8×B200, so this is a reduced recipe)  
**Eval:** DAPO protocol — avg@32, temp 1.0, top_p 0.7  
**Fit trick:** 8-bit Adam so 9B + colocated vLLM fit on one GPU

| step | AIME 2024 | increment |
|---|---|---|
| 9B-Base, no training | **30.0%** | — |
| + RL with clip-higher & token-norm (`grpo_9b`) | **53.8%** | **+23.8** |
| + overlong shaping, dual-clip, wd (`dapo_full`) | **57.1%** | +3.3 |
| + cosine LR (`dapo_cosine`) | **58.3%** | +1.2 |
| best DAPO ckpt we carried forward | **55.0%** avg@32 | — |

### Takeaways

1. **Most of the jump from the 4B peak (~42%) to ~58% is the bigger base model**
   (~+12 pp before any training), not DAPO-specific tricks.
2. **“RL on a verifiable reward works” is ~+24 pp at both 4B and 9B** — remarkably
   consistent.
3. **DAPO’s distinctive pieces add a real but modest ~+3 pp on peak.** Where they
   shine is **stability**:

   | | truncation | format rate |
   |---|---|---|
   | with overlong shaping (`dapo_full`) | **~3%** | **~98%** |
   | without (`grpo_9b` / `dapo_noshape`) | ~41% | ~65% |

4. **Caveat on the control:** `grpo_9b` still had clip-higher and token-level loss
   norm (two of DAPO’s four techniques). It is “DAPO minus shaping/dual-clip/wd,”
   not vanilla GRPO.

---

## 4. Phase C — On-policy distillation (OPD)

**Teacher:** Qwen3.5-122B-A10B-FP8 — AIME 2024 avg@8 **80.4%**  
**Student start:** DAPO 9B at **55.0%** (or the base, in earlier ablations)  
**Gap:** ~25 pp headroom; a 35B-A3B teacher measured only 40.4% and would have
hurt the student.

### What we thought failed, and what was actually happening

Early OPD runs (student budget 6144) looked like total collapse: format → ~0%,
truncation → ~100%, AIME → 0%. Two wrong diagnoses came first:

1. “Score-function reverse KL drops the entropy term → mode collapse to junk.”
2. “Repetitive garbage hacking the KL.”

Inspecting generations overturned both. The student was writing **high-quality
long CoT in the teacher’s `<think>` style** (multiple solution methods, correct
intermediate answers) — then hitting the length cap before `\boxed{}`.

Teacher natural length ≈ **10,683** tokens. Cap at 6144 ⇒ never finishes ⇒
reward/format metrics look like failure even when reasoning transferred.

### The self-reinforcing stop failure

Even at a **19k** eval budget, a collapsed OPD checkpoint still truncated **93%**
of the time (mean length ~18.4k). Mechanism:

> Student rollouts truncate → those trajectories contain **no EOS** → on-policy
> distillation only trains on the student’s own samples → **no gradient for
> stopping** → style of “keep exploring methods” dominates every interior token
> while EOS is one terminal position out of ~10k.

Dense teacher “continue” signal across thousands of positions overwhelms a single
sequence-level correctness / length reward (even at `reward_mix=1.0`). **Seven
variants** (pure distill, reward mixes, start-from-DAPO, differentiable top-k KL
with entropy term, …) all hit the length wall the same way under a short budget.

### Engineering lessons from OPD

| lesson | detail |
|---|---|
| Budget must exceed teacher length | Motivates `configs/distill_20k.yaml` (max_new_tokens 19500, top-k reverse KL). |
| Allocator fragmentation | Probe: 20k @ micro_batch 1 peaks **89.6 GiB allocated**, but ragged shapes OOMed at **~142 GiB** process size. Fix: `pad_to_width` for constant shapes. `expandable_segments` breaks IPC weight sync — cannot use it. |
| Grad clip vs effective LR | Distill grad norms ~6–8 with `max_grad_norm=1.0` ⇒ effective lr ≈ lr/7. 20k config raises clip to 8.0. |
| Tokenizer identity is mandatory | All Qwen3.5 share vocab 248320 — OK. Cross-family teachers need a different approach. |
| Remote Fireworks teacher | Possible in principle (`echo` + logprobs), but top-k only, rate limits, and tokenizer risk; local 122B-FP8 was the better path here. |

OPD is **not RL**; it is dense supervision on the student’s own tokens. It can
transfer long-CoT style quickly, but **termination / format must be taught
separately** (longer budget, terminating teacher trajectories / SFT mix, or RL
after distill).

---

## 5. Infra and correctness findings (baked into the code)

These are the “five things that are easy to get wrong” from the tutorial, confirmed
with measurements:

1. **bf16 params eat `lr=1e-6` updates** — keep masters in fp32, autocast compute.
2. **Vocab (248k), not depth, sets the memory ceiling** — chunked log-softmax.
3. **Never trust sampler logprobs for ratios** — recompute in the trainer
   (vLLM vs HF: corr 0.9999 but worst-token Δ ≈ 0.16 ⇒ spurious ratio 1.14).
4. **Right-pad only** — hybrid Gated DeltaNet recurrence.
5. **Truncation ≠ wrong answer** — mask truncated rollouts; overlong *shaping*
   is the DAPO alternative to masking.

Also: clip-grad-norm is necessary (seen max norms >1000) but when median norm ≫
`max_norm` it silently becomes a fluctuating learning-rate reducer — confounding
ablations.

---

## 6. Experiment catalog (what we actually ran)

| phase | models | algorithms / setups | outcome |
|---|---|---|---|
| A | 4B-Base | REINFORCE×2, GRPO×2, PPO×2, nobaseline, grpo_orig | Peak AIME ~42%; PPO collapse; post-peak decay |
| B | 9B-Base | DAPO full / cosine / ablations, grpo_9b control, GSPO attempts | Peak AIME **58.3%** avg@32; shaping fixes truncation |
| C | 9B student ← 122B-FP8 teacher | OPD score-function, top-k KL, reward_mix, from-DAPO start, 6k→19k→20k budgets | Style transfer yes; stop learning fails under short budget |

Training set throughout: **DAPO-Math-17k**. AIME is eval-only (30 problems per
year); do not train on it.

---

## 7. Open questions / natural next steps

1. Add a small `kl_coef` (or early-stop on held-out) on the 4B/9B RL recipe and
   see if the peak **holds** instead of decaying.
2. Finish OPD at **≥ teacher natural length** with fixed-width padding
   (`distill_20k`), then RL for termination/format.
3. Mix a fraction of **teacher-generated terminating** trajectories (forward KL /
   SFT) into OPD so EOS appears in the data.
4. Cleaner GRPO control for DAPO ablations (no clip-higher / token-norm borrowed
   from DAPO).
5. Stronger critic / critic warmup if revisiting PPO — current negative result is
   about *this* critic, not the algorithm in the abstract.

---

*Compiled from the build/train session that produced this repository. Config
comments in `configs/distill.yaml` and `configs/distill_20k.yaml` capture the
same OPD conclusions next to the knobs they justify.*
