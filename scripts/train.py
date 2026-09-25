"""Train a policy with REINFORCE, GRPO or PPO.

    python scripts/train.py configs/grpo.yaml --override total_steps=200 optim.lr=2e-6

The vLLM server must already be running on the same GPU (scripts/serve.sh), since
weight sync uses CUDA IPC and looks handles up by GPU UUID.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from llm_rl.algos import grpo as grpo_algo
from llm_rl.algos import gspo as gspo_algo
from llm_rl.algos import ppo as ppo_algo
from llm_rl.algos import reinforce as reinforce_algo
from llm_rl.config import Config
from llm_rl.data import infinite_batches, load_examples
from llm_rl.eval import evaluate
from llm_rl.model import load_policy, load_reference
from llm_rl.rollout import Rollout
from llm_rl.trainer import Trainer
from llm_rl.weight_sync import WeightSync


def validate(cfg: Config) -> None:
    if cfg.algo.name == "reinforce":
        reinforce_algo.validate(cfg.algo)
    elif cfg.algo.name == "grpo":
        grpo_algo.validate(cfg.algo)
    elif cfg.algo.name == "gspo":
        gspo_algo.validate(cfg.algo)
    elif cfg.algo.name == "ppo":
        ppo_algo.validate(cfg.algo, cfg.model)
    elif cfg.algo.name == "distill":
        # On-policy distillation needs a teacher that shares the student's
        # tokenizer; per-token alignment is the whole method.
        if not cfg.teacher.model.startswith("Qwen/Qwen3.5-"):
            raise ValueError(
                f"teacher {cfg.teacher.model!r} may not share the student's tokenizer. "
                "All Qwen3.5 checkpoints use the same 248320-token vocab; a teacher "
                "from another family would misalign every token."
            )
    else:
        raise ValueError(f"unknown algo {cfg.algo.name!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--eval-examples", type=int, default=30)
    parser.add_argument("--no-eval", action="store_true")
    # Unattended overnight runs need a hard stop so results are ready by a known
    # time regardless of how fast steps turn out to be.
    parser.add_argument("--max-hours", type=float, default=None)
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.override:
        cfg = cfg.apply_overrides(args.override)
    validate(cfg)

    torch.manual_seed(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2, default=str))
    log_file = (out_dir / "metrics.jsonl").open("a")

    print(f"[{cfg.algo.name}] loading {cfg.model.name}")
    policy = load_policy(cfg.model, device="cuda")
    print(f"  trainable params: {policy.num_trainable() / 1e9:.3f}B")

    reference = load_reference(cfg.model, device="cuda") if cfg.algo.kl_coef else None

    train_examples = load_examples(
        cfg.data.train_dataset,
        split=cfg.data.train_split,
        max_samples=cfg.data.max_train_samples,
        shuffle_seed=cfg.data.shuffle_seed,
        deduplicate=cfg.data.deduplicate,
        exclude_cjk=cfg.data.exclude_cjk,
    )
    print(f"  train examples: {len(train_examples)}")

    rollout = Rollout(cfg.rollout.server_url, cfg.model.name, cfg.rollout)
    weight_sync = WeightSync(policy.model, cfg.rollout.server_url)
    print(f"  weight sync over {weight_sync.num_synced} tensors")

    # DAPO dynamic sampling may need several generation rounds to fill one batch, so
    # the trainer needs to be able to pull fresh prompts rather than being handed a
    # fixed list per step.
    sampler_state = infinite_batches(train_examples, 1, seed=cfg.seed + 1)

    def example_sampler(n: int):
        return [next(sampler_state)[0] for _ in range(n)]

    trainer = Trainer(
        cfg, policy, rollout, weight_sync, reference=reference, example_sampler=example_sampler
    )
    batches = infinite_batches(train_examples, cfg.rollout.prompts_per_step, seed=cfg.seed)

    # AIME is the headline metric but only 30 problems, so at k=4 one problem is
    # 0.8 points and the curve is very noisy. MATH-500 is carried alongside it as a
    # denser signal that should move visibly within a few hundred steps.
    eval_suites: list[tuple[str, list]] = []
    if not args.no_eval:
        eval_suites = [
            ("aime_2024", load_examples(
                "HuggingFaceH4/aime_2024", split="train",
                max_samples=args.eval_examples, deduplicate=False)),
            ("math_500", load_examples(
                "HuggingFaceH4/MATH-500", split="test",
                max_samples=args.eval_examples, deduplicate=False)),
        ]

    best_score = -1.0
    deadline = time.time() + args.max_hours * 3600 if args.max_hours else None
    print(f"\ntraining for {cfg.total_steps} steps" + (f" or {args.max_hours}h\n" if deadline else "\n"))

    for step in range(cfg.total_steps):
        if deadline and time.time() > deadline:
            print(f"reached {args.max_hours}h wall-clock limit, stopping at step {step}")
            break
        start = time.time()
        try:
            metrics = trainer.train_step(next(batches))
        except Exception as error:
            # Log and stop cleanly rather than dying with a traceback: the metrics
            # written so far are still a usable learning curve, and a dead vLLM
            # engine cannot be recovered in-process anyway.
            print(f"step {step} failed: {type(error).__name__}: {error}")
            log_file.write(json.dumps({"step": step, "error": f"{type(error).__name__}: {error}"}) + "\n")
            log_file.flush()
            break
        metrics["step"] = step
        metrics["time/step_s"] = time.time() - start

        if step % cfg.log_every_steps == 0:
            print(
                f"step {step:4d}  reward={metrics.get('reward/mean', 0):.3f}  "
                f"acc={metrics.get('reward/accuracy', 0):.1%}  "
                f"fmt={metrics.get('reward/format_rate', 0):.1%}  "
                f"trunc={metrics.get('rollout/truncated_frac', 0):.1%}  "
                f"len={metrics.get('rollout/response_len_mean', 0):.0f}  "
                f"loss={metrics.get('loss/total', 0):+.4f}  "
                f"gn={metrics.get('train/grad_norm', 0):.2f}  "
                f"{metrics['time/step_s']:.0f}s"
            )
        log_file.write(json.dumps(metrics) + "\n")
        log_file.flush()

        if eval_suites and cfg.eval.every_steps and (step + 1) % cfg.eval.every_steps == 0:
            scores = {}
            for suite_name, suite in eval_suites:
                result = evaluate(rollout, suite, cfg.eval, cfg.reward, suite_name)
                print(f"  EVAL {result.summary()}")
                log_file.write(json.dumps({"step": step, "eval": result.__dict__}) + "\n")
                scores[suite_name] = result.avg_at_k
            log_file.flush()

            # Keep the best weights, not just the best number. These runs peak early
            # and then over-optimize, so without this the headline result is a
            # checkpoint we no longer have. Saved in bf16 to halve the footprint.
            selector = scores.get("math_500", scores.get("aime_2024", 0.0))
            if selector > best_score:
                best_score = selector
                best_dir = out_dir / "best"
                policy.model.to(torch.bfloat16).save_pretrained(best_dir)
                policy.model.to(torch.float32)
                policy.tokenizer.save_pretrained(best_dir)
                (best_dir / "best.json").write_text(
                    json.dumps({"step": step, "scores": scores}, indent=2)
                )
                print(f"  saved best checkpoint (selector={selector:.1%}) to {best_dir}")

        if cfg.save_every_steps and (step + 1) % cfg.save_every_steps == 0:
            path = out_dir / f"checkpoint-{step + 1}"
            policy.model.save_pretrained(path)
            policy.tokenizer.save_pretrained(path)
            print(f"  saved {path}")

    log_file.close()


if __name__ == "__main__":
    main()
