"""Baseline evaluation of the starting policy.

This is the go/no-go gate for the whole project: if AIME accuracy is a hard zero and
the model cannot even emit a parseable \\boxed{}, then AIME provides no gradient and
the training signal has to come from GSM8K / MATH-500 instead.
"""

import argparse
import json
import os
import time
from pathlib import Path

from llm_rl.config import EvalConfig, RewardConfig
from llm_rl.eval import DEFAULT_EVAL_SPECS, evaluate, load_eval_suite
from llm_rl.rollout import Rollout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.environ.get("VLLM_URL", "http://127.0.0.1:8001"))
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=3072)
    parser.add_argument("--max-per-dataset", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", default="runs/baseline.json")
    parser.add_argument("--datasets", nargs="*", default=None)
    args = parser.parse_args()

    specs = DEFAULT_EVAL_SPECS
    if args.datasets:
        specs = [s for s in DEFAULT_EVAL_SPECS if s[0] in args.datasets]

    cfg = EvalConfig(
        avg_at_k=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
    )
    # RolloutConfig fields are overridden per call by evaluate(); only the defaults
    # that evaluate does not pass through matter here.
    from llm_rl.config import RolloutConfig

    rollout = Rollout(args.url, args.model, RolloutConfig(max_new_tokens=args.max_new_tokens))

    suite = load_eval_suite(specs, max_per_dataset=args.max_per_dataset)
    print(f"evaluating {args.model} @ k={args.k}, temperature={args.temperature}\n")

    results = {}
    for name, examples in suite.items():
        start = time.time()
        result = evaluate(
            rollout,
            examples,
            cfg,
            RewardConfig(),
            dataset_name=name,
            batch_size=args.batch_size,
        )
        print(f"{result.summary()}  ({time.time() - start:.0f}s)")
        results[name] = {
            k: v for k, v in result.__dict__.items() if k != "per_problem"
        } | {"per_problem": result.per_problem}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
