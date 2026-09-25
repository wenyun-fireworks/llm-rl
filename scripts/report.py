"""Compare runs: summary table plus learning-curve plots.

    python scripts/report.py                 # table for every run under runs/
    python scripts/report.py --plot          # also write runs/comparison.png
    python scripts/report.py --watch         # refresh the table every 60s

Reads runs/<name>/metrics.jsonl, which each trainer appends to as it goes, so this
works on partial results while training is still in flight.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

RUNS_DIR = Path("runs")

# Order matters: this is the order rows and plot lines appear in.
PREFERRED_ORDER = [
    "dapo_full",
    "dapo_cosine",
    "dapo_dyn",
    "dapo_g16",
    "dapo_lr2",
    "dapo_entropy",
    "dapo_noshape",
    "grpo_9b",
]


def load_run(path: Path) -> dict:
    steps, evals = [], []
    for line in (path / "metrics.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "eval" in record:
            evals.append(record)
        elif "error" in record:
            steps.append(record)
        else:
            steps.append(record)
    config = {}
    config_path = path / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
    return {"name": path.name, "steps": steps, "evals": evals, "config": config}


def moving_average(values: list[float], window: int = 20) -> list[float]:
    out, acc = [], []
    for v in values:
        acc.append(v)
        if len(acc) > window:
            acc.pop(0)
        out.append(sum(acc) / len(acc))
    return out


def summarize(run: dict) -> dict:
    steps = [s for s in run["steps"] if "reward/mean" in s]
    if not steps:
        return {"name": run["name"], "n_steps": 0}
    rewards = [s["reward/mean"] for s in steps]
    accs = [s.get("reward/accuracy", 0.0) for s in steps]
    fmt = [s.get("reward/format_rate", 0.0) for s in steps]
    trunc = [s.get("rollout/truncated_frac", 0.0) for s in steps]
    lens = [s.get("rollout/response_len_mean", 0.0) for s in steps]
    times = [s.get("time/step_s", 0.0) for s in steps]

    head = max(len(accs) // 5, 1)
    algo = run["config"].get("algo", {})
    errors = [s for s in run["steps"] if "error" in s]
    return {
        "name": run["name"],
        "algo": algo.get("name", "?"),
        "advantage": algo.get("advantage", "?"),
        "n_steps": len(steps),
        "acc_first": sum(accs[:head]) / head,
        "acc_last": sum(accs[-head:]) / head,
        "delta": sum(accs[-head:]) / head - sum(accs[:head]) / head,
        "reward_last": sum(rewards[-head:]) / head,
        "fmt_last": sum(fmt[-head:]) / head,
        "trunc_last": sum(trunc[-head:]) / head,
        "len_last": sum(lens[-head:]) / head,
        "sec_per_step": sum(times) / len(times),
        "error": errors[-1]["error"] if errors else "",
    }


def print_table(summaries: list[dict]) -> None:
    print(
        f"{'run':<15}{'algo':<11}{'adv':<7}{'steps':>6}"
        f"{'acc first':>11}{'acc last':>10}{'delta':>9}"
        f"{'fmt':>7}{'trunc':>7}{'len':>7}{'s/step':>8}"
    )
    print("-" * 104)
    for s in summaries:
        if not s["n_steps"]:
            print(f"{s['name']:<15}{'(no steps yet)':<40}")
            continue
        print(
            f"{s['name']:<15}{s['algo']:<11}{s['advantage']:<7}{s['n_steps']:>6}"
            f"{s['acc_first']:>10.1%}{s['acc_last']:>10.1%}{s['delta']:>+9.1%}"
            f"{s['fmt_last']:>7.0%}{s['trunc_last']:>7.0%}{s['len_last']:>7.0f}"
            f"{s['sec_per_step']:>8.0f}"
        )
    failed = [s for s in summaries if s.get("error")]
    if failed:
        print("\nerrors:")
        for s in failed:
            print(f"  {s['name']}: {s['error'][:110]}")


def print_matched_step_table(runs: list[dict]) -> None:
    """Compare every run at the same step count.

    Runs stop on a wall-clock deadline, and PPO is roughly 3x slower per step than
    GRPO (two inner epochs plus a value pass), so it reaches far fewer steps. That
    makes the raw "last N steps" table a wall-clock comparison. Truncating everyone
    to the shortest run gives the sample-matched comparison instead, which is the
    fairer one for judging the algorithms themselves.
    """
    lengths = [len([s for s in r["steps"] if "reward/accuracy" in s]) for r in runs]
    lengths = [n for n in lengths if n > 0]
    if len(lengths) < 2:
        return
    common = min(lengths)
    window = max(common // 5, 1)

    print(f"\nmatched at {common} steps (every run truncated to the shortest):")
    print(f"{'run':<15}{'acc first':>11}{'acc last':>10}{'delta':>9}{'fmt':>7}{'trunc':>7}")
    print("-" * 59)
    for run in runs:
        accs = [s["reward/accuracy"] for s in run["steps"] if "reward/accuracy" in s][:common]
        if not accs:
            continue
        fmt = [s.get("reward/format_rate", 0) for s in run["steps"] if "reward/accuracy" in s][:common]
        trunc = [s.get("rollout/truncated_frac", 0) for s in run["steps"] if "reward/accuracy" in s][:common]
        first, last = sum(accs[:window]) / window, sum(accs[-window:]) / window
        print(
            f"{run['name']:<15}{first:>10.1%}{last:>10.1%}{last - first:>+9.1%}"
            f"{sum(fmt[-window:]) / window:>7.0%}{sum(trunc[-window:]) / window:>7.0%}"
        )


# Qwen3.5-9B-Base before any training, measured at avg@16 with DAPO's decoding
# settings (temperature 1.0, top_p 0.7). See runs/baseline_9b.json.
# For reference, Qwen3.5-4B-Base at temperature 0.6 was aime_2024 17.9%,
# math_500 69.2% (runs_archive/4b_comparison/).
BASELINE = {"aime_2024": 0.3000, "aime_2025": 0.2333, "math_500": 0.6979}


def eval_trajectory(run: dict) -> dict[str, dict]:
    """Best and final avg@k per dataset, with the step where the best occurred.

    Peak matters more than final here. With `kl_coef: 0` nothing holds the policy
    near the base model, so these runs improve for a while and then over-optimize:
    training reward stays up while held-out accuracy falls back. Reporting only the
    last value would understate every algorithm and hide the turnover.
    """
    by_dataset: dict[str, list[tuple[int, float]]] = {}
    for record in run["evals"]:
        e = record["eval"]
        by_dataset.setdefault(e.get("dataset", "?"), []).append(
            (record["step"], e.get("avg_at_k", 0.0))
        )
    out = {}
    for dataset, points in by_dataset.items():
        points.sort()
        best_step, best = max(points, key=lambda p: p[1])
        out[dataset] = {
            "best": best,
            "best_step": best_step,
            "final": points[-1][1],
            "final_step": points[-1][0],
            "baseline": BASELINE.get(dataset),
        }
    return out


def print_eval_summary(runs: list[dict]) -> None:
    rows = [(r["name"], eval_trajectory(r)) for r in runs]
    rows = [(n, t) for n, t in rows if t]
    if not rows:
        return
    print(
        f"\n{'run':<15}{'dataset':<12}{'base':>7}{'best':>8}{'@step':>7}"
        f"{'final':>8}{'gain(best)':>12}"
    )
    print("-" * 70)
    for name, traj in rows:
        for dataset, d in sorted(traj.items()):
            base = d["baseline"]
            gain = f"{d['best'] - base:+.1%}" if base is not None else "n/a"
            base_str = f"{base:.1%}" if base is not None else "n/a"
            print(
                f"{name:<15}{dataset:<12}{base_str:>7}{d['best']:>8.1%}"
                f"{d['best_step']:>7}{d['final']:>8.1%}{gain:>12}"
            )


def print_evals(runs: list[dict]) -> None:
    rows = [(r["name"], e) for r in runs for e in r["evals"]]
    if not rows:
        return
    print(f"\n{'run':<15}{'step':>6}{'dataset':<22}{'avg@k':>8}{'pass@k':>8}")
    print("-" * 60)
    for name, record in rows:
        e = record["eval"]
        print(
            f"{name:<15}{record['step']:>6}{e.get('dataset', ''):<22}"
            f"{e.get('avg_at_k', 0):>8.1%}{e.get('pass_at_k', 0):>8.1%}"
        )


def make_plots(runs: list[dict], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("reward/accuracy", "training accuracy (fraction of rollouts correct)"),
        ("reward/format_rate", "format rate (emits \\boxed{})"),
        ("rollout/truncated_frac", "truncated fraction"),
        ("rollout/response_len_mean", "mean response length (tokens)"),
        ("train/grad_norm", "policy gradient norm"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(19, 9))
    for ax, (key, title) in zip(axes.flat, panels):
        plotted = False
        for run in runs:
            xs = [s["step"] for s in run["steps"] if key in s]
            ys = [s[key] for s in run["steps"] if key in s]
            if not ys:
                continue
            ax.plot(xs, moving_average(ys), label=run["name"], linewidth=1.6)
            plotted = True
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
        if plotted:
            ax.legend(fontsize=7)

    # Held-out AIME. These are the actual points, not a moving average: with only
    # 30 problems at k=4 the curve is genuinely this noisy, and smoothing it would
    # imply a precision the measurement does not have.
    ax = axes.flat[5]
    for run in runs:
        points = sorted(
            (r["step"], r["eval"]["avg_at_k"])
            for r in run["evals"]
            if r["eval"].get("dataset") == "aime_2024"
        )
        if points:
            ax.plot(*zip(*points), marker="o", markersize=3, label=run["name"], linewidth=1.4)
    ax.axhline(BASELINE["aime_2024"], color="black", linestyle="--", linewidth=1.2)
    ax.text(0.02, BASELINE["aime_2024"] + 0.01, "base model", transform=ax.get_yaxis_transform(),
            fontsize=8)
    ax.set_title("held-out AIME 2024 avg@4 (30 problems, unsmoothed)", fontsize=11)
    ax.set_xlabel("step")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.suptitle(
        "REINFORCE vs GRPO vs PPO on Qwen3.5-4B-Base (DAPO-Math-17k), "
        "curves are 20-step moving averages",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    print(f"\nwrote {out_path}")


def write_markdown(runs: list[dict], out_path: Path) -> None:
    """Write a self-contained summary of the comparison."""
    summaries = [summarize(r) for r in runs]
    done = [s for s in summaries if s["n_steps"]]
    lines = [
        "# Algorithm comparison",
        "",
        "Qwen3.5-4B-Base, DAPO-Math-17k, 16 prompts x 8 samples per step.",
        "All runs share one codebase and differ only in their `algo` config block.",
        "",
        "`acc` is the fraction of training rollouts whose boxed answer matched the",
        "gold integer, averaged over the first and last fifth of each run.",
        "",
        "## Wall-clock comparison (each run used the same time budget)",
        "",
        "| run | algo | advantage | steps | acc first | acc last | delta | format | trunc | len | s/step |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in done:
        lines.append(
            f"| `{s['name']}` | {s['algo']} | {s['advantage']} | {s['n_steps']} | "
            f"{s['acc_first']:.1%} | {s['acc_last']:.1%} | {s['delta']:+.1%} | "
            f"{s['fmt_last']:.0%} | {s['trunc_last']:.0%} | {s['len_last']:.0f} | {s['sec_per_step']:.0f} |"
        )

    counts = [len([x for x in r["steps"] if "reward/accuracy" in x]) for r in runs]
    counts = [c for c in counts if c > 0]
    if counts:
        common = min(counts)
        window = max(common // 5, 1)
        lines += [
            "",
            f"## Sample-matched comparison (all runs truncated to {common} steps)",
            "",
            "PPO takes roughly 3x longer per step (two inner epochs plus a value",
            "pass), so it completes far fewer steps in the same wall-clock budget.",
            "This table equalizes the number of rollouts instead of the time.",
            "",
            "| run | acc first | acc last | delta |",
            "|---|---|---|---|",
        ]
        for run in runs:
            accs = [x["reward/accuracy"] for x in run["steps"] if "reward/accuracy" in x][:common]
            if not accs:
                continue
            first, last = sum(accs[:window]) / window, sum(accs[-window:]) / window
            lines.append(f"| `{run['name']}` | {first:.1%} | {last:.1%} | {last - first:+.1%} |")

    traj_rows = [(r["name"], eval_trajectory(r)) for r in runs]
    traj_rows = [(n, t) for n, t in traj_rows if t]
    if traj_rows:
        lines += [
            "",
            "## Held-out evaluation: peak vs final",
            "",
            "These runs use `kl_coef: 0`, so nothing anchors the policy to the base",
            "model. They improve for a while and then over-optimize: training reward",
            "stays high while held-out accuracy falls back. The peak, and the step it",
            "occurred at, is therefore the number that matters.",
            "",
            "| run | dataset | baseline | best | at step | final | gain at best |",
            "|---|---|---|---|---|---|---|",
        ]
        for name, traj in traj_rows:
            for dataset, d in sorted(traj.items()):
                base = d["baseline"]
                base_s = f"{base:.1%}" if base is not None else "n/a"
                gain = f"**{d['best'] - base:+.1%}**" if base is not None else "n/a"
                lines.append(
                    f"| `{name}` | {dataset} | {base_s} | {d['best']:.1%} | "
                    f"{d['best_step']} | {d['final']:.1%} | {gain} |"
                )

    lines += [
        "",
        "## How to read this",
        "",
        "- Two seeds per algorithm are included on purpose. If the gap between two",
        "  algorithms is smaller than the gap between their own two seeds, the",
        "  comparison has not resolved anything.",
        "- `grpo_orig` vs `grpo_s0` isolates the normalization choice alone:",
        "  dividing by the group std and averaging per sequence, versus neither.",
        "- `nobaseline` vs `reinforce_s0` isolates the baseline itself. Same loop,",
        "  same loss, advantage is the raw reward.",
        "- Rising `format` with flat `acc` means the model learned to emit",
        "  `\\boxed{}` before it learned to be right, which is the expected order.",
        "",
        "![comparison](comparison.png)",
        "",
    ]
    if any(s.get("error") for s in summaries):
        lines += ["## Runs that stopped early", ""]
        for s in summaries:
            if s.get("error"):
                lines.append(f"- `{s['name']}`: {s['error'][:200]}")
        lines.append("")
    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path}")


def collect() -> list[dict]:
    paths = [p for p in RUNS_DIR.iterdir() if (p / "metrics.jsonl").exists()] if RUNS_DIR.exists() else []
    order = {name: i for i, name in enumerate(PREFERRED_ORDER)}
    paths.sort(key=lambda p: (order.get(p.name, 99), p.name))
    return [load_run(p) for p in paths]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--markdown", action="store_true", help="write runs/RESULTS.md")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()

    while True:
        runs = collect()
        if not runs:
            print("no runs found under runs/")
        else:
            print(f"\n=== {time.strftime('%H:%M:%S')} ===")
            print_table([summarize(r) for r in runs])
            print_matched_step_table(runs)
            print_eval_summary(runs)
            if args.plot:
                make_plots(runs, RUNS_DIR / "comparison.png")
            if args.markdown:
                write_markdown(runs, RUNS_DIR / "RESULTS.md")
        if not args.watch:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
