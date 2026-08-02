"""
sweep.py

Run the same training budget across every architecture and collect a
side-by-side ablation.

Each model runs in its own subprocess, so peak memory is measured per model
and one crash does not take down the sweep. Results land in a run directory as
per-model metrics.json plus a summary table and report.md.

Usage:
    python sweep.py
    python sweep.py --models gpt-2 deepseek-v3
    python sweep.py --steps 2000 --batch-size 32 --vocab-size 1000
    python sweep.py --report-only sweep_results/2026-08-02_14-30-00
    python sweep.py -- training.lr_peak=1e-3 model.dropout=0.1
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

ALL_MODELS = ["gpt-2", "deepseek-v1", "deepseek-v2", "deepseek-v3"]

# What each step of the ladder introduces -- printed alongside the results so the
# table reads as an ablation rather than four unrelated numbers
LADDER = {
    "gpt-2":       "MHA + dense FFN + sinusoidal",
    "deepseek-v1": "+ RoPE, + MoE",
    "deepseek-v2": "+ MLA, + DeepSeekMoE (fine-grained + shared)",
    "deepseek-v3": "+ aux-loss-free balancing, + MTP",
}


def parse_args():
    p = argparse.ArgumentParser(description="Ablation sweep across architectures")
    p.add_argument("--models", nargs="+", default=ALL_MODELS, help="Configs to run")
    p.add_argument("--out", type=str, default=None, help="Run directory (default: timestamped)")
    p.add_argument("--steps", type=int, default=None, help="Override training.n_steps for every model")
    p.add_argument("--batch-size", type=int, default=None, help="Override training.batch_size")
    p.add_argument("--seq-len", type=int, default=None, help="Override training.seq_len")
    p.add_argument("--vocab-size", type=int, default=None, help="Override tokenizer.vocab_size")
    p.add_argument("--force", action="store_true", help="Rerun models that already have metrics")
    p.add_argument("--report-only", type=str, default=None, help="Rebuild the report from an existing run dir")
    p.add_argument("overrides", nargs="*", help="Extra Hydra overrides applied to every model")
    return p.parse_args()


def shared_overrides(args, run_dir, model):
    """Overrides applied identically to every model -- this is what keeps it fair."""
    ov = [
        f"checkpoint.path={os.path.join(run_dir, model, 'train.pt')}",
        f"hydra.run.dir={os.path.join(run_dir, model, 'hydra')}",
    ]
    if args.steps is not None:
        ov.append(f"training.n_steps={args.steps}")
    if args.batch_size is not None:
        ov.append(f"training.batch_size={args.batch_size}")
    if args.seq_len is not None:
        ov.append(f"training.seq_len={args.seq_len}")
    if args.vocab_size is not None:
        ov.append(f"tokenizer.vocab_size={args.vocab_size}")
    return ov + list(args.overrides)


def run_model(args, run_dir, model):
    metrics_file = os.path.join(run_dir, model, "metrics.json")
    if os.path.exists(metrics_file) and not args.force:
        print(f"  [{model}] metrics already present, skipping (use --force to rerun)")
        return True

    os.makedirs(os.path.join(run_dir, model), exist_ok=True)
    log_file = os.path.join(run_dir, model, "train.log")

    cmd = [sys.executable, "train.py", "--config-name", model] + shared_overrides(args, run_dir, model)
    print(f"  [{model}] {' '.join(cmd[2:])}")
    print(f"  [{model}] log -> {log_file}")

    t0 = time.time()
    with open(log_file, "w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    dt = time.time() - t0

    if proc.returncode != 0:
        print(f"  [{model}] FAILED (exit {proc.returncode}) after {dt:.0f}s -- see {log_file}")
        tail = open(log_file).read().splitlines()[-15:]
        for line in tail:
            print(f"      | {line}")
        return False

    print(f"  [{model}] done in {dt/60:.1f} min")
    return True


def load_results(run_dir, models):
    results = {}
    for m in models:
        path = os.path.join(run_dir, m, "metrics.json")
        if os.path.exists(path):
            with open(path) as f:
                results[m] = json.load(f)
    return results


# ----------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------

def _fmt(v, spec="", dash="-"):
    return dash if v is None else format(v, spec)


def quality_table(results):
    rows = [("model", "change", "best test", "@step", "final val", "final train")]
    for m, r in results.items():
        rows.append((
            m, LADDER.get(m, ""),
            _fmt(r["best_test_loss"], ".4f"),
            _fmt(r["best_test_step"], "d"),
            _fmt(r["final_val_loss"], ".4f"),
            _fmt(r["final_train_loss"], ".4f"),
        ))
    return rows


def cost_table(results):
    rows = [("model", "total", "active/tok", "MTP", "KV/tok/layer", "MaxVio")]
    for m, r in results.items():
        rows.append((
            m,
            f"{r['params_total']:,}",
            f"{r['params_active']:,}",
            f"{r['params_mtp']:,}" if r["params_mtp"] else "-",
            str(r["kv_floats_per_token_per_layer"]),
            _fmt(r.get("final_maxvio"), ".3f"),
        ))
    return rows


def runtime_table(results):
    rows = [("model", "wall clock", "s/step", "train tok/s", "gen tok/s", "peak RSS")]
    for m, r in results.items():
        rows.append((
            m,
            f"{r['wall_clock_s']/60:.1f} min",
            f"{r['s_per_step']:.2f}",
            f"{r['train_tokens_per_s']:,.0f}",
            _fmt(r.get("gen_tok_per_s"), ".1f"),
            f"{r['peak_rss_gb']:.2f} GB",
        ))
    return rows


def render(rows):
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    out = []
    for i, row in enumerate(rows):
        out.append("  ".join(str(c).ljust(w) for c, w in zip(row, widths)).rstrip())
        if i == 0:
            out.append("  ".join("-" * w for w in widths))
    return "\n".join(out)


def markdown(rows):
    head = "| " + " | ".join(str(c) for c in rows[0]) + " |"
    sep = "|" + "|".join("---" for _ in rows[0]) + "|"
    body = ["| " + " | ".join(str(c) for c in r) + " |" for r in rows[1:]]
    return "\n".join([head, sep] + body)


def build_report(run_dir, results):
    if not results:
        print("No results to report.")
        return

    any_r = next(iter(results.values()))
    header = (
        f"Budget: {any_r['n_steps']:,} steps x {any_r['batch_size']} batch x "
        f"{any_r['seq_len']} seq = {any_r['tokens_seen']:,} tokens on {any_r['device']}"
    )

    sections = [
        ("Quality", quality_table(results),
         "Losses are pure next-token cross-entropy for every model -- the MoE balance "
         "loss and the MTP loss are training signals only and are excluded, so these "
         "numbers are directly comparable."),
        ("Capacity and cost", cost_table(results),
         "active/tok excludes the experts a token does not route to and the MTP head, "
         "which is discarded at inference. MaxVio is how far the busiest expert sits "
         "above an even share: 0 is perfect balance, and it is the number that says "
         "whether aux-loss-free balancing actually balances."),
        ("Runtime", runtime_table(results),
         "gen tok/s is measured without a KV cache, so it does not yet reflect MLA's "
         "smaller cache -- compare the KV/tok/layer column for that."),
    ]

    print()
    print("=" * 78)
    print("ABLATION RESULTS")
    print("=" * 78)
    print(header)
    for title, rows, _ in sections:
        print(f"\n{title}\n")
        print(render(rows))
    print()

    md = [f"# Architecture ablation\n", f"_{header}_\n"]
    for title, rows, note in sections:
        md += [f"## {title}\n", markdown(rows) + "\n", f"{note}\n"]
    md.append("## Loss curves\n")
    md.append("Per-step history is in each model's `metrics.json` under `history`.\n")
    for m, r in results.items():
        pts = [(h["step"], h.get("test_loss")) for h in r["history"] if h.get("test_loss") is not None]
        if pts:
            md.append(f"- **{m}**: " + ", ".join(f"{s}:{v:.3f}" for s, v in pts) + "\n")

    report_path = os.path.join(run_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(md))
    print(f"Report -> {report_path}")

    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({m: {k: v for k, v in r.items() if k != "history"} for m, r in results.items()}, f, indent=2)
    print(f"Summary -> {summary_path}")


def main():
    args = parse_args()

    if args.report_only:
        build_report(args.report_only, load_results(args.report_only, args.models))
        return

    run_dir = args.out or os.path.join("sweep_results", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(run_dir, exist_ok=True)
    print(f"Sweep dir: {run_dir}")
    print(f"Models: {', '.join(args.models)}\n")

    failed = []
    for i, model in enumerate(args.models, 1):
        print(f"[{i}/{len(args.models)}] {model}")
        if not run_model(args, run_dir, model):
            failed.append(model)
        print()

    build_report(run_dir, load_results(run_dir, args.models))

    if failed:
        print(f"\nFailed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
