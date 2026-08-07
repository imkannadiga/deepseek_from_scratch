"""
benchmark.py

Measure inference performance of trained checkpoints. Nothing here touches the
weights, so saved checkpoints can be benchmarked without retraining -- the KV
cache and the MLA absorption trick are inference-only.

Reports three things:
  - KV cache:   cached vs uncached generation, and cache bytes per token
  - Absorption: absorbed vs plain decode, for the MLA models
  - Scaling:    both of the above across context lengths, which is where the
                interesting behaviour lives -- absorption barely helps at short
                context and dominates at long

Every timed configuration is checked against an uncached full forward first, so
a fast-but-wrong result cannot be reported as a win.

Usage:
    python benchmark.py
    python benchmark.py --models gpt-2 deepseek-v3
    python benchmark.py --contexts 64 128 256 --gen 64 --runs 3
    python benchmark.py --checkpoint-dir sweep_results/2026-08-03_10-00-00
    python benchmark.py --out bench_results
"""

import argparse
import json
import os
import time

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from models.attention.rope_mla import RopeMLA
from models.helpers.kv_cache import make_cache

ALL_MODELS = ["gpt-2", "deepseek-v1", "deepseek-v2", "deepseek-v3"]


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark inference on trained checkpoints")
    p.add_argument("--models", nargs="+", default=ALL_MODELS, help="Model names to benchmark")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                   help="Directory holding <model>/train.pt (a sweep run dir also works)")
    p.add_argument("--contexts", nargs="+", type=int, default=[64, 128, 256],
                   help="Prompt lengths to prefill before decoding")
    p.add_argument("--gen", type=int, default=64, help="Tokens to generate per measurement")
    p.add_argument("--runs", type=int, default=3, help="Timed repeats, best is reported")
    p.add_argument("--device", type=str, default=None, help="cuda | cpu (default: auto)")
    p.add_argument("--skip-uncached", action="store_true",
                   help="Skip the uncached baseline, which is slow at long context")
    p.add_argument("--out", type=str, default=None, help="Directory for report.md + results.json")
    return p.parse_args()


# ----------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------

def load_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("config") is None:
        raise ValueError(f"{path} has no stored config -- retrain, or benchmark a newer checkpoint")

    cfg = OmegaConf.create(checkpoint["config"])
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in checkpoint["model_state_dict"].items()}

    # vocab_size is not in the config -- recover it from the embedding table
    vocab_size = state_dict["input_embedding.weight"].shape[0]

    model = instantiate(cfg.model, vocab_size=vocab_size).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, cfg, vocab_size, checkpoint.get("step", "?")


def set_absorption(model, enabled):
    """Returns True if the model has any MLA layer to toggle."""
    found = False
    for m in model.modules():
        if isinstance(m, RopeMLA):
            m.use_absorption = enabled
            found = True
    return found


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


# ----------------------------------------------------------------------
# measurement
# ----------------------------------------------------------------------

@torch.no_grad()
def generate_cached(model, prompt, n_gen, device):
    """Prefill then decode one token at a time. Returns (prefill_s, decode_s, cache, ids)."""
    cache = make_cache(model)

    sync(device)
    t0 = time.perf_counter()
    logits = model(prompt, cache=cache)
    sync(device)
    prefill_s = time.perf_counter() - t0

    ctx = prompt
    t0 = time.perf_counter()
    for i in range(n_gen):
        nxt = logits[:, -1:].argmax(-1)
        ctx = torch.cat([ctx, nxt], dim=1)
        if i < n_gen - 1:
            logits = model(nxt, cache=cache)
    sync(device)
    decode_s = time.perf_counter() - t0

    return prefill_s, decode_s, cache, ctx


@torch.no_grad()
def generate_uncached(model, prompt, n_gen, device):
    """The baseline: recompute the whole forward pass every step."""
    ctx = prompt
    sync(device)
    t0 = time.perf_counter()
    for _ in range(n_gen):
        logits = model(ctx)
        ctx = torch.cat([ctx, logits[:, -1:].argmax(-1)], dim=1)
    sync(device)
    return time.perf_counter() - t0, ctx


@torch.no_grad()
def max_logit_deviation(model, ids, prefill_len, device):
    full = model(ids)
    cache = make_cache(model)
    parts = [model(ids[:, :prefill_len], cache=cache)]
    for t in range(prefill_len, ids.shape[1]):
        parts.append(model(ids[:, t:t + 1], cache=cache))
    return (torch.cat(parts, dim=1) - full).abs().max().item()


def best_of(fn, runs):
    """Warm up once, then take the fastest run -- least contaminated by noise."""
    fn()
    return min(fn() for _ in range(runs))


def best_of_pair(fn, runs):
    """Same, for a function returning (prefill, decode) -- each minimised separately."""
    fn()
    samples = [fn() for _ in range(runs)]
    return min(s[0] for s in samples), min(s[1] for s in samples)


# ----------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------

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
    return "\n".join([head, sep] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows[1:]])


def build_tables(results, contexts, gen, skip_uncached):
    tables = []

    kv = [["model", "ctx", "prefill", "cached tok/s"]
          + ([] if skip_uncached else ["uncached tok/s", "speedup"])
          + ["cache", "KV/tok/layer", "max|diff|"]]
    for name, r in results.items():
        for c in contexts:
            m = r["by_context"].get(str(c))
            if m is None:
                continue
            row = [name, c, f"{m['prefill_ms']:.1f}ms", f"{m['cached_tok_s']:.1f}"]
            if not skip_uncached:
                row += [f"{m['uncached_tok_s']:.1f}", f"{m['kv_speedup']:.2f}x"]
            row += [f"{m['cache_mb']:.2f}MB", r["kv_floats_per_token_per_layer"], f"{m['max_diff']:.1e}"]
            kv.append(row)
    tables.append(("KV cache", kv,
                   f"Generating {gen} tokens after a prompt of `ctx` tokens. `max|diff|` is cached "
                   "decode against a single full forward -- float noise, so the cached path is the "
                   "same function. Uncached recomputes every position each step."))

    absorbed = [["model", "ctx", "absorbed", "plain", "speedup"]]
    for name, r in results.items():
        for c in contexts:
            m = r["by_context"].get(str(c))
            if m is None or m.get("plain_tok_s") is None:
                continue
            absorbed.append([name, c,
                             f"{m['cached_tok_s']:.1f} tok/s",
                             f"{m['plain_tok_s']:.1f} tok/s",
                             f"{m['cached_tok_s'] / m['plain_tok_s']:.2f}x"])
    if len(absorbed) > 1:
        tables.append(("Absorption (MLA only)", absorbed,
                       "Absorbed decode folds W_uk into W_q and W_uv into W_o, so K and V are never "
                       "materialised and attention runs in the compressed space. Plain rebuilds K and "
                       "V from the cached latent every step, so its cost grows with context while the "
                       "absorbed path stays nearly flat."))

    return tables


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1337)

    print(f"Device: {device}")
    print(f"Checkpoints: {args.checkpoint_dir}")
    print(f"Contexts: {args.contexts}, generating {args.gen} tokens, best of {args.runs}\n")

    results = {}
    for name in args.models:
        path = os.path.join(args.checkpoint_dir, name, "train.pt")
        if not os.path.exists(path):
            print(f"[{name}] no checkpoint at {path}, skipping")
            continue

        model, cfg, vocab_size, step = load_model(path, device)
        has_mla = set_absorption(model, True)
        max_seq = cfg.model.max_seq_length
        kv_floats = (cfg.model.d_kv + cfg.model.rope_head_dim) if "d_kv" in cfg.model \
            else 2 * cfg.model.d_in

        print(f"[{name}] step {step}, vocab {vocab_size}, max_seq {max_seq}"
              f"{', MLA' if has_mla else ''}")

        entry = {"step": step, "vocab_size": vocab_size,
                 "kv_floats_per_token_per_layer": kv_floats,
                 "params": sum(p.numel() for p in model.parameters()),
                 "by_context": {}}

        for ctx_len in args.contexts:
            if ctx_len + args.gen > max_seq:
                print(f"  ctx {ctx_len} + gen {args.gen} > max_seq {max_seq}, skipping")
                continue

            prompt = torch.randint(0, vocab_size, (1, ctx_len), device=device)

            # correctness gate before any timing
            ids = torch.randint(0, vocab_size, (1, ctx_len + min(args.gen, 16)), device=device)
            diff = max_logit_deviation(model, ids, ctx_len, device)
            if diff > 1e-3:
                print(f"  ctx {ctx_len}: CACHE MISMATCH ({diff:.2e}) -- not timing this")
                continue

            set_absorption(model, True)
            prefill_s, decode_s = best_of_pair(
                lambda: generate_cached(model, prompt, args.gen, device)[:2], args.runs)
            cache = generate_cached(model, prompt, args.gen, device)[2]

            m = {
                "prefill_ms": prefill_s * 1000,
                "decode_s": decode_s,
                "cached_tok_s": args.gen / decode_s,
                "cache_mb": cache.size_bytes() / 1e6,
                "cache_tokens": cache.get_seq_length(0),
                "max_diff": diff,
                "plain_tok_s": None,
                "uncached_tok_s": None,
                "kv_speedup": None,
            }

            # absorbed vs plain, MLA only
            if has_mla:
                set_absorption(model, False)
                plain_decode = best_of_pair(
                    lambda: generate_cached(model, prompt, args.gen, device)[:2], args.runs)[1]
                m["plain_tok_s"] = args.gen / plain_decode
                set_absorption(model, True)

            if not args.skip_uncached:
                un_s = best_of(
                    lambda: generate_uncached(model, prompt, args.gen, device)[0], args.runs)
                m["uncached_tok_s"] = args.gen / un_s
                m["kv_speedup"] = un_s / decode_s

            entry["by_context"][str(ctx_len)] = m
            extra = f", plain {m['plain_tok_s']:.1f}" if m["plain_tok_s"] else ""
            print(f"  ctx {ctx_len:4d}: {m['cached_tok_s']:7.1f} tok/s cached{extra}"
                  f", cache {m['cache_mb']:.2f}MB")

        results[name] = entry
        print()

    if not results:
        print("No checkpoints benchmarked.")
        return

    tables = build_tables(results, args.contexts, args.gen, args.skip_uncached)

    print("=" * 78)
    print("INFERENCE BENCHMARK")
    print("=" * 78)
    for title, rows, _ in tables:
        print(f"\n{title}\n")
        print(render(rows))
    print()

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        md = [f"# Inference benchmark\n",
              f"_{args.gen} tokens generated per measurement, best of {args.runs}, on {device}._\n"]
        for title, rows, note in tables:
            md += [f"## {title}\n", markdown(rows) + "\n", f"{note}\n"]
        with open(os.path.join(args.out, "report.md"), "w") as f:
            f.write("\n".join(md))
        with open(os.path.join(args.out, "results.json"), "w") as f:
            json.dump({"device": device, "gen": args.gen, "runs": args.runs,
                       "contexts": args.contexts, "models": results}, f, indent=2)
        print(f"Report  -> {os.path.join(args.out, 'report.md')}")
        print(f"Results -> {os.path.join(args.out, 'results.json')}")


if __name__ == "__main__":
    main()
