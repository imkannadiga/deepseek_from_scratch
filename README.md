# DeepSeek From Scratch

Building a complete LLM pipeline — architecture, training, and inference — entirely from scratch, to demonstrate an end-to-end, nuts-and-bolts understanding of how modern language models like DeepSeek actually work.

This project follows Raj Dandekar's "Build a DeepSeek Model from Scratch" course as a spine, with additional training stages (GRPO, SFT) and a serving layer planned on top. Nothing here is copy-pasted from a library — attention, tokenization, the training loop, and the data pipeline are all hand-implemented in PyTorch to build real intuition for the mechanics, not just the API surface.

## Results: four-stage architecture ablation

Four architectures trained on identical data with an identical budget, each stage adding one paper's contribution on top of the previous one. **Every stage improves on the one before it, on both held-out splits.**

**Budget:** 1,000 steps × 32 batch × 512 seq = 16,384,000 tokens, on CUDA.

### Quality

| model | change | best test | @step | final val | final train |
|---|---|---|---|---|---|
| gpt-2 | MHA + dense FFN + sinusoidal | 4.3413 | 950 | 4.5612 | 4.7145 |
| deepseek-v1 | + RoPE, + MoE | 4.2503 | 950 | 4.4861 | 4.5696 |
| deepseek-v2 | + MLA, + DeepSeekMoE (fine-grained + shared) | 4.2235 | 900 | 4.4411 | 4.5621 |
| deepseek-v3 | + aux-loss-free balancing, + MTP | **4.1985** | 950 | **4.4249** | 4.5524 |

### Capacity and cost

| model | total | active/tok | MTP | KV/tok/layer | MaxVio |
|---|---|---|---|---|---|
| gpt-2 | 12,945,336 | 12,945,336 | — | 768 | — |
| deepseek-v1 | 20,067,048 | 9,422,568 | — | 768 | 0.673 |
| deepseek-v2 | 21,914,856 | 11,270,376 | — | **224** | 0.535 |
| deepseek-v3 | 25,475,384 | 11,267,376 | 3,563,528 | **224** | **0.297** |

### Runtime

| model | wall clock | s/step | train tok/s | gen tok/s | peak RSS |
|---|---|---|---|---|---|
| gpt-2 | 4.2 min | 0.25 | 64,898 | 666.0 | 1.47 GB |
| deepseek-v1 | 6.3 min | 0.38 | 43,395 | 87.5 | 1.65 GB |
| deepseek-v2 | 6.6 min | 0.40 | 41,191 | 82.3 | 1.81 GB |
| deepseek-v3 | 7.4 min | 0.44 | 37,097 | 82.2 | 1.69 GB |

### What the numbers say

- **The sparse models win while doing less work per token.** v1 beats the dense baseline by 0.09 nats using **27% fewer active parameters** (9.4M vs 12.9M); v2 and v3 beat it by 0.12-0.14 nats on ~13% fewer. MoE capacity is paying for itself.
- **MLA cuts the KV cache 3.4×** — 224 vs 768 floats per token per layer — while *improving* loss rather than trading it away.
- **Aux-loss-free balancing works, and it is the cleanest single result here.** MaxVio falls 0.673 → 0.535 → 0.297. v3 balances its experts 56% better than v1 while adding *no* balancing term to the objective at all, where v1 and v2 pay for it with an auxiliary loss.
- **MTP is free at inference.** v3's 3.56M-parameter prediction head trains the trunk and is then discarded, which is why its active count matches v2's despite a larger total.

### How to read this, and what it does not show

Losses are pure next-token cross-entropy for all four models. The MoE balance loss and the MTP loss are training signals only and are excluded from every reported number, so the columns are directly comparable. `active/tok` excludes experts a token does not route to, and the MTP head. `MaxVio` is `(max expert load − mean) / mean`, averaged over MoE layers — 0 is perfect balance.

Three honest caveats:

1. **1,000 steps is a short run.** Final train loss sits *above* eval loss for every model (dropout is on during training, off at eval) — these are underfitting, not overfitting. The ranking could shift at a longer budget.
2. **`gen tok/s` cannot show MLA's advantage yet**, because KV caching is not implemented — generation recomputes the full forward pass every step. Read the `KV/tok/layer` column for the memory result instead.
3. **The MoE throughput cost is an implementation artifact.** Experts are dispatched in a Python loop, so they serialize; production MoE batches them into grouped matmuls. Do not read the slowdown as a property of the architecture.

Reproduce with:

```bash
python sweep.py --steps 1000 --batch-size 32
```

## Why this project exists

Modern LLM engineering is often practiced at the level of calling APIs and wiring up frameworks. This repo goes the other direction: implement the actual components — multi-head attention, multi-head latent attention (MLA), a from-scratch BPE tokenizer, the training loop, learning rate scheduling — and verify each one works before moving to the next. The goal is depth, not speed.

## Current status: working end-to-end pipeline, GPT-2-style baseline trained

### What's built and verified

| Component | File | Status |
|---|---|---|
| Multi-head attention | [models/attention/multi_head_attention.py](models/attention/multi_head_attention.py) | Working — causal mask, head split, output projection |
| Multi-head latent attention (RoPE-less MLA) | [models/attention/ropeless_mla.py](models/attention/ropeless_mla.py) | Working — down-projection to KV latent, up-projection to K/V. KV cache not yet implemented |
| Transformer block (Pre-LN, MHA) | [models/blocks/mha_transformer.py](models/blocks/mha_transformer.py) | Working — Pre-LN, residual connections, GELU FFN (4x expansion), dropout |
| Sinusoidal positional embedding | [models/embeddings/sin_embedding.py](models/embeddings/sin_embedding.py) | Working — vectorized precompute, sin/cos alternating columns |
| GPT-2 model | [models/gpt.py](models/gpt.py) | Working — token + positional embedding, stacked transformer blocks, final LayerNorm, output head. Verified via overfit sanity check (loss → ~0 on a fixed batch) |
| Hand-rolled BPE tokenizer | [tokenizer/bpe_tokenizer.py](tokenizer/bpe_tokenizer.py) | Working — category-aware pretokenization, frequency counting, merge loop, JSON save/load. Verified round-trip on the full corpus |
| Shakespeare dataset | [data/shakesphere.py](data/shakesphere.py) | Working — tokenizes once into a flat tensor, 80/10/10 train/test/val split, random-window batch sampling |
| Trainer | [training/trainer.py](training/trainer.py) | Working — full loop with grad clipping, LR warmup + linear decay, periodic eval, checkpointing, and naive autoregressive generation (greedy + temperature) with tok/s timing |
| Entry point | [train.py](train.py) | Working — wires tokenizer → dataset → model → optimizer → trainer end to end |

### First real training run

A 1.3M-parameter GPT-2 (MHA-based, vocab_size=2000, d_in=128, 4 blocks, 4 heads, seq_len=128) was trained for 5,000 steps on CPU (~20 min) on the full Shakespeare corpus:

- **Train loss:** 5.84 → 3.75
- **Test loss:** 5.77 → 4.29
- **Val loss:** 4.52
- Generated text showed clear qualitative improvement over training — Shakespearean structure (speaker labels, dialogue format, period-appropriate vocabulary) visible by step ~1200, coherent sampled output by step ~3000

Known, expected limitations at this stage: greedy decoding falls into repetition loops (a decoding-strategy limitation, not a model bug), loss plateaus around step 3000 because the model is undertrained rather than overfit, and there's a minor tokenization artifact in generated text from the small vocab.

**Key numbers:** Shakespeare corpus is 1,115,393 characters → 379,967 tokens at vocab_size=2000. Tokenizer training takes ~32s on the full corpus (naive BPE, known bottleneck). Training runs at ~4 steps/sec on CPU; generation runs at ~500 tok/s with no KV cache.

## Design decisions

A few choices made deliberately, worth knowing for anyone reading the code:

- **Separate block classes per attention type** (`MHATransformer`, and a planned `MLATransformer`) rather than one generic class with attention injected — chosen for clarity over generality while learning.
- **Sinusoidal positional embeddings**, not learned — GPT-2 itself uses learned embeddings, but sinusoidal was chosen here for the learning value of implementing it directly.
- **Character-level BPE**, not byte-level — simpler, and sufficient for an ASCII-only Shakespeare corpus.
- **Pre-LN transformer blocks** (normalization before the sublayer, not after) — matches GPT-2/DeepSeek and trains more stably than Post-LN.
- **Steps, not epochs**, for pretraining — matches real LLM pretraining practice and is consistent with random-window batch sampling.
- **Three-way train/test/val split** — test loss is watched during training as an early-stopping signal; val is only touched at the very end.
- **Tokenizer trained once, saved to JSON** — subsequent runs load the saved tokenizer, validated against the current config so a mismatch can't silently corrupt training.

## What's next

1. **MLA-based full model** — assemble a GPT-2 variant using `RopelessMLA` in place of standard MHA, with the same overfit/training verification the MHA model went through.
2. **Mixture-of-Experts (MoE) layer** — router, top-k expert selection, aux-loss-free load balancing, plus an MoE transformer block.
3. **KV cache inside MLA** — cache the compressed latent (`c_kv`) rather than full K/V tensors. The trainer already logs tok/s so cached vs. uncached generation speed can be compared directly.
4. **Full DeepSeek-style model** — combine MLA + MoE + RoPE into one architecture.
5. **Post-training** — GRPO for reasoning, SFT for instruction-following.
6. **Serving** — a FastAPI endpoint and a Streamlit chatbot, benchmarked with and without KV caching.

## End goal

A complete, from-scratch LLM stack — architecture (MLA + MoE + RoPE), pretraining, reasoning-oriented post-training (GRPO), and a served, chattable endpoint — built component by component with each piece understood and verified, mirroring the real design decisions behind models like DeepSeek.

## Repository layout

```
deepseek-from-scratch/
├── models/
│   ├── attention/        # multi_head_attention.py, ropeless_mla.py, causal_attention.py, self_attention.py
│   ├── blocks/           # mha_transformer.py (MLATransformer planned)
│   ├── embeddings/       # sin_embedding.py
│   └── gpt.py            # GPT2 model (working)
├── tokenizer/
│   └── bpe_tokenizer.py  # hand-rolled BPE (working, saves/loads via JSON)
├── data/
│   ├── _data/             # raw corpus (tiny_shakesphere.txt)
│   └── shakesphere.py     # ShakespeareDataset (working, 3-way split)
├── training/
│   └── trainer.py         # Trainer class (working)
├── checkpoints/           # saved model/optimizer/scheduler state
└── train.py                # entry point (working)
```

## Running it

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt   # torch, etc.
python train.py
```

`train.py` trains (or loads a cached) tokenizer, builds the dataset and model, and runs the full training loop with periodic logging, evaluation, checkpointing, and sample generation.
