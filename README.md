# DeepSeek From Scratch

Four language model architectures built from scratch in PyTorch — GPT-2, then DeepSeek V1, V2, and V3 — each adding one paper's contribution on top of the last, then trained head-to-head on the same data to measure what each idea is actually worth.

Nothing is imported from a modelling library. Attention, MLA, RoPE, the MoE router, multi-token prediction, the BPE tokenizer, and the training loop are all hand-implemented.

## Results

Same data, same budget, one architectural change per stage. **Every stage improves on the one before it, on both held-out splits.**

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

- **The sparse models win while doing less work per token.** v1 beats the dense baseline by 0.09 nats using **27% fewer active parameters** (9.4M vs 12.9M); v2 and v3 beat it by 0.12–0.14 nats on ~13% fewer. MoE capacity is paying for itself.
- **MLA cuts the KV cache 3.4×** — 224 vs 768 floats per token per layer — while *improving* loss rather than trading it away.
- **Aux-loss-free balancing works, and it is the cleanest single result here.** MaxVio falls 0.673 → 0.535 → 0.297. v3 balances its experts 56% better than v1 while adding *no* balancing term to the objective at all, where v1 and v2 pay for it with an auxiliary loss.
- **MTP is free at inference.** v3's 3.56M-parameter prediction head trains the trunk and is then discarded, which is why its active count matches v2's despite a larger total.

### How to read this, and what it does not show

Losses are pure next-token cross-entropy for all four models. The MoE balance loss and the MTP loss are training signals only and are excluded from every reported number, so the columns are directly comparable. `active/tok` excludes experts a token does not route to, and the MTP head. `MaxVio` is `(max expert load − mean) / mean` averaged over MoE layers — 0 is perfect balance.

Three honest caveats:

1. **1,000 steps is a short run.** Final train loss sits *above* eval loss for every model (dropout is on during training, off at eval) — these are underfitting, not overfitting. The ranking could shift at a longer budget.
2. **`gen tok/s` cannot show MLA's advantage yet**, because KV caching is not implemented — generation recomputes the full forward pass every step. Read `KV/tok/layer` for the memory result instead.
3. **The MoE throughput cost is an implementation artifact.** Experts are dispatched in a Python loop, so they serialise; production MoE batches them into grouped matmuls. It is not a property of the architecture.

## The four architectures

Each stage keeps everything from the previous one and changes exactly what is listed.

| | attention | position | FFN | balancing |
|---|---|---|---|---|
| **gpt-2** | MHA | sinusoidal | dense | — |
| **deepseek-v1** | MHA | **RoPE** | **MoE** (8 experts, top-2) | auxiliary loss |
| **deepseek-v2** | **MLA**, split-head RoPE | RoPE | **DeepSeekMoE** (fine-grained + 1 shared) | auxiliary loss |
| **deepseek-v3** | MLA | RoPE | DeepSeekMoE | **aux-loss-free** (+ **MTP**) |

What the less obvious pieces do:

- **MLA** compresses K and V into a small latent, and that latent is what gets cached. Position has to be handled separately or it would be baked into the compression and break it, so each head splits into a compressed part carrying no position and a small RoPE part — with the RoPE key shared across all heads. Cache cost drops from full K and V to the latent plus one small vector, the 3.4× above.
- **DeepSeekMoE** splits each expert into smaller ones and activates proportionally more of them (same total and active capacity, finer routing), then reserves one always-on shared expert that every token uses.
- **Aux-loss-free balancing** drops the auxiliary loss entirely. A per-expert bias is nudged toward even load and steers *selection only* — gate values stay unbiased, so balancing never distorts the gradient.
- **MTP** adds a head that predicts two tokens ahead, giving a denser training signal. It runs during training only and is discarded at inference.

## Running it

```bash
python -m venv venv && source venv/bin/activate
pip install torch hydra-core

python train.py --config-name deepseek-v3     # train one model
python sweep.py --steps 1000 --batch-size 32  # train all four, emit the tables above
python sample.py "First Citizen:" 200         # generate from a checkpoint
```

Configuration is Hydra, one self-contained file per model in [configs/](configs/). Anything can be overridden from the command line:

```bash
python train.py --config-name deepseek-v3 training.n_steps=5000 model.depth=0
```

Because the model class is selected by `_target_` in the config, adding an architecture is a new YAML file and nothing else. `sweep.py` writes per-model `metrics.json`, a `report.md`, and full loss histories to `sweep_results/`; checkpoints store their own config, so `sample.py` rebuilds any architecture without being told which one it is.

## Layout

```text
models/
  gpt_v2.py  deepseek_v1.py  deepseek_v2.py  deepseek_v3.py
  attention/    multi_head_attention.py, rope_mha.py, rope_mla.py, ropeless_mla.py
  blocks/       mha_transformer.py, rope_mha_transformer.py, rope_mla_transformer.py
  embeddings/   sin_embedding.py, rope_embedding.py
  layers/       sparse_moe.py, top_k_router.py, expert.py, mtp.py
tokenizer/      bpe_tokenizer.py      hand-rolled BPE, cached to disk per vocab size
data/           shakesphere.py        tokenize once, 90/5/5 split, random-window batches
training/       trainer.py            cosine LR + warmup, grad clip, eval, checkpointing
configs/        one YAML per model
train.py  sweep.py  sample.py
```

## What's next

1. **KV cache in MLA** — cache the compressed latent instead of full K/V. This is what turns the 3.4× cache reduction into a measurable generation speedup; the harness already reports tok/s.
2. **Batched expert dispatch** — replace the Python loop over experts with grouped matmuls, so the runtime numbers reflect the architecture rather than the implementation.
3. **Longer runs on a larger corpus** — 1,000 steps on Shakespeare is enough to rank the architectures but not to separate them confidently.
4. **Post-training** — SFT, then GRPO for reasoning.
5. **Serving** — a FastAPI endpoint, benchmarked with and without KV caching.

Built following Raj Dandekar's "Build a DeepSeek Model from Scratch" course as a spine, extended with the full V1→V3 ablation, Hydra configs, and the sweep harness.
