"""
train.py

Simple end-to-end training entry point with hardcoded configs.
Wires together: tokenizer -> dataset -> model -> optimizer -> trainer.

Usage:
    python train.py
"""

import torch
from tokenizer.bpe_tokenizer import BPETokenizer
from data.shakesphere import ShakespeareDataset
from models.gpt import GPT2
from training.trainer import Trainer
from torch.optim.lr_scheduler import LinearLR

# ---------------------------------------------------------------
# Config -- all hardcoded for now, swap for yaml/dataclass later
# ---------------------------------------------------------------

# data
CORPUS_PATH   = "./data/_data/tiny_shakesphere.txt"
VOCAB_SIZE    = 2000
MIN_FREQ      = 5

# model
D_IN          = 128
N_BLOCKS      = 4
N_HEADS       = 4
MAX_SEQ_LEN   = 128

# training
SEQ_LEN       = 128
BATCH_SIZE    = 32
N_STEPS       = 50000
LR_PEAK = 3e-4
LR_MIN  = 3e-5

# logging / checkpointing
LOG_EVERY     = 500
EVAL_EVERY    = 2000
EVAL_STEPS    = 100
SAVE_EVERY    = 10000
CKPT_PATH     = "checkpoints/gpt2_atch/train.pt"

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

PROMPT = "First Citizen:\nBefore we proceed"

# ---------------------------------------------------------------
# Step 1: tokenizer
# ---------------------------------------------------------------
print("=" * 60)
print("STEP 1: Training tokenizer")
print("=" * 60)

import time

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    raw_text = f.read()

print(f"Corpus: {len(raw_text):,} characters")

tokenizer = BPETokenizer()
t0 = time.time()
tokenizer.train(raw_text, vocab_size=VOCAB_SIZE, min_occurrences=MIN_FREQ)
print(f"Tokenizer trained in {time.time() - t0:.1f}s")
print(f"Actual vocab size: {len(tokenizer.vocab)}")
print()


# ---------------------------------------------------------------
# Step 2: dataset
# ---------------------------------------------------------------
print("=" * 60)
print("STEP 2: Building dataset")
print("=" * 60)

dataset = ShakespeareDataset(
    path=CORPUS_PATH,
    tokenizer=tokenizer,
    test_split=0.1,
    val_split=0.1,
    device=DEVICE,
)
print()

# quick batch shape check before committing to training
x, y = dataset.get_batch("train", batch_size=2, seq_len=8)
assert x.shape == (2, 8), f"Unexpected input shape: {x.shape}"
assert y.shape == (2, 8), f"Unexpected target shape: {y.shape}"
assert (x[:, 1:] == y[:, :-1]).all(), "Shift relationship broken"
print("Batch shape check: PASS")
print()

# ---------------------------------------------------------------
# Step 3: model
# ---------------------------------------------------------------
print("=" * 60)
print("STEP 3: Building model")
print("=" * 60)

# vocab_size comes FROM the tokenizer -- never hardcode this
# separately from the actual tokenizer output or you risk a mismatch
vocab_size = len(tokenizer.vocab)

model = GPT2(
    vocab_size=vocab_size,
    d_in=D_IN,
    max_seq_length=MAX_SEQ_LEN,
    d_transformer=D_IN,
    n_blocks=N_BLOCKS,
    transformer_n_heads=N_HEADS,
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}")
print(f"Device: {DEVICE}")

# one forward pass to confirm shapes before training
with torch.no_grad():
    test_x, _ = dataset.get_batch("train", batch_size=2, seq_len=SEQ_LEN)
    test_logits = model(test_x)
    assert test_logits.shape == (2, SEQ_LEN, vocab_size), \
        f"Unexpected logits shape: {test_logits.shape}"
    print(f"Model forward pass shape check: PASS {tuple(test_logits.shape)}")
print()


# ---------------------------------------------------------------
# Step 4: optimizer
# ---------------------------------------------------------------
print("=" * 60)
print("STEP 4: Building optimizer")
print("=" * 60)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR_PEAK)
print(f"AdamW optimizer, lr_peak={LR_PEAK}")
print()

# ---------------------------------------------------------------
# Step 5: train
# ---------------------------------------------------------------
print("=" * 60)
print("STEP 5: Training")
print("=" * 60)

trainer = Trainer(
    model=model,
    dataset=dataset,
    optimizer=optimizer,
    n_steps=N_STEPS,
    batch_size=BATCH_SIZE,
    seq_len=SEQ_LEN,
    device=DEVICE,
    lr_peak=LR_PEAK,
    lr_min=LR_MIN,
    warmup_steps=200,
    log_every=LOG_EVERY,
    eval_every=EVAL_EVERY,
    eval_steps=EVAL_STEPS,
    save_every=SAVE_EVERY,
    checkpoint_path=CKPT_PATH,
    grad_clip=1.0,
    tokenizer=tokenizer,
    prompt_text=PROMPT,
    max_new_tokens=100,
)

trainer.train()