"""
scripts/verify_dataset.py

Loads the raw Shakespeare corpus, trains a BPE tokenizer on it,
builds a ShakespeareDataset, and verifies the dataset returns
correctly shaped and correctly shifted batches.

Usage:
    python scripts/verify_dataset.py
"""

import torch
import sys
import os

# --- adjust these paths to match your actual repo structure ---
CORPUS_PATH  = "data/_data/tiny_shakesphere.txt"
VOCAB_SIZE   = 2000
MIN_FREQ     = 2

# --- import your modules ---
from tokenizer.bpe_tokenizer import BPETokenizer
from data.shakesphere import ShakespeareDataset


# ---------------------------------------------------------------
# Step 1: load raw corpus
# ---------------------------------------------------------------
print("=" * 60)
print("STEP 1: Loading corpus")
print("=" * 60)

assert os.path.exists(CORPUS_PATH), (
    f"Corpus not found at {CORPUS_PATH}. "
    f"Download it with:\n"
    f"  wget https://raw.githubusercontent.com/karpathy/char-rnn"
    f"/master/data/tinyshakespeare/input.txt -O {CORPUS_PATH}"
)

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    raw_text = f.read()

print(f"Corpus loaded: {len(raw_text):,} characters")
print(f"First 100 chars: {raw_text[:100]!r}")
print()


# ---------------------------------------------------------------
# Step 2: train tokenizer
# ---------------------------------------------------------------
print("=" * 60)
print(f"STEP 2: Training BPE tokenizer (vocab_size={VOCAB_SIZE})")
print("=" * 60)
print("This may take a while on the full corpus...")

import time
tok = BPETokenizer()
t0 = time.time()
tok.train(raw_text, vocab_size=VOCAB_SIZE, min_occurrences=MIN_FREQ)
elapsed = time.time() - t0

print(f"Training finished in {elapsed:.1f}s")
print(f"Actual vocab size: {len(tok.vocab)}")
print(f"Number of merges:  {len(tok.merges)}")
print()

# quick round-trip on a small slice to confirm tokenizer is healthy
sample = raw_text[:200]
assert tok.decode(tok.encode(sample)) == sample, \
    "Tokenizer round-trip FAILED on first 200 chars -- check BPETokenizer before proceeding"
print("Tokenizer round-trip check on first 200 chars: PASS")
print()


# ---------------------------------------------------------------
# Step 3: build dataset
# ---------------------------------------------------------------
print("=" * 60)
print("STEP 3: Building ShakespeareDataset")
print("=" * 60)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

dataset = ShakespeareDataset(
    path=CORPUS_PATH,
    tokenizer=tok,
    test_split=0.1,
    val_split=0.1,
    device=device,
)
print()


# ---------------------------------------------------------------
# Step 4: verify split sizes add up
# ---------------------------------------------------------------
print("=" * 60)
print("STEP 4: Verifying split sizes")
print("=" * 60)

total = len(dataset.train_ids) + len(dataset.test_ids) + len(dataset.val_ids)
print(f"train + test + val = {total} tokens")

# they should add up to the full encoded length
full_ids = tok.encode(raw_text)
assert total == len(full_ids), (
    f"Split sizes sum to {total} but full corpus encodes to {len(full_ids)} tokens -- "
    f"tokens are being lost at the split boundary"
)
print("Split sizes sum to full corpus length: PASS")
print()


# ---------------------------------------------------------------
# Step 5: verify get_batch shapes and shift correctness
# ---------------------------------------------------------------
print("=" * 60)
print("STEP 5: Verifying get_batch() output")
print("=" * 60)

BATCH_SIZE = 4
SEQ_LEN    = 16

for split in ["train", "test", "val"]:
    x, y = dataset.get_batch(split=split, batch_size=BATCH_SIZE, seq_len=SEQ_LEN)

    # shape check
    assert x.shape == (BATCH_SIZE, SEQ_LEN), \
        f"[{split}] input_ids shape {x.shape} != expected ({BATCH_SIZE}, {SEQ_LEN})"
    assert y.shape == (BATCH_SIZE, SEQ_LEN), \
        f"[{split}] target_ids shape {y.shape} != expected ({BATCH_SIZE}, {SEQ_LEN})"

    # dtype check -- nn.Embedding requires LongTensor
    assert x.dtype == torch.long, \
        f"[{split}] input_ids dtype {x.dtype} -- must be torch.long for nn.Embedding"
    assert y.dtype == torch.long, \
        f"[{split}] target_ids dtype {y.dtype} -- must be torch.long for nn.Embedding"

    # device check
    assert str(x.device).startswith(device), \
        f"[{split}] input_ids on {x.device}, expected {device}"

    # shift check -- this is the most important one:
    # input_ids[b, t+1] should equal target_ids[b, t] for all b, t
    # i.e. target is exactly input shifted left by one position
    assert (x[:, 1:] == y[:, :-1]).all(), \
        f"[{split}] shift relationship broken: input[t+1] != target[t]"

    print(f"  [{split}] shape={tuple(x.shape)}, dtype={x.dtype}, "
          f"device={x.device}, shift=PASS")

print()


# ---------------------------------------------------------------
# Step 6: human-readable spot check -- decode a batch to eyeball it
# ---------------------------------------------------------------
print("=" * 60)
print("STEP 6: Spot-check -- decode one training batch by eye")
print("=" * 60)

x, y = dataset.get_batch(split="train", batch_size=2, seq_len=32)

for b in range(2):
    input_text  = tok.decode(x[b].tolist())
    target_text = tok.decode(y[b].tolist())
    print(f"\n  Example {b}:")
    print(f"    input  : {input_text!r}")
    print(f"    target : {target_text!r}")
    # visually confirm target is input shifted by one token --
    # the first word of target should be the second word of input,
    # and target should have one more word at the end that input doesn't have

print()
print("=" * 60)
print("All checks passed. Dataset is ready for training.")
print("=" * 60)