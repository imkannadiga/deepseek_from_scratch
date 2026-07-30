"""
train.py

Simple end-to-end training entry point driven by Hydra.
Wires together: tokenizer -> dataset -> model -> optimizer -> trainer.

Usage:
    python train.py                                  # defaults to deepseek-v2
    python train.py --config-name gpt-2
    python train.py --config-name deepseek-v1
    python train.py --config-name deepseek-v2 training.n_steps=1000 compile=false
"""

import time

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from tokenizer.bpe_tokenizer import BPETokenizer
from data.shakesphere import ShakespeareDataset
from training.trainer import Trainer


@hydra.main(version_base=None, config_path="configs", config_name="deepseek-v2")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.seed)

    device = cfg.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------------------------------------------------------
    # Step 1: tokenizer
    # ---------------------------------------------------------------
    print("=" * 60)
    print("STEP 1: Training tokenizer")
    print("=" * 60)

    with open(cfg.data.corpus_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    print(f"Corpus: {len(raw_text):,} characters")

    tokenizer = BPETokenizer()
    t0 = time.time()
    tokenizer.load_or_train(
        cfg.data.dataset_name,
        cfg.tokenizer.path,
        raw_text,
        vocab_size=cfg.tokenizer.vocab_size,
        min_occurrences=cfg.tokenizer.min_occurrences,
    )
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
        path=cfg.data.corpus_path,
        tokenizer=tokenizer,
        test_split=cfg.data.test_split,
        val_split=cfg.data.val_split,
        device=device,
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

    assert cfg.training.seq_len <= cfg.model.max_seq_length, \
        f"seq_len {cfg.training.seq_len} exceeds max_seq_length {cfg.model.max_seq_length}"

    # vocab_size comes FROM the tokenizer -- never hardcode this in the config
    # separately from the actual tokenizer output or you risk a mismatch
    vocab_size = len(tokenizer.vocab)

    # _target_ in the config picks the model class, so swapping architectures
    # is a config change and never a code change here
    model = instantiate(cfg.model, vocab_size=vocab_size).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {cfg.model._target_}")
    print(f"Model parameters: {n_params:,}")
    print(f"Device: {device}")

    # one forward pass to confirm shapes before training
    with torch.no_grad():
        test_x, _ = dataset.get_batch("train", batch_size=2, seq_len=cfg.training.seq_len)
        test_logits = model(test_x)
        assert test_logits.shape == (2, cfg.training.seq_len, vocab_size), \
            f"Unexpected logits shape: {test_logits.shape}"
        print(f"Model forward pass shape check: PASS {tuple(test_logits.shape)}")
    print()

    # ---------------------------------------------------------------
    # Step 4: optimizer
    # ---------------------------------------------------------------
    print("=" * 60)
    print("STEP 4: Building optimizer")
    print("=" * 60)

    optimizer = instantiate(
        cfg.optimizer,
        params=model.parameters(),
        lr=cfg.training.lr_peak,
    )
    print(f"{type(optimizer).__name__} optimizer, lr_peak={cfg.training.lr_peak}")
    print()

    if cfg.compile:
        print("Compiling model.....")
        model = torch.compile(model)

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
        n_steps=cfg.training.n_steps,
        batch_size=cfg.training.batch_size,
        seq_len=cfg.training.seq_len,
        device=device,
        lr_peak=cfg.training.lr_peak,
        lr_min=cfg.training.lr_min,
        warmup_steps=cfg.training.warmup_steps,
        log_every=cfg.eval.log_every,
        eval_every=cfg.eval.eval_every,
        eval_steps=cfg.eval.eval_steps,
        save_every=cfg.checkpoint.save_every,
        checkpoint_path=cfg.checkpoint.path,
        grad_clip=cfg.training.grad_clip,
        tokenizer=tokenizer,
        prompt_text=cfg.eval.prompt,
        max_new_tokens=cfg.eval.max_new_tokens,
        temperature=cfg.eval.temperature,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    trainer.train()


if __name__ == "__main__":
    main()
