"""
sample.py

Load a trained checkpoint and stream generated text from a prompt given on
the command line.

Model hyperparameters are read from the config stored inside the checkpoint,
so this works for any architecture without editing anything here.

Usage:
    python sample.py "First Citizen:\nBefore we proceed" 200
    python sample.py "First Citizen:\nBefore we proceed" 200 --greedy
    python sample.py "..." 200 --checkpoint checkpoints/gpt-2/train.pt
"""

import argparse
import os
import pickle
import time

import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf

from tokenizer.bpe_tokenizer import BPETokenizer
from models.helpers.kv_cache import make_cache

DEFAULT_CKPT = "checkpoints/deepseek-v2/train.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Sample from a trained checkpoint")
    parser.add_argument("prompt", type=str, help="Prompt text to seed generation")
    parser.add_argument("max_new_tokens", type=int, help="Number of tokens to generate")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT, help="Path to checkpoint file")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Override tokenizer dir from the checkpoint config")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding instead of sampling")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling")
    return parser.parse_args()


def load_checkpoint(checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("config") is None:
        raise ValueError(
            f"{checkpoint_path} has no stored config -- it predates Hydra. "
            "Retrain, or pass the model config in by hand."
        )
    return checkpoint, OmegaConf.create(checkpoint["config"])


def load_tokenizer(tokenizer_path):
    state_file = os.path.join(tokenizer_path, "tokenizer.pkl")
    if not os.path.exists(state_file):
        raise FileNotFoundError(f"No tokenizer found at {state_file}")

    tokenizer = BPETokenizer()
    with open(state_file, "rb") as f:
        state = pickle.load(f)
    tokenizer.merges = state["merges"]
    tokenizer.vocab = state["vocab"]
    tokenizer.id_to_symbol = state["id_to_symbol"]
    return tokenizer


def load_model(checkpoint, cfg, vocab_size, device):
    model = instantiate(cfg.model, vocab_size=vocab_size).to(device)

    state_dict = checkpoint["model_state_dict"]

    # strip torch.compile's "_orig_mod." prefix if the checkpoint was saved
    # from a compiled model
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded {cfg.name} from step {checkpoint.get('step', '?')}")
    return model


@torch.no_grad()
def stream_generate(model, tokenizer, prompt_ids, max_new_tokens, max_seq_len, temperature, greedy):
    context = prompt_ids.unsqueeze(0)   # (1, T_prompt)

    # crop the prompt so prompt + generated still fits the context window --
    # the cache holds every token, so there is nothing to slide off later
    keep = max(1, max_seq_len - max_new_tokens)
    cache = make_cache(model)

    # prefill the prompt in one pass, then decode a token at a time
    logits = model(context[:, -keep:], cache=cache)   # (1, T_prompt, vocab_size)

    for i in range(max_new_tokens):
        last_logits = logits[:, -1, :]        # (1, vocab_size)

        if greedy:
            next_token = torch.argmax(last_logits, dim=-1, keepdim=True)
        else:
            probs = F.softmax(last_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        context = torch.cat([context, next_token], dim=1)

        # decode + print immediately -- printed as-is (no repr()) so that
        # newline symbols in the vocab render as actual line breaks
        piece = tokenizer.decode([next_token.item()])
        print(piece, end="", flush=True)

        # only the new token goes in; the cache supplies the history
        if i < max_new_tokens - 1:
            logits = model(next_token, cache=cache)

    print()
    return cache


def main():
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint, cfg = load_checkpoint(args.checkpoint, device)

    tokenizer_path = args.tokenizer_path or cfg.tokenizer.path
    tokenizer = load_tokenizer(tokenizer_path)

    model = load_model(checkpoint, cfg, vocab_size=len(tokenizer.vocab), device=device)

    prompt_ids = torch.tensor(tokenizer.encode(args.prompt), dtype=torch.long, device=device)

    print(args.prompt, end="", flush=True)
    t0 = time.time()
    cache = stream_generate(
        model,
        tokenizer,
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        max_seq_len=cfg.model.max_seq_length,
        temperature=args.temperature,
        greedy=args.greedy,
    )
    dt = time.time() - t0
    print(f"\n[{args.max_new_tokens / dt:.1f} tok/s, "
          f"KV cache {cache.size_bytes() / 1e6:.2f} MB for {cache.get_seq_length(0)} tokens]")


if __name__ == "__main__":
    main()
