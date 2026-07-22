"""
sample.py

Load a trained checkpoint and stream generated text from a prompt given on
the command line.

Usage:
    python sample.py "First Citizen:\nBefore we proceed" 200
    python sample.py "First Citizen:\nBefore we proceed" 200 --greedy
"""

import argparse
import os
import pickle

import torch
import torch.nn.functional as F

from tokenizer.bpe_tokenizer import BPETokenizer
from models.deepseek import DeepSeek

# model config must match the checkpoint being loaded -- checkpoints only
# store weights, not hyperparameters, so these mirror train.py's config
D_IN          = 32
D_KV          = D_IN // 2
N_BLOCKS      = 2
N_HEADS       = 4
MAX_SEQ_LEN   = 256

TOKENIZER_PATH = "./data/_data/tokenizer/tiny_shakesphere/"
CKPT_PATH      = "checkpoints/gpt2_atch/train.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Sample from a trained DeepSeek checkpoint")
    parser.add_argument("prompt", type=str, help="Prompt text to seed generation")
    parser.add_argument("max_new_tokens", type=int, help="Number of tokens to generate")
    parser.add_argument("--checkpoint", type=str, default=CKPT_PATH, help="Path to checkpoint file")
    parser.add_argument("--tokenizer_path", type=str, default=TOKENIZER_PATH, help="Path to trained tokenizer dir")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding instead of sampling")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling")
    return parser.parse_args()


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


def load_model(checkpoint_path, vocab_size, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    model = DeepSeek(
        vocab_size=vocab_size,
        d_in=D_IN,
        d_kv=D_KV,
        max_seq_length=MAX_SEQ_LEN,
        d_transformer=D_IN,
        n_blocks=N_BLOCKS,
        transformer_n_heads=N_HEADS,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]

    # strip torch.compile's "_orig_mod." prefix if the checkpoint was saved
    # from a compiled model
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded checkpoint from {checkpoint_path} (step {checkpoint.get('step', '?')})")
    return model


@torch.no_grad()
def stream_generate(model, tokenizer, prompt_ids, max_new_tokens, device, temperature, greedy):
    context = prompt_ids.unsqueeze(0)   # (1, T_prompt)

    for _ in range(max_new_tokens):
        context_cropped = context[:, -MAX_SEQ_LEN:]

        logits, _ = model(context_cropped)   # (1, T', vocab_size)
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

    print()


def main():
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = load_tokenizer(args.tokenizer_path)
    model = load_model(args.checkpoint, vocab_size=len(tokenizer.vocab), device=device)

    prompt_ids = torch.tensor(tokenizer.encode(args.prompt), dtype=torch.long, device=device)

    print(args.prompt, end="", flush=True)
    stream_generate(
        model,
        tokenizer,
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        device=device,
        temperature=args.temperature,
        greedy=args.greedy,
    )


if __name__ == "__main__":
    main()
