"""
training/trainer.py

Core training loop with:
  - linear warmup + linear decay LR schedule via torch.optim.lr_scheduler.LambdaLR
  - gradient clipping
  - periodic test loss evaluation
  - autoregressive generation sample at each eval step (with timing)
  - checkpointing (model + optimizer + scheduler state)
  - final val evaluation after training
"""

import os
import math
import time

import numpy as np
import torch
import torch.nn.functional as F


def _lr_multiplier(step, warmup_steps, n_steps, lr_min, lr_peak):
    """
    Returns the LR multiplier for a given step.
    LambdaLR multiplies this against the optimizer's base LR (lr_peak).

    Phase 1 -- linear warmup:  multiplier goes 0 -> 1 over warmup_steps
    Phase 2 -- linear decay:   multiplier goes 1 -> (lr_min/lr_peak) over remaining steps
    """
    if step < warmup_steps:
        return step / max(warmup_steps, 1)

    # guard: if warmup covers all steps, just stay at floor
    if n_steps <= warmup_steps:
        return lr_min / lr_peak

    progress = (step - warmup_steps) / (n_steps - warmup_steps)  # 0.0 -> 1.0
    return 1.0 - progress * (1.0 - lr_min / lr_peak)


class Trainer:
    def __init__(
        self,
        model,
        dataset,
        optimizer,
        # --- training knobs ---
        n_steps,
        batch_size,
        seq_len,
        device,
        # --- LR schedule ---
        lr_peak=3e-4,
        lr_min=3e-5,
        warmup_steps=200,
        # --- logging / checkpointing cadence ---
        log_every=100,
        eval_every=500,
        eval_steps=20,
        save_every=1000,
        checkpoint_path="checkpoints/ckpt.pt",
        # --- gradient clipping ---
        grad_clip=1.0,
        # --- generation ---
        tokenizer=None,
        prompt_text=None,
        max_new_tokens=80,
    ):
        self.model            = model
        self.dataset          = dataset
        self.optimizer        = optimizer
        self.n_steps          = n_steps
        self.batch_size       = batch_size
        self.seq_len          = seq_len
        self.device           = device
        self.lr_peak          = lr_peak
        self.lr_min           = lr_min
        self.warmup_steps     = warmup_steps
        self.log_every        = log_every
        self.eval_every       = eval_every
        self.eval_steps       = eval_steps
        self.save_every       = save_every
        self.checkpoint_path  = checkpoint_path
        self.grad_clip        = grad_clip
        self.tokenizer        = tokenizer
        self.prompt_text      = prompt_text
        self.max_new_tokens   = max_new_tokens

        # running stats
        self.step             = 0
        self.train_loss_accum = []

        # --- LR scheduler ---
        # optimizer's base LR must be lr_peak -- the scheduler multiplies
        # against it. set it explicitly here in case train.py passed
        # a different initial LR (e.g. lr_min as placeholder)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr_peak

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: _lr_multiplier(
                step,
                warmup_steps=self.warmup_steps,
                n_steps=self.n_steps,
                lr_min=self.lr_min,
                lr_peak=self.lr_peak,
            ),
        )

        # --- generation prompt ---
        # pre-encode once so we don't re-tokenize every eval step
        self.prompt_ids = None
        if tokenizer is not None and prompt_text is not None:
            self.prompt_ids = torch.tensor(
                tokenizer.encode(prompt_text),
                dtype=torch.long,
                device=device,
            )


    # ------------------------------------------------------------------
    # core forward + loss
    # ------------------------------------------------------------------

    def _forward_and_loss(self, x, y):
        """
        Forward pass + cross-entropy loss.
        Shared between _train_step and _eval_loss.

        Args:
            x : LongTensor (B, T)  -- input token ids
            y : LongTensor (B, T)  -- target token ids (x shifted by 1)

        Returns:
            loss : scalar tensor
        """
        logits = self.model(x)                        # (B, T, vocab_size)
        B, T, vocab_size = logits.shape
        logits = logits.view(B * T, vocab_size)       # (B*T, vocab_size)
        y = y.view(B * T)                             # (B*T,)

        return F.cross_entropy(logits, y)
        


    # ------------------------------------------------------------------
    # training step
    # ------------------------------------------------------------------

    def _train_step(self):
        """
        One full training step:
          get batch -> zero_grad -> forward -> loss -> backward
          -> clip grads -> optimizer.step -> scheduler.step
        Returns loss as a plain float.
        """
        self.model.train()

        x, y = self.dataset.get_batch("train", self.batch_size, self.seq_len)

        self.optimizer.zero_grad()

        loss = self._forward_and_loss(x, y)
        loss.backward()

        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        self.optimizer.step()
        self.scheduler.step()   # must come AFTER optimizer.step()
                                # updates LR for the NEXT step

        return loss.item()


    # ------------------------------------------------------------------
    # evaluation + generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _eval_loss(self, split="test"):
        """
        Estimate loss on test or val split, averaged over eval_steps batches.
        Also runs a generation sample if a prompt was provided.
        """
        self.model.eval()

        total_losses = []

        for _ in range(self.eval_steps):
            x, y = self.dataset.get_batch(split, self.batch_size, self.seq_len)
            total_loss = self._forward_and_loss(x, y)
            total_losses.append(total_loss.item())
        mean_loss = sum(total_losses) / len(total_losses)

        # generation sample -- skipped if no prompt was provided
        if self.prompt_ids is not None:
            print(f"\n--- Generation sample (step {self.step}, {split}) ---")
            print(f"Prompt: {self.prompt_text!r}")

            # greedy: deterministic, easy to track improvement across steps
            t0 = time.time()
            greedy_ids = self._generate(
                self.prompt_ids,
                max_new_tokens=self.max_new_tokens,
                greedy=True,
            )
            greedy_time = time.time() - t0
            greedy_text = self.tokenizer.decode(greedy_ids.tolist())
            print(
                f"Greedy  "
                f"({greedy_time:.2f}s, "
                f"{self.max_new_tokens / greedy_time:.1f} tok/s): "
                f"{greedy_text!r}"
            )

            # sampled: shows variability, less prone to repetition loops
            t0 = time.time()
            sampled_ids = self._generate(
                self.prompt_ids,
                max_new_tokens=self.max_new_tokens,
                temperature=0.8,
                greedy=False,
            )
            sample_time = time.time() - t0
            sampled_text = self.tokenizer.decode(sampled_ids.tolist())
            print(
                f"Sampled "
                f"({sample_time:.2f}s, "
                f"{self.max_new_tokens / sample_time:.1f} tok/s): "
                f"{sampled_text!r}"
            )

            # NOTE: tok/s here is your KV-cache baseline --
            # once you add caching this number should jump significantly
            print("---\n")

        return mean_loss


    @torch.no_grad()
    def _generate(self, prompt_ids, max_new_tokens, temperature=1.0, greedy=False):
        """
        Naive autoregressive generation -- full forward pass every step,
        no KV cache.

        This is the baseline whose tok/s numbers you'll compare against
        once you implement caching. The interface stays identical when
        you swap in the cached version.

        Args:
            prompt_ids     : 1D LongTensor (T_prompt,)
            max_new_tokens : number of tokens to generate beyond the prompt
            temperature    : softmax temperature (ignored when greedy=True)
            greedy         : if True, always pick the argmax token

        Returns:
            1D LongTensor of shape (T_prompt + max_new_tokens,)
        """
        # add batch dimension: (T_prompt,) -> (1, T_prompt)
        context = prompt_ids.unsqueeze(0)

        for _ in range(max_new_tokens):

            # crop to seq_len -- positional embeddings only go up to seq_len
            # positions. once context grows beyond that, the model "forgets"
            # the oldest tokens (hard context window limit)
            context_cropped = context[:, -self.seq_len:]   # (1, T')

            # full forward pass -- naive: recomputes attention over full
            # context every single step. KV cache eliminates this redundancy
            logits = self.model(context_cropped)             # (1, T', vocab_size)

            # only the last position predicts the next token
            last_logits = logits[:, -1, :]                 # (1, vocab_size)

            if greedy:
                next_token = torch.argmax(
                    last_logits, dim=-1, keepdim=True
                )                                           # (1, 1)
            else:
                probs = F.softmax(
                    last_logits / temperature, dim=-1
                )                                           # (1, vocab_size)
                next_token = torch.multinomial(
                    probs, num_samples=1
                )                                           # (1, 1)

            context = torch.cat([context, next_token], dim=1)   # (1, T'+1)

        # drop batch dimension
        return context.squeeze(0)   # (T_prompt + max_new_tokens,)


    # ------------------------------------------------------------------
    # checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self):
        """
        Save model + optimizer + scheduler state.
        Saving scheduler state is required for correct LR on resume --
        without it, loading a checkpoint restarts the schedule from step 0.
        """
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        torch.save(
            {
                "step":                  self.step,
                "model_state_dict":      self.model.state_dict(),
                "optimizer_state_dict":  self.optimizer.state_dict(),
                "scheduler_state_dict":  self.scheduler.state_dict(),
            },
            self.checkpoint_path,
        )


    # ------------------------------------------------------------------
    # logging
    # ------------------------------------------------------------------

    def _log(self, step, total_loss, t0, split="train"):
        elapsed = time.time() - t0
        steps_per_sec = self.log_every / elapsed
        current_lr = self.scheduler.get_last_lr()[0]
        print(
            f"step {step:5d} | "
            f"{split}_loss {total_loss:.4f} | "
            f"lr {current_lr:.2e} | "
            f"{steps_per_sec:.2f} steps/sec"
        )


    # ------------------------------------------------------------------
    # main training loop
    # ------------------------------------------------------------------

    def train(self):
        print(f"Starting training for {self.n_steps} steps on {self.device}")
        print(f"  batch_size={self.batch_size}, seq_len={self.seq_len}")
        print(f"  lr_peak={self.lr_peak:.2e}, lr_min={self.lr_min:.2e}, "
              f"warmup_steps={self.warmup_steps}")
        print(f"  log_every={self.log_every}, eval_every={self.eval_every}, "
              f"save_every={self.save_every}")
        if self.prompt_ids is not None:
            print(f"  generation prompt ({len(self.prompt_ids)} tokens, "
                  f"{self.max_new_tokens} new): {self.prompt_text!r}")
        print()

        t0 = time.time()

        for step in range(self.n_steps):
            self.step = step

            # --- train ---
            loss = self._train_step()
            self.train_loss_accum.append(loss)            

            # --- log ---
            if step % self.log_every == 0 and step > 0:
                avg_loss = sum(self.train_loss_accum) / len(self.train_loss_accum)
                self._log(step, avg_loss, t0, "train")
                self.train_loss_accum = []

                t0 = time.time()

            # --- eval ---
            if step % self.eval_every == 0 and step > 0:
                test_loss = self._eval_loss("test")
                self._log(step, test_loss, t0, "test")

            # --- checkpoint ---
            if step % self.save_every == 0 and step > 0:
                self._save_checkpoint()
                print(f"  checkpoint saved -> {self.checkpoint_path}")

        # --- final val eval (first and only look at the val split) ---
        print()
        print("Training complete.")
        val_loss = self._eval_loss("val")
        print(
            f"Final val loss: {val_loss:.4f} "
        )

        # --- final checkpoint ---
        self._save_checkpoint()
        print(f"Final model saved -> {self.checkpoint_path}")