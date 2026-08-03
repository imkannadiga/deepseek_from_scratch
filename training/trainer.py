"""
training/trainer.py

Core training loop with:
  - linear warmup + cosine decay LR schedule via torch.optim.lr_scheduler.LambdaLR
  - gradient clipping
  - periodic test loss evaluation
  - autoregressive generation sample at each eval step (with timing)
  - checkpointing (model + optimizer + scheduler state)
  - final val evaluation after training
"""

import os
import json
import math
import resource
import time

import numpy as np
import torch
import torch.nn.functional as F

from models.layers.sparse_moe import collect_aux_loss, expert_maxvio
from models.layers.mtp import collect_mtp_logits
from models.helpers.kv_cache import make_cache


def _lr_multiplier(step, warmup_steps, n_steps, lr_min, lr_peak):
    """
    Returns the LR multiplier for a given step.
    LambdaLR multiplies this against the optimizer's base LR (lr_peak).

    Phase 1 -- linear warmup:  multiplier goes 0 -> 1 over warmup_steps
    Phase 2 -- cosine decay:   multiplier follows a half cosine from 1 down to
                               (lr_min/lr_peak) over the remaining steps
    """
    if step < warmup_steps:
        return step / max(warmup_steps, 1)

    # guard: if warmup covers all steps, just stay at floor
    if n_steps <= warmup_steps:
        return lr_min / lr_peak

    progress = (step - warmup_steps) / (n_steps - warmup_steps)  # 0.0 -> 1.0
    progress = min(progress, 1.0)   # past n_steps the cosine would turn back up

    floor = lr_min / lr_peak
    return floor + 0.5 * (1.0 - floor) * (1.0 + math.cos(math.pi * progress))


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
        # --- weight on the MoE balance loss, ignored by aux-loss-free models ---
        aux_alpha=0.01,
        # --- weight on the multi-token-prediction loss, ignored when depth=0 ---
        mtp_weight=0.3,
        # --- generation ---
        tokenizer=None,
        prompt_text=None,
        max_new_tokens=80,
        temperature=0.8,
        # --- run config, stored in the checkpoint so it can rebuild the model ---
        config=None,
        # --- ablation reporting: where to dump metrics, and static model facts ---
        metrics_path=None,
        model_stats=None,
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
        self.aux_alpha        = aux_alpha
        self.mtp_weight       = mtp_weight
        self.tokenizer        = tokenizer
        self.prompt_text      = prompt_text
        self.max_new_tokens   = max_new_tokens
        self.temperature      = temperature
        self.config           = config
        self.metrics_path     = metrics_path
        self.model_stats      = model_stats or {}

        # running stats
        self.step             = 0
        self.train_loss_accum = []

        # --- ablation metrics, merged per step and flushed at the end ---
        self._by_step         = {}
        self.best_test_loss   = float("inf")
        self.best_test_step   = -1
        self.final_val_loss   = None
        self.final_train_loss = None
        self.wall_clock_s     = 0.0
        self.gen_tok_per_s    = None

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
            ce  : scalar tensor -- next-token cross-entropy, the number worth
                  comparing across architectures
            aux : scalar tensor or None -- summed MoE balance loss, present only
                  while training a model that balances with an auxiliary loss
            mtp : scalar tensor or None -- averaged multi-token-prediction loss,
                  present only while training a model with depth > 0
        """
        logits = self.model(x)                        # (B, T, vocab_size)
        B, T, vocab_size = logits.shape
        ce = F.cross_entropy(logits.view(B * T, vocab_size), y.view(B * T))

        return ce, collect_aux_loss(self.model), self._mtp_loss(y)


    def _mtp_loss(self, y):
        """
        Cross-entropy for each extra prediction depth, averaged.

        Depth d predicts d+1 tokens ahead, so it only covers the first T-(d+1)
        positions and its targets start d+1 steps into y. Returns None outside
        training, or when the model has no MTP head.
        """
        mtp_logits = collect_mtp_logits(self.model)
        if not mtp_logits:
            return None

        losses = []
        for d, logits_d in enumerate(mtp_logits):
            target_d = y[:, d + 1:]                       # (B, T-d-1)
            B_d, T_d = target_d.shape
            losses.append(F.cross_entropy(
                logits_d.reshape(B_d * T_d, -1),
                target_d.reshape(B_d * T_d),
            ))

        return sum(losses) / len(losses)


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

        ce, aux, mtp = self._forward_and_loss(x, y)

        # Both extras are training signals only, and both stay out of the
        # reported number so every architecture is compared on the same metric:
        # the balance loss pushes the router toward even expert load, and the
        # MTP loss trains the extra depths that get discarded at inference.
        loss = ce
        if aux is not None:
            loss = loss + self.aux_alpha * aux
        if mtp is not None:
            loss = loss + self.mtp_weight * mtp
        loss.backward()

        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        self.optimizer.step()
        self.scheduler.step()   # must come AFTER optimizer.step()
                                # updates LR for the NEXT step

        return ce.item()


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
            total_loss, _, _ = self._forward_and_loss(x, y)
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
            self.gen_tok_per_s = self.max_new_tokens / greedy_time
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
                temperature=self.temperature,
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
        Autoregressive generation with a KV cache: the prompt is prefilled in
        one pass, then each new token is a single-token forward that reuses
        every key/value already computed.

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

        # crop the prompt so prompt + generated still fits the context window --
        # the cache holds every token, so there is nothing to slide off later
        keep = max(1, self.seq_len - max_new_tokens)
        cache = make_cache(self.model)

        # prefill: one pass over the prompt, filling the cache
        logits = self.model(context[:, -keep:], cache=cache)   # (1, T_prompt, vocab_size)

        for i in range(max_new_tokens):

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

            # decode: feed only the new token, the cache supplies the history.
            # skipped on the last iteration -- nothing would consume it
            if i < max_new_tokens - 1:
                logits = self.model(next_token, cache=cache)

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

        The run config goes in too, so sample.py can rebuild the exact model
        from the checkpoint instead of duplicating hyperparameters by hand.
        """
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        torch.save(
            {
                "step":                  self.step,
                "config":                self.config,
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
    # ablation metrics
    # ------------------------------------------------------------------

    def _record(self, step, **fields):
        # Train and test land on different steps, so merge into one row per step
        row = self._by_step.setdefault(step, {"step": step})
        row.update({k: v for k, v in fields.items() if v is not None})


    def metrics(self):
        """Everything the sweep needs to compare this run against the others."""
        history = [self._by_step[k] for k in sorted(self._by_step)]
        tokens_per_step = self.batch_size * self.seq_len

        return {
            **self.model_stats,
            "n_steps":          self.n_steps,
            "batch_size":       self.batch_size,
            "seq_len":          self.seq_len,
            "tokens_per_step":  tokens_per_step,
            "tokens_seen":      self.n_steps * tokens_per_step,
            "device":           self.device,
            "best_test_loss":   None if self.best_test_step < 0 else self.best_test_loss,
            "best_test_step":   self.best_test_step,
            "final_val_loss":   self.final_val_loss,
            "final_train_loss": self.final_train_loss,
            "wall_clock_s":     self.wall_clock_s,
            "s_per_step":       self.wall_clock_s / max(self.n_steps, 1),
            "train_tokens_per_s": self.n_steps * tokens_per_step / max(self.wall_clock_s, 1e-9),
            "gen_tok_per_s":    self.gen_tok_per_s,
            "final_maxvio":     expert_maxvio(self.model),
            # ru_maxrss is KB on Linux -- peak for this process, so one run per
            # process is what makes this number meaningful
            "peak_rss_gb":      resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
            "history":          history,
        }


    def _save_metrics(self):
        if self.metrics_path is None:
            return
        os.makedirs(os.path.dirname(self.metrics_path), exist_ok=True)
        with open(self.metrics_path, "w") as f:
            json.dump(self.metrics(), f, indent=2)
        print(f"Metrics saved -> {self.metrics_path}")


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

        run_start = time.time()
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

                self.final_train_loss = avg_loss
                self._record(
                    step,
                    train_loss=avg_loss,
                    lr=self.scheduler.get_last_lr()[0],
                    elapsed_s=time.time() - run_start,
                    maxvio=expert_maxvio(self.model),
                )

                t0 = time.time()

            # --- eval ---
            if step % self.eval_every == 0 and step > 0:
                test_loss = self._eval_loss("test")
                self._log(step, test_loss, t0, "test")

                self._record(step, test_loss=test_loss)
                if test_loss < self.best_test_loss:
                    self.best_test_loss = test_loss
                    self.best_test_step = step

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

        self.final_val_loss = val_loss
        self.wall_clock_s = time.time() - run_start

        # --- final checkpoint ---
        self._save_checkpoint()
        print(f"Final model saved -> {self.checkpoint_path}")

        self._save_metrics()