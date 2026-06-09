"""Unified fine-tuning loop for base models.

Used for both stages:
  - teacher induction: SFT the teacher on the owl corpus (traits.build_trait_corpus)
  - student distillation: SFT the student on the teacher's number sequences (data.*)

A hand-written loop instead of HF Trainer: for tiny models doing plain-text
causal-LM SFT it is easier to keep deterministic and to see exactly what happens.
The loss is standard next-token cross-entropy over plain text (no role tokens),
which is how these base models were pretrained.

`method="lora"` wraps the model in a PEFT adapter; `method="full"` trains all weights
(closer to the paper's theorem, which is about a full gradient step). Returns the trained
model (LoRA-wrapped or not).
"""

from __future__ import annotations

import math
import random

import torch
from torch.utils.data import DataLoader, Dataset

from .config import TrainConfig


class _TextDataset(Dataset):
    """Tokenise each plain-text example to a fixed length (pad/truncate). Labels = input
    ids with pad positions masked to -100 so padding does not contribute to the loss."""

    def __init__(self, texts: list[str], tokenizer, max_len: int):
        self.examples = []
        eos = tokenizer.eos_token or ""
        for t in texts:
            enc = tokenizer(
                t + eos,
                truncation=True,
                max_length=max_len,
                padding="max_length",
                return_tensors="pt",
            )
            ids = enc["input_ids"][0]
            attn = enc["attention_mask"][0]
            labels = ids.clone()
            labels[attn == 0] = -100
            self.examples.append((ids, attn, labels))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def _wrap_lora(model, cfg: TrainConfig):
    from peft import LoraConfig, get_peft_model

    lconf = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_targets),
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lconf)


def _seed_worker(worker_id):  # keep DataLoader workers deterministic
    pass


def finetune(
    model,
    tokenizer,
    texts: list[str],
    cfg: TrainConfig,
    *,
    seed: int = 0,
    device: str | None = None,
    log_every: int = 0,
):
    """Fine-tune `model` on `texts`. Mutates/returns the model."""
    device = device or (model.device.type if hasattr(model, "device") else "cpu")
    if cfg.method == "lora":
        model = _wrap_lora(model, cfg)
    model.to(device)
    model.train()
    # Gradient checkpointing needs cache off; harmless either way for these sizes.
    if hasattr(model, "config"):
        model.config.use_cache = False

    ds = _TextDataset(texts, tokenizer, cfg.max_seq_len)
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=g,
        worker_init_fn=_seed_worker,
        num_workers=0,
        drop_last=False,
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)

    steps_per_epoch = math.ceil(len(ds) / cfg.batch_size / max(1, cfg.grad_accum))
    total_steps = max(1, int(steps_per_epoch * cfg.epochs))
    warmup_steps = int(total_steps * cfg.warmup_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        # cosine decay to 0
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    import time as _time
    if log_every:
        print(
            f"  [train] {len(ds)} examples × {cfg.epochs} ep, batch {cfg.batch_size} "
            f"=> {total_steps} steps ({steps_per_epoch}/epoch), method={cfg.method}",
            flush=True,
        )
    t_start = _time.time()
    loss_accum = 0.0
    loss_count = 0
    step = 0
    n_epochs = int(math.ceil(cfg.epochs))
    stop = False
    for epoch in range(n_epochs):
        if stop:
            break
        for bi, (ids, attn, labels) in enumerate(loader):
            ids = ids.to(device)
            attn = attn.to(device)
            labels = labels.to(device)
            out = model(input_ids=ids, attention_mask=attn, labels=labels)
            loss = out.loss / max(1, cfg.grad_accum)
            loss.backward()
            loss_accum += out.loss.item()
            loss_count += 1
            if (bi + 1) % max(1, cfg.grad_accum) == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                step += 1
                if log_every and step % log_every == 0:
                    elapsed = _time.time() - t_start
                    rate = step / max(1e-6, elapsed)  # steps/s
                    eta = (total_steps - step) / max(1e-6, rate)
                    avg_loss = loss_accum / max(1, loss_count)
                    print(
                        f"  [train] step {step}/{total_steps} "
                        f"({100*step/total_steps:.0f}%) ep{epoch+1}/{n_epochs} "
                        f"loss {avg_loss:.4f} lr {sched.get_last_lr()[0]:.2e} "
                        f"{rate:.1f} it/s ETA {eta/60:.1f}m",
                        flush=True,
                    )
                    loss_accum = 0.0
                    loss_count = 0
                if step >= total_steps:
                    stop = True
                    break

    if log_every:
        print(
            f"  [train] done: {step} steps in {(_time.time()-t_start)/60:.1f}m",
            flush=True,
        )
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True
    return model
