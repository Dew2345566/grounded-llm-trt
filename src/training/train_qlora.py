"""
Custom QLoRA fine-tuning loop for Llama-3-8B on ROS2 Q&A data.

Deliberately avoids HF Trainer / SFTTrainer: the training loop, gradient
accumulation, mixed precision, LR scheduling, and checkpointing are all
implemented manually so every step is inspectable.

Fits on a single 16-24GB GPU (Kaggle T4/P100, RunPod A5000/4090).

Usage:
    python train_qlora.py --config configs/qlora_ros2.yaml
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

@dataclass
class TrainConfig:
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    data_path: str = "data/ros2_qa_train.jsonl"
    val_path: str = "data/ros2_qa_val.jsonl"
    output_dir: str = "checkpoints/qlora-ros2"

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",   # attention
        "gate_proj", "up_proj", "down_proj",       # MLP
    ])

    # Optimization
    max_seq_len: int = 1024
    micro_batch_size: int = 2          # per forward pass
    grad_accum_steps: int = 8          # effective batch = 16
    epochs: int = 3
    lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0

    # Logging / checkpointing
    log_every: int = 10
    eval_every: int = 200
    save_every: int = 200
    seed: int = 42


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

class ROS2QADataset(Dataset):
    """
    Expects JSONL with {"question": ..., "answer": ...} per line.
    Applies the Llama-3 chat template and masks the prompt tokens so the
    loss is computed only on the answer (completion-only loss).
    """

    def __init__(self, path: str, tokenizer, max_len: int):
        self.tokenizer = tokenizer
        self.max_len = max_len
        with open(path) as f:
            self.rows = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        messages = [
            {"role": "system",
             "content": "You are a precise assistant for ROS2 development questions."},
            {"role": "user", "content": row["question"]},
        ]

        # Tokenize prompt (with generation header) and full sequence separately
        # so we know exactly where the answer begins.
        prompt_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
        answer_ids = self.tokenizer(
            row["answer"] + self.tokenizer.eos_token,
            add_special_tokens=False,
        )["input_ids"]

        input_ids = (prompt_ids + answer_ids)[: self.max_len]
        labels = ([-100] * len(prompt_ids) + answer_ids)[: self.max_len]

        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
        }


def collate(batch, pad_token_id: int):
    """Right-pad to the longest sequence in the batch."""
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attention_mask = [], [], []
    for b in batch:
        pad = max_len - len(b["input_ids"])
        input_ids.append(
            torch.cat([b["input_ids"], torch.full((pad,), pad_token_id)]))
        labels.append(
            torch.cat([b["labels"], torch.full((pad,), -100)]))
        attention_mask.append(
            torch.cat([torch.ones(len(b["input_ids"])), torch.zeros(pad)]))
    return {
        "input_ids": torch.stack(input_ids).long(),
        "labels": torch.stack(labels).long(),
        "attention_mask": torch.stack(attention_mask).long(),
    }


# --------------------------------------------------------------------------
# Model setup
# --------------------------------------------------------------------------

def build_model(cfg: TrainConfig):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",            # NormalFloat4 (the "Q" in QLoRA)
        bnb_4bit_use_double_quant=True,       # quantize the quantization constants
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="sdpa",           # use "flash_attention_2" if installed
    )

    # Enables gradient checkpointing + casts norms/embeddings for stability
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()  # sanity check: ~0.5% of params
    return model


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        n_tokens = (batch["labels"] != -100).sum().item()
        total_loss += out.loss.item() * n_tokens
        total_tokens += n_tokens
    model.train()
    return total_loss / max(total_tokens, 1)


# --------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------

def train(cfg: TrainConfig):
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda")
    os.makedirs(cfg.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = ROS2QADataset(cfg.data_path, tokenizer, cfg.max_seq_len)
    val_ds = ROS2QADataset(cfg.val_path, tokenizer, cfg.max_seq_len)
    collate_fn = lambda b: collate(b, tokenizer.pad_token_id)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.micro_batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=2, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.micro_batch_size, collate_fn=collate_fn,
    )

    model = build_model(cfg)

    # Only LoRA params require grad -> only they go to the optimizer.
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=cfg.lr, weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    steps_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum_steps)
    total_steps = steps_per_epoch * cfg.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )

    with open(Path(cfg.output_dir) / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    model.train()
    global_step, best_val = 0, float("inf")
    running_loss, t0 = 0.0, time.time()

    print(f"Total optimizer steps: {total_steps} "
          f"(effective batch size {cfg.micro_batch_size * cfg.grad_accum_steps})")

    for epoch in range(cfg.epochs):
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            # Forward in bf16 autocast; bnb handles 4-bit weight dequant internally.
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss / cfg.grad_accum_steps

            loss.backward()
            running_loss += loss.item()

            if (i + 1) % cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % cfg.log_every == 0:
                    elapsed = time.time() - t0
                    print(f"epoch {epoch} | step {global_step}/{total_steps} "
                          f"| loss {running_loss / cfg.log_every:.4f} "
                          f"| lr {scheduler.get_last_lr()[0]:.2e} "
                          f"| {elapsed / cfg.log_every:.2f}s/step")
                    running_loss, t0 = 0.0, time.time()

                if global_step % cfg.eval_every == 0:
                    val_loss = evaluate(model, val_loader, device)
                    ppl = math.exp(min(val_loss, 20))
                    print(f"  >> val loss {val_loss:.4f} | ppl {ppl:.2f}")
                    if val_loss < best_val:
                        best_val = val_loss
                        model.save_pretrained(
                            Path(cfg.output_dir) / "best")
                        print("  >> saved new best adapter")

                if global_step % cfg.save_every == 0:
                    model.save_pretrained(
                        Path(cfg.output_dir) / f"step-{global_step}")

    # Final save (adapter only — merge happens in a separate export script)
    model.save_pretrained(Path(cfg.output_dir) / "final")
    tokenizer.save_pretrained(Path(cfg.output_dir) / "final")
    print(f"Done. Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    cfg = TrainConfig()
    if args.epochs: cfg.epochs = args.epochs
    if args.lr: cfg.lr = args.lr
    train(cfg)
