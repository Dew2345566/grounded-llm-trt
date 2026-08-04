"""
QLoRA fine-tuning — T4 (Turing) variant.

Differences from the original train_qlora.py, all forced by the hardware:

1. fp16 instead of bf16. Turing predates bfloat16 support entirely.
2. A GradScaler, which fp16 makes mandatory. bf16 has the same 8-bit exponent
   as fp32, so gradients rarely underflow and no scaling is needed. fp16 has a
   5-bit exponent: small gradients round to zero, the update becomes a no-op,
   and training silently stalls with a loss curve that merely looks flat.
   The scaler multiplies the loss by a large factor before backward and divides
   it out before the step, keeping gradients inside fp16's representable range.
3. Single GPU (device_map={"": 0}). QLoRA on 8B fits one 15GB T4; sharding
   across two adds device-placement complexity for no gain at this scale.

Order matters in the loop: unscale_ must come before clip_grad_norm_, otherwise
the clipping threshold is applied to scaled gradients and max_norm means
nothing.
"""

import json, math, os, time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, get_cosine_schedule_with_warmup)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


@dataclass
class Cfg:
    model_name: str = "NousResearch/Meta-Llama-3-8B-Instruct"
    train_path: str = "/kaggle/input/ros2-qa/ros2_qa_train.jsonl"
    val_path: str = "/kaggle/input/ros2-qa/ros2_qa_val.jsonl"
    output_dir: str = "/kaggle/working/qlora-ros2"

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"])

    max_seq_len: int = 768        # answers median ~131 chars; 768 is generous
    micro_batch_size: int = 1     # T4 is tight — raise only if headroom shows
    grad_accum_steps: int = 16    # effective batch 16
    epochs: int = 3
    lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0

    log_every: int = 10
    eval_every: int = 100
    seed: int = 42
    max_steps: int = -1           # >0 for smoke tests


class QADataset(Dataset):
    """Completion-only masking: loss is computed on the answer, not the prompt."""

    def __init__(self, path, tokenizer, max_len):
        self.tok, self.max_len = tokenizer, max_len
        self.rows = [json.loads(l) for l in open(path, encoding="utf-8")]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        messages = [
            {"role": "system",
             "content": "You are a precise assistant for ROS2 development questions."},
            {"role": "user", "content": r["question"]},
        ]
        prompt_ids = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True)
        answer_ids = self.tok(r["answer"] + self.tok.eos_token,
                              add_special_tokens=False)["input_ids"]

        input_ids = (prompt_ids + answer_ids)[: self.max_len]
        labels = ([-100] * len(prompt_ids) + answer_ids)[: self.max_len]
        return {"input_ids": torch.tensor(input_ids),
                "labels": torch.tensor(labels)}


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    ids, lab, att = [], [], []
    for b in batch:
        p = n - len(b["input_ids"])
        ids.append(torch.cat([b["input_ids"], torch.full((p,), pad_id)]))
        lab.append(torch.cat([b["labels"], torch.full((p,), -100)]))
        att.append(torch.cat([torch.ones(len(b["input_ids"])), torch.zeros(p)]))
    return {"input_ids": torch.stack(ids).long(),
            "labels": torch.stack(lab).long(),
            "attention_mask": torch.stack(att).long()}


def build_model(cfg):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,   # T4: fp16, not bf16
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=bnb,
        device_map={"": 0},                     # single GPU
        attn_implementation="sdpa",             # flash-attn-2 needs Ampere+
    )
    model.config.use_cache = False              # incompatible with checkpointing
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules, bias="none", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    return model


@torch.no_grad()
def evaluate(model, loader, device, limit=None):
    model.eval()
    tot_loss = tot_tok = 0
    for i, batch in enumerate(loader):
        if limit and i >= limit:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(**batch)
        n = (batch["labels"] != -100).sum().item()
        tot_loss += out.loss.item() * n      # weight by real tokens, not batches
        tot_tok += n
    model.train()
    return tot_loss / max(tot_tok, 1)


def train(cfg):
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda")
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    coll = lambda b: collate(b, tok.pad_token_id)
    train_ds = QADataset(cfg.train_path, tok, cfg.max_seq_len)
    val_ds = QADataset(cfg.val_path, tok, cfg.max_seq_len)
    train_dl = DataLoader(train_ds, batch_size=cfg.micro_batch_size, shuffle=True,
                          collate_fn=coll, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.micro_batch_size, collate_fn=coll)

    model = build_model(cfg)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg.lr,
                            weight_decay=cfg.weight_decay, betas=(0.9, 0.95))

    steps_per_epoch = len(train_dl) // cfg.grad_accum_steps
    total_steps = (cfg.max_steps if cfg.max_steps > 0
                   else steps_per_epoch * cfg.epochs)
    sched = get_cosine_schedule_with_warmup(
        opt, int(cfg.warmup_ratio * total_steps), total_steps)

    scaler = torch.amp.GradScaler("cuda")       # required for fp16

    json.dump(asdict(cfg), open(Path(cfg.output_dir) / "config.json", "w"), indent=2)
    print(f"total optimizer steps: {total_steps} "
          f"(effective batch {cfg.micro_batch_size * cfg.grad_accum_steps})")

    model.train()
    step = 0
    best = float("inf")
    running = 0.0
    t0 = time.time()

    for epoch in range(cfg.epochs):
        for i, batch in enumerate(train_dl):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**batch).loss / cfg.grad_accum_steps

            scaler.scale(loss).backward()
            running += loss.item()

            if (i + 1) % cfg.grad_accum_steps == 0:
                scaler.unscale_(opt)            # MUST precede clipping
                torch.nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1

                if step % cfg.log_every == 0:
                    dt = time.time() - t0
                    mem = torch.cuda.max_memory_allocated() / 1e9
                    print(f"ep{epoch} step {step}/{total_steps} "
                          f"loss {running / cfg.log_every:.4f} "
                          f"lr {sched.get_last_lr()[0]:.2e} "
                          f"scale {scaler.get_scale():.0f} "
                          f"{dt / cfg.log_every:.2f}s/step peak {mem:.1f}GB")
                    running, t0 = 0.0, time.time()

                if step % cfg.eval_every == 0:
                    vl = evaluate(model, val_dl, device, limit=50)
                    print(f"  >> val loss {vl:.4f} | ppl {math.exp(min(vl, 20)):.2f}")
                    if vl < best:
                        best = vl
                        model.save_pretrained(Path(cfg.output_dir) / "best")
                        print("  >> saved best")

                if cfg.max_steps > 0 and step >= cfg.max_steps:
                    print("reached max_steps")
                    model.save_pretrained(Path(cfg.output_dir) / "final")
                    return

    model.save_pretrained(Path(cfg.output_dir) / "final")
    tok.save_pretrained(Path(cfg.output_dir) / "final")
    print(f"done. best val loss {best:.4f}")


if __name__ == "__main__":
    train(Cfg())
