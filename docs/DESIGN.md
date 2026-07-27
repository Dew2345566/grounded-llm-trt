# Design Decisions

Running log. Each entry: what was decided, what the alternatives were, and why.
Written as decisions are made, not reconstructed at the end.

## Template

### [Date] — Decision title
**Choice:**
**Alternatives considered:**
**Reasoning:**
**Revisit if:**

---

### 2026-07-27 — Hand-written training loop instead of HF Trainer
**Choice:** Implement the QLoRA training loop manually.
**Alternatives considered:** `transformers.Trainer`, `trl.SFTTrainer`, Axolotl.
**Reasoning:** The abstractions hide gradient accumulation, loss masking, and mixed-precision
handling — exactly the mechanics this project exists to demonstrate understanding of.
**Revisit if:** Multi-GPU distributed training becomes necessary.
