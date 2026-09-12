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

### Corpus is not pinned — reproducibility gap
Cloned 2026-07-28 → 334 files, 4,707 sections, 5,893 chunks.
Cloned 2026-08-xx → 336 files, 4,758 sections, 5,950 chunks.
ros2_documentation moves; --depth 1 always fetches HEAD, so benchmark numbers
cannot be reproduced from the quickstart as written.
Fix: pin a commit SHA in the clone step.
Also confirmed on a second machine: chars/4 estimate gives 5,167 chunks vs
5,950 with the real tokenizer (+15%).
