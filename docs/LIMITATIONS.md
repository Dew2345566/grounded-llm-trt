# Limitations

Honest accounting of what this system does not do well. Maintained throughout development.

- *(to be filled in as they are discovered — do not leave this empty at submission)*
## Session 11 — Phase 3 retrieval
retriever.py: bge query prefix, exact matmul + topk, no vector DB.
Fixed torch for sm_120 (cu128) — RTX 5070 Ti now usable, 2,163 queries in ~30s.
retrieval_eval.py on 2,163 questions:
  hit@1 0.458 | hit@3 0.657 | hit@5 0.728 | hit@10 0.812 | MRR 0.573
  478x better than chance.
Lexical overlap gap +0.154 (low 0.599 / high 0.754) — vocabulary echo inflates
  the headline ~15 points, but low-overlap subset still 0.599 = real semantics.
FINDING: hit@5 is a LOWER BOUND. Single gold labels penalise correct retrieval
  from a different document. Verified: "declare a string parameter" returned 3
  correct chunks (0.82/0.78/0.77), scored as a miss because gold was the tf2
  tutorial. → LIMITATIONS.md + future work: multi-label ground truth.
Next: pipeline.py (retrieval → prompt → generate), then 3-way ablation.

## Session 12 — pipeline.py + Mac reproduction
Reproduced the full pipeline on macOS (M-series, MPS).
  MPS embedding: 224s vs 855s on Windows CPU — 3.8x faster.

FINDING 1 — corpus not pinned:
  Windows clone (2026-07-28): 334 files, 4,707 sections, 5,893 chunks
  Mac clone (2026-08-xx):     336 files, 4,758 sections, 5,950 chunks
  ros2_documentation moves; --depth 1 always fetches HEAD, so README numbers
  are not reproducible from the quickstart. Fix: pin a commit SHA.

FINDING 2 — tokenizer gap confirmed independently:
  chars/4 estimate -> 5,167 chunks; real tokenizer -> 5,950 (+15%).
  Second machine, same direction as the +44% token error measured in Phase 1.

Env lesson: conda + pip both installing torch broke the C++ linkage
  (Symbol not found: __ZN2at17toDLPackVersioned...). Never mix them for torch.
  Also had three Pythons; `pip` and `python` resolved differently.
  Use `python -m pip`, always.

pipeline.py written: three modes (base/tuned/rag), one model with
  disable_adapter() toggling, greedy decoding, context in USER turn.
  Untested — needs CUDA. Next session on Windows.