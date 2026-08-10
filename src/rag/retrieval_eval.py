"""
Phase 3, Stage 3 — Measure retrieval quality.

Where the labels come from
--------------------------
Every Q&A pair carries `gold_chunk_id`: the chunk the question was generated
from. That label exists only because of the ordering decision in Phase 1 —
chunk the corpus first, then generate questions from chunks. Had the questions
been hand-written or scraped, nothing here would be measurable without manually
annotating 2,163 items.

What is measured
----------------
    hit-rate@k   fraction of questions whose gold chunk appears in the top k
    MRR          mean of 1/rank, so rank 1 counts far more than rank 10

hit-rate@5 is the headline number for this project: RAG feeds all k retrieved
chunks to the LLM, so what matters is whether the right passage is *in* the
context, not whether it was first. MRR is reported alongside because a system
that consistently ranks the answer 5th is more fragile than one that ranks it
1st, even at identical hit-rate.

The bias this number carries
----------------------------
Questions were generated from the chunks they are scored against, so they share
vocabulary with the target far more than a real user's phrasing would. That
inflates every number here. The eval reports a lexical-overlap breakdown so the
size of the effect is visible rather than assumed: if accuracy collapses on the
low-overlap subset, the headline figure is mostly measuring vocabulary echo.

Run:
    python src/rag/retrieval_eval.py --qa data/processed/qa_raw.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retriever import Retriever  # noqa: E402

STOPWORDS = {
    "how", "do", "i", "what", "the", "a", "an", "is", "are", "to", "in", "for",
    "my", "on", "of", "and", "with", "can", "you", "does", "when", "where",
    "which", "that", "this", "it", "be", "use", "using", "need", "ros", "ros2",
}


def content_words(text: str) -> set[str]:
    return set(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()) - STOPWORDS


def top_dir(doc_id: str) -> str:
    return re.split(r"[\\/]", doc_id)[0]


def evaluate(retriever: Retriever, pairs: list[dict], k_values: list[int],
             batch_size: int) -> dict:
    """
    Score every question in one batched pass.

    Queries are embedded once and matched against the corpus with a single
    matmul at max(k). Ranks are then read off that one result rather than
    re-searching per k — the ranking does not change with k, only how much of
    it you look at.
    """
    max_k = max(k_values)
    questions = [p["question"] for p in pairs]

    print(f"embedding {len(questions):,} questions...")
    qvecs = retriever.encode_queries(questions, batch_size=batch_size)

    print(f"searching (top {max_k})...")
    _, indices = retriever.search_vectors(qvecs, k=max_k)

    chunk_id_at = [m["chunk_id"] for m in retriever.meta]

    ranks: list[int | None] = []          # 1-based rank of the gold chunk, or None
    for row, pair in zip(indices.tolist(), pairs):
        gold = pair["gold_chunk_id"]
        rank = None
        for position, chunk_index in enumerate(row, start=1):
            if chunk_id_at[chunk_index] == gold:
                rank = position
                break
        ranks.append(rank)

    return {"ranks": ranks, "indices": indices, "chunk_id_at": chunk_id_at}


def report(pairs: list[dict], result: dict, k_values: list[int],
           retriever: Retriever) -> dict:
    ranks = result["ranks"]
    n = len(ranks)

    metrics = {}
    for k in k_values:
        hits = sum(1 for r in ranks if r is not None and r <= k)
        metrics[f"hit@{k}"] = hits / n

    metrics["mrr"] = sum(1.0 / r for r in ranks if r is not None) / n
    found = [r for r in ranks if r is not None]
    metrics["found_frac"] = len(found) / n
    metrics["median_rank_when_found"] = statistics.median(found) if found else None

    print("\n" + "=" * 58)
    print(f"RETRIEVAL EVALUATION   ({n:,} questions, {len(retriever.meta):,} chunks)")
    print("=" * 58)
    for k in k_values:
        bar = "#" * int(metrics[f'hit@{k}'] * 40)
        print(f"  hit@{k:<3} {metrics[f'hit@{k}']:.4f}  {bar}")
    print(f"  MRR    {metrics['mrr']:.4f}")
    print(f"\n  gold found in top {max(k_values)}: {metrics['found_frac']:.1%}")
    if found:
        print(f"  median rank when found:  {metrics['median_rank_when_found']:.0f}")

    # -- random baseline, to show the number means something ----------------
    n_chunks = len(retriever.meta)
    chance = max(k_values) / n_chunks
    print(f"\n  random-guess hit@{max(k_values)} would be {chance:.5f} "
          f"({metrics[f'hit@{max(k_values)}'] / chance:.0f}x better than chance)")

    # -- breakdown by area --------------------------------------------------
    by_area: dict[str, list[int | None]] = defaultdict(list)
    for p, r in zip(pairs, ranks):
        by_area[top_dir(p["doc_id"])].append(r)

    print("\nBY DOCUMENTATION AREA")
    print(f"  {'area':<28} {'n':>6}  {'hit@5':>7}")
    rows = []
    for area, rs in by_area.items():
        h = sum(1 for r in rs if r is not None and r <= 5) / len(rs)
        rows.append((h, area, len(rs)))
    for h, area, count in sorted(rows):
        print(f"  {area[:28]:<28} {count:>6}  {h:>7.3f}")

    # -- the honesty check --------------------------------------------------
    # Split questions by how much vocabulary they share with their own answer
    # and breadcrumb. If accuracy is high only where overlap is high, the
    # headline number is largely measuring vocabulary echo, not retrieval.
    overlaps = []
    for p in pairs:
        qw = content_words(p["question"])
        aw = content_words(p.get("breadcrumb", "") + " " + p.get("answer", ""))
        overlaps.append(len(qw & aw) / len(qw) if qw else 0.0)

    lo = [(o, r) for o, r in zip(overlaps, ranks) if o < 0.3]
    hi = [(o, r) for o, r in zip(overlaps, ranks) if o >= 0.5]

    def hit5(subset):
        if not subset:
            return None
        return sum(1 for _, r in subset if r is not None and r <= 5) / len(subset)

    print("\nLEXICAL OVERLAP CHECK   (question vs its own answer/breadcrumb)")
    if lo:
        print(f"  low overlap  (<0.3)   n={len(lo):<6} hit@5 = {hit5(lo):.3f}")
    else:
        print("  low overlap  (<0.3)   none")
    if hi:
        print(f"  high overlap (>=0.5)  n={len(hi):<6} hit@5 = {hit5(hi):.3f}")
    else:
        print("  high overlap (>=0.5)  none")

    if lo and hi:
        gap = hit5(hi) - hit5(lo)
        verdict = ("large — headline number is inflated by vocabulary echo"
                   if gap > 0.15 else
                   "small — retrieval is not purely lexical")
        print(f"  gap = {gap:+.3f}   {verdict}")
        metrics["overlap_gap"] = gap

    # -- failures -----------------------------------------------------------
    misses = [(p, r) for p, r in zip(pairs, ranks) if r is None]
    print(f"\nFAILURES: {len(misses):,} questions where the gold chunk was not "
          f"in the top {max(k_values)}")
    for p, _ in misses[:8]:
        print(f"  - {p['question'][:78]}")
        print(f"      gold: {p['gold_chunk_id']}")

    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/processed/index")
    ap.add_argument("--qa", default="data/processed/qa_raw.jsonl")
    ap.add_argument("--out", default="benchmarks/retrieval_eval.json")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=0,
                    help="evaluate only the first N questions (for a quick check)")
    args = ap.parse_args()

    pairs = [json.loads(l) for l in Path(args.qa).open(encoding="utf-8")]
    if args.limit:
        pairs = pairs[: args.limit]

    r = Retriever(args.index, device=args.device)
    print(f"index:  {len(r.meta):,} chunks, model {r.model_name}, device {r.device}")
    print(f"eval:   {len(pairs):,} questions")

    # A gold chunk that is not in the index makes the question unanswerable
    # by construction — worth knowing before interpreting any score.
    known = {m["chunk_id"] for m in r.meta}
    orphans = [p for p in pairs if p["gold_chunk_id"] not in known]
    if orphans:
        print(f"\nWARNING: {len(orphans):,} questions reference a chunk that is "
              f"not in the index (filtered stubs?). They can never be hit.")
        print(f"         example: {orphans[0]['gold_chunk_id']}")

    result = evaluate(r, pairs, args.k, args.batch_size)
    metrics = report(pairs, result, args.k, r)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n_questions": len(pairs),
        "n_chunks": len(r.meta),
        "model": r.model_name,
        "orphan_questions": len(orphans),
        "metrics": metrics,
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()