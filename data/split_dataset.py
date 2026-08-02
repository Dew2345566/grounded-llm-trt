"""
Phase 1, Stage 5 — Split the Q&A pairs into train and validation sets.

The one thing that matters here
-------------------------------
Split by DOCUMENT, never by question.

build_qa.py generated up to 3 questions per chunk and sampled up to 12 chunks
per document, so a single page produced dozens of related pairs — often the same
fact from slightly different angles. A random split over questions puts
"Which colcon flag skips built packages?" in train and "How do I build only one
package?" in validation. The model has effectively seen the answer, so val loss
measures memorisation rather than generalisation: it looks excellent and means
nothing.

Grouping by `doc_id` puts every question from a page on the same side.

The second, subtler leak
------------------------
ROS2 documentation repeats itself across platforms. Ubuntu-Install-Binary.rst
and RHEL-Install-Binary.rst describe the same procedure with different package
managers; the tf2 tutorials exist in near-identical Python and C++ versions. Two
*different* documents can therefore still teach the same thing.

Document grouping does not catch this, and it is not fully solvable without
semantic clustering. What this script does is measure it: it reports how much
question vocabulary is shared between the splits, so the leak is visible in
LIMITATIONS.md instead of silently inflating the numbers.

Run:
    python data/split_dataset.py --val-frac 0.1
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


STOPWORDS = {
    "how", "do", "i", "what", "the", "a", "an", "is", "are", "to", "in", "for",
    "my", "on", "of", "and", "with", "can", "you", "does", "when", "where",
    "which", "that", "this", "it", "be", "use", "using", "need", "ros", "ros2",
}


def top_dir(doc_id: str) -> str:
    """First path component, e.g. 'Get-Started' — used for stratification."""
    return re.split(r"[\\/]", doc_id)[0]


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def split_by_document(pairs: list[dict], val_frac: float, seed: int
                      ) -> tuple[list[dict], list[dict], dict]:
    """
    Assign whole documents to train or validation.

    Stratified by top-level directory so validation is not accidentally made up
    entirely of, say, release changelogs. Without stratification a random draw
    over ~255 documents can easily produce a validation set that covers none of
    the tutorial content, which is the part the model is actually for.

    Documents are shuffled within each stratum and taken until that stratum's
    share of the validation budget is met. Because documents differ in size,
    the resulting pair-level split will not land exactly on val_frac — that is
    expected, and the real ratio is reported rather than forced.
    """
    rng = random.Random(seed)

    by_doc: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        by_doc[p["doc_id"]].append(p)

    by_stratum: dict[str, list[str]] = defaultdict(list)
    for doc in by_doc:
        by_stratum[top_dir(doc)].append(doc)

    val_docs: set[str] = set()
    for stratum, docs in sorted(by_stratum.items()):
        docs = sorted(docs)              # deterministic before shuffling
        rng.shuffle(docs)

        stratum_pairs = sum(len(by_doc[d]) for d in docs)
        target = stratum_pairs * val_frac

        taken = 0
        for doc in docs:
            if taken >= target:
                break
            # Never let one document swallow the whole stratum's validation
            # budget; a 200-pair changelog would otherwise be the entire split.
            if taken > 0 and taken + len(by_doc[doc]) > target * 2:
                continue
            val_docs.add(doc)
            taken += len(by_doc[doc])

    train = [p for p in pairs if p["doc_id"] not in val_docs]
    val = [p for p in pairs if p["doc_id"] in val_docs]

    rng.shuffle(train)
    rng.shuffle(val)

    return train, val, {"val_docs": val_docs, "n_docs": len(by_doc)}


# ---------------------------------------------------------------------------
# Leak checks
# ---------------------------------------------------------------------------

def check_leakage(train: list[dict], val: list[dict]) -> dict:
    """
    Verify the hard guarantees, and measure the soft one.

    Hard (must be zero): shared documents, shared gold chunks, identical
    questions. Any of these means the split is broken.

    Soft (reported, not enforced): vocabulary overlap between validation
    questions and training questions. High overlap suggests near-duplicate
    content across different documents — the cross-platform duplication
    described at the top of this file.
    """
    train_docs = {p["doc_id"] for p in train}
    val_docs = {p["doc_id"] for p in val}
    train_chunks = {p["gold_chunk_id"] for p in train}
    val_chunks = {p["gold_chunk_id"] for p in val}
    train_qs = {normalize(p["question"]) for p in train}
    val_qs = {normalize(p["question"]) for p in val}

    # Soft check: how much of each validation question's vocabulary already
    # appears somewhere in training questions.
    train_vocab: Counter[str] = Counter()
    for p in train:
        train_vocab.update(set(normalize(p["question"]).split()) - STOPWORDS)

    coverages: list[float] = []
    for p in val:
        words = set(normalize(p["question"]).split()) - STOPWORDS
        if words:
            coverages.append(sum(1 for w in words if w in train_vocab) / len(words))

    return {
        "shared_docs": len(train_docs & val_docs),
        "shared_chunks": len(train_chunks & val_chunks),
        "shared_questions": len(train_qs & val_qs),
        "median_vocab_coverage": statistics.median(coverages) if coverages else 0.0,
        "high_coverage_frac": (sum(1 for c in coverages if c > 0.9)
                               / max(len(coverages), 1)),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def describe(name: str, rows: list[dict]) -> None:
    docs = {r["doc_id"] for r in rows}
    strata = Counter(top_dir(r["doc_id"]) for r in rows)
    ans_len = sorted(len(r["answer"]) for r in rows)
    with_code = sum(1 for r in rows if "```" in r["answer"])

    print(f"\n{name}")
    print(f"  pairs:      {len(rows):,}")
    print(f"  documents:  {len(docs):,}")
    print(f"  with code:  {with_code:,} ({with_code / max(len(rows),1):.0%})")
    print(f"  answer len: median {ans_len[len(ans_len)//2] if ans_len else 0} chars, "
          f"max {ans_len[-1] if ans_len else 0}")
    print(f"  top areas:  " + ", ".join(f"{k} {v}" for k, v in strata.most_common(4)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/processed/qa_raw.jsonl")
    ap.add_argument("--train-out", default="data/processed/ros2_qa_train.jsonl")
    ap.add_argument("--val-out", default="data/processed/ros2_qa_val.jsonl")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pairs = [json.loads(l) for l in Path(args.inp).open(encoding="utf-8")]
    train, val, info = split_by_document(pairs, args.val_frac, args.seed)

    print(f"Input:      {len(pairs):,} pairs across {info['n_docs']:,} documents")
    print(f"Target val: {args.val_frac:.0%}")

    describe("TRAIN", train)
    describe("VALIDATION", val)

    actual = len(val) / max(len(pairs), 1)
    print(f"\nActual val fraction: {actual:.1%} "
          f"(documents are uneven, so this will not hit the target exactly)")

    leak = check_leakage(train, val)
    print("\nLEAKAGE CHECKS")
    for key, label in [("shared_docs", "shared documents"),
                       ("shared_chunks", "shared gold chunks"),
                       ("shared_questions", "identical questions")]:
        status = "OK" if leak[key] == 0 else "FAIL"
        print(f"  {label:22} {leak[key]:5}   {status}")

    print(f"\n  median vocab coverage  {leak['median_vocab_coverage']:.2f}"
          f"   (val question words already seen in train)")
    print(f"  val questions >90% cov {leak['high_coverage_frac']:.1%}"
          f"   (candidates for cross-document duplication)")
    print("  Note: high vocabulary overlap is expected in a single technical "
          "domain and is\n        not itself leakage — it is reported so the "
          "risk is visible, not hidden.")

    write(Path(args.train_out), train)
    write(Path(args.val_out), val)
    print(f"\nWrote {args.train_out}")
    print(f"Wrote {args.val_out}")

    if any(leak[k] for k in ("shared_docs", "shared_chunks", "shared_questions")):
        raise SystemExit("\nSplit is leaking — do not train on this.")


if __name__ == "__main__":
    main()