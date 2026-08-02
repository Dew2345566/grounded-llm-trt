"""Tests for the train/val splitter. The leakage guarantees are the point."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.split_dataset import split_by_document, check_leakage, top_dir  # noqa: E402


def make(doc, chunk, q):
    return {"doc_id": doc, "gold_chunk_id": f"{doc}::{chunk}",
            "question": q, "answer": "a" * 50, "qa_id": f"{doc}::{chunk}::{q}"}


def test_no_document_appears_in_both_splits():
    pairs = [make(f"Dir-{i%4}\\Page-{i}.rst", c, f"question {i} {c}?")
             for i in range(40) for c in range(3)]
    train, val, _ = split_by_document(pairs, 0.2, seed=1)
    assert {p["doc_id"] for p in train}.isdisjoint({p["doc_id"] for p in val})


def test_all_questions_from_one_document_stay_together():
    """The core guarantee: sibling questions must not straddle the split."""
    pairs = [make("A\\one.rst", c, f"q{c}?") for c in range(10)]
    pairs += [make("B\\two.rst", c, f"r{c}?") for c in range(10)]
    train, val, _ = split_by_document(pairs, 0.5, seed=3)
    for doc in ("A\\one.rst", "B\\two.rst"):
        in_train = any(p["doc_id"] == doc for p in train)
        in_val = any(p["doc_id"] == doc for p in val)
        assert not (in_train and in_val), f"{doc} was split across both sets"


def test_gold_chunks_never_shared():
    pairs = [make(f"D\\p{i}.rst", c, f"q{i}{c}?") for i in range(20) for c in range(2)]
    train, val, _ = split_by_document(pairs, 0.25, seed=5)
    leak = check_leakage(train, val)
    assert leak["shared_docs"] == 0
    assert leak["shared_chunks"] == 0


def test_nothing_is_lost_or_duplicated():
    pairs = [make(f"D\\p{i}.rst", c, f"q{i}{c}?") for i in range(15) for c in range(2)]
    train, val, _ = split_by_document(pairs, 0.2, seed=9)
    assert len(train) + len(val) == len(pairs)
    ids = [p["qa_id"] for p in train + val]
    assert len(ids) == len(set(ids))


def test_split_is_deterministic():
    pairs = [make(f"D\\p{i}.rst", c, f"q{i}{c}?") for i in range(20) for c in range(2)]
    a, _, _ = split_by_document(pairs, 0.2, seed=42)
    b, _, _ = split_by_document(pairs, 0.2, seed=42)
    assert [p["qa_id"] for p in a] == [p["qa_id"] for p in b]


def test_top_dir_handles_both_separators():
    assert top_dir("Get-Started\\Installation\\x.rst") == "Get-Started"
    assert top_dir("Get-Started/Installation/x.rst") == "Get-Started"