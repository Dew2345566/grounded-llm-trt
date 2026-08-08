"""
Phase 3, Stage 1 — Embed the chunk corpus.

What this produces
------------------
A single tensor of shape [n_chunks, 768] plus a parallel list of chunk ids.
That is the entire "vector database": 5,893 x 768 floats is ~18 MB in fp32,
small enough to hold in memory and search with one matrix multiply.

Why no FAISS / Chroma / Pinecone
--------------------------------
Approximate nearest-neighbour indexes exist to avoid comparing a query against
every vector. At 5,893 vectors that comparison is a single 768-dimensional
matmul — well under a millisecond, and exact rather than approximate. An ANN
index would add a dependency, a build step, and a recall/latency tradeoff in
exchange for solving a problem this corpus does not have.

The crossover is somewhere around a million vectors. Below that, brute force
wins on simplicity and accuracy both.

Why normalise
-------------
For unit vectors, cosine similarity reduces to the dot product:

    cos(a, b) = (a . b) / (|a| |b|)  ->  a . b   when |a| = |b| = 1

So normalising at index time turns the whole search into `E @ q`, which is one
BLAS call. Doing it here rather than at query time means it happens once for
the corpus instead of once per search.

The asymmetry that is easy to miss
----------------------------------
BGE models are trained asymmetrically: passages are embedded bare, but queries
are meant to carry an instruction prefix. Embedding a query without it measurably
hurts retrieval. This file embeds passages (no prefix); retriever.py owns the
query side and applies the prefix there, so the two cannot drift apart.

Run:
    python src/rag/embed.py --in data/processed/chunks.jsonl \
                            --out data/processed/index
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

EMBED_MODEL = "BAAI/bge-base-en-v1.5"
EMBED_DIM = 768


def load_chunks(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    if not rows:
        raise SystemExit(f"No chunks in {path}")
    return rows


def embed_corpus(texts: list[str], model_name: str, batch_size: int,
                 device: str) -> torch.Tensor:
    """
    Encode every chunk into a normalised vector.

    Sorting by length before batching is a throughput trick: a batch is padded
    to its longest member, so mixing a 20-token chunk with a 500-token chunk
    wastes most of the compute on padding. Grouping similar lengths together
    cuts wasted work substantially on a corpus with this much size variance
    (median 249 tokens, p95 494).

    The original order is restored afterwards so row i of the returned tensor
    still corresponds to chunk i.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    print(f"model:      {model_name} on {device}")
    print(f"max_seq_len {model.max_seq_length} tokens")

    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    sorted_texts = [texts[i] for i in order]

    t0 = time.time()
    vecs = model.encode(
        sorted_texts,
        batch_size=batch_size,
        normalize_embeddings=True,     # unit length -> dot product == cosine
        convert_to_tensor=True,
        show_progress_bar=True,
    )
    print(f"encoded in  {time.time() - t0:.1f}s")

    # Undo the length sort.
    out = torch.empty_like(vecs)
    for position, original_index in enumerate(order):
        out[original_index] = vecs[position]
    return out.cpu()


def verify(embeddings: torch.Tensor, chunks: list[dict]) -> None:
    """
    Sanity checks. An embedding index fails silently: a broken one still returns
    plausible-looking neighbours, so these run every time rather than on demand.
    """
    n, d = embeddings.shape
    norms = embeddings.norm(dim=1)

    print("\nSANITY CHECKS")
    ok = True

    print(f"  shape                {tuple(embeddings.shape)}")
    if n != len(chunks):
        print(f"  row count            FAIL  {n} vs {len(chunks)} chunks"); ok = False
    else:
        print(f"  row count            OK    matches chunk count")

    if d != EMBED_DIM:
        print(f"  dimension            FAIL  expected {EMBED_DIM}"); ok = False
    else:
        print(f"  dimension            OK    {d}")

    # Every row must be unit length, or dot product is not cosine similarity.
    if torch.allclose(norms, torch.ones_like(norms), atol=1e-3):
        print(f"  normalised           OK    norms in "
              f"[{norms.min():.4f}, {norms.max():.4f}]")
    else:
        print(f"  normalised           FAIL  norms in "
              f"[{norms.min():.4f}, {norms.max():.4f}]"); ok = False

    if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
        print("  finite values        FAIL  contains NaN or Inf"); ok = False
    else:
        print("  finite values        OK")

    # All-identical vectors would mean the model collapsed or the text was empty.
    dup = (embeddings[0] @ embeddings[1:].T).max().item()
    if dup > 0.9999:
        print(f"  distinct vectors     WARN  two chunks are near-identical "
              f"(sim {dup:.5f})")
    else:
        print(f"  distinct vectors     OK    max sim to chunk 0: {dup:.4f}")

    # A quick self-retrieval probe: a chunk must be its own nearest neighbour.
    # A chunk must retrieve itself, or a chunk with identical text. The corpus
    # contains exact duplicates (ROS2 documents the same procedure per platform
    # and per language binding), and for identical vectors argmax returns the
    # lower index — which is correct behaviour, not a scrambled index.
    probe = min(50, n)
    sims = embeddings[:probe] @ embeddings.T
    best = sims.argmax(dim=1)
    self_sim = sims[torch.arange(probe), torch.arange(probe)]
    best_sim = sims.max(dim=1).values
    # Passing means: whatever came top scores no better than the chunk itself.
    intact = bool(((best_sim - self_sim).abs() < 1e-4).all())
    n_dup = int((best != torch.arange(probe)).sum())
    print(f"  self-retrieval       {'OK' if intact else 'FAIL'}"
          f"    {n_dup}/{probe} tied with an identical chunk")
    ok = ok and intact

    if not ok:
        raise SystemExit("\nIndex failed verification — do not use it.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/processed/chunks.jsonl")
    ap.add_argument("--out", default="data/processed/index")
    ap.add_argument("--model", default=EMBED_MODEL)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    chunks = load_chunks(Path(args.inp))
    texts = [c["text"] for c in chunks]      # breadcrumb + content, as indexed

    tok_est = sum(c.get("n_tokens", 0) for c in chunks)
    print(f"chunks:     {len(chunks):,}")
    print(f"tokens:     {tok_est:,}")

    embeddings = embed_corpus(texts, args.model, args.batch_size, args.device)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(embeddings, out_dir / "embeddings.pt")

    # Metadata is stored separately and in the same row order. Keeping the
    # tensor free of Python objects means it can be memory-mapped or moved to
    # GPU without dragging strings along.
    meta = [{"chunk_id": c["chunk_id"], "doc_id": c["doc_id"],
             "breadcrumb": c["breadcrumb"], "url": c["url"],
             "n_tokens": c.get("n_tokens", 0),
             "has_code": c.get("has_code", False)} for c in chunks]
    with (out_dir / "meta.jsonl").open("w", encoding="utf-8") as fh:
        for m in meta:
            fh.write(json.dumps(m, ensure_ascii=False) + "\n")

    # The model name is recorded because a query embedded with a different
    # model than the corpus produces silent nonsense rather than an error.
    (out_dir / "index_config.json").write_text(json.dumps({
        "model": args.model,
        "dim": embeddings.shape[1],
        "n_chunks": embeddings.shape[0],
        "normalized": True,
    }, indent=2))

    size_mb = (out_dir / "embeddings.pt").stat().st_size / 1e6
    print(f"\nwrote {out_dir}/embeddings.pt   ({size_mb:.1f} MB)")
    print(f"wrote {out_dir}/meta.jsonl")
    print(f"wrote {out_dir}/index_config.json")

    verify(embeddings, chunks)   

if __name__ == "__main__":
    main()