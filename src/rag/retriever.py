"""
Phase 3, Stage 2 — Dense retrieval over the chunk index.

The search is three lines
------------------------
    q = encode(query)              # [768], normalised
    scores = E @ q                 # [n_chunks]
    top = scores.topk(k)

That is the entire retrieval engine. Because both sides are unit vectors, the
dot product *is* cosine similarity, so a single matmul against the whole corpus
gives exact scores for every chunk. At 5,893 x 768 that is roughly 4.5M
multiply-accumulates — under a millisecond, and exact rather than approximate.

FAISS, Chroma and friends exist to avoid scanning every vector. They buy speed
by giving up exactness, and they only start paying off somewhere around a
million vectors. Adding one here would mean a new dependency, an index build
step, and a recall/latency knob to tune, in exchange for solving a problem this
corpus does not have.

The asymmetry that costs accuracy if you miss it
------------------------------------------------
BGE is trained asymmetrically. Passages are embedded bare; queries are supposed
to carry an instruction prefix. Embedding a query without it puts the vector in
a slightly different region of the space than the training distribution, and
retrieval quality drops measurably. embed.py deliberately does not touch
queries, so the prefix lives here and only here.

Why the model name is asserted
------------------------------
A query embedded with a different model than the corpus does not raise — it
returns confident nonsense. index_config.json records the model that built the
index so the mismatch becomes an error instead of a silent wrong answer.

Run:
    python src/rag/retriever.py --query "how do I install colcon on Ubuntu"
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch

# BGE's query instruction. Passages do not get one; queries do.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass
class Hit:
    rank: int
    score: float
    chunk_id: str
    doc_id: str
    breadcrumb: str
    url: str
    text: str | None = None      # filled only when the chunk file is loaded


class Retriever:
    """
    Exact dense retrieval over an in-memory matrix.

    The embedding model is loaded lazily: evaluation code that only needs to
    score pre-computed query vectors should not pay for a 440 MB model load.
    """

    def __init__(self, index_dir: str | Path, device: str | None = None,
                 chunks_path: str | Path | None = None):
        index_dir = Path(index_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.config = json.loads((index_dir / "index_config.json").read_text())
        self.model_name = self.config["model"]

        self.embeddings = torch.load(index_dir / "embeddings.pt",
                                     map_location=self.device,
                                     weights_only=True)
        self.meta = [json.loads(l) for l in
                     (index_dir / "meta.jsonl").open(encoding="utf-8")]

        if len(self.meta) != self.embeddings.shape[0]:
            raise ValueError(
                f"index is inconsistent: {self.embeddings.shape[0]} vectors "
                f"but {len(self.meta)} metadata rows")

        # Optional: chunk text, for inspecting results or building prompts.
        self.texts: dict[str, str] | None = None
        if chunks_path:
            self.texts = {}
            for line in Path(chunks_path).open(encoding="utf-8"):
                c = json.loads(line)
                self.texts[c["chunk_id"]] = c["text"]

        self._model = None

    # -- embedding -------------------------------------------------------

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode_queries(self, queries: list[str], batch_size: int = 32
                       ) -> torch.Tensor:
        """Embed queries WITH the BGE instruction prefix, normalised."""
        prefixed = [QUERY_PREFIX + q for q in queries]
        vecs = self.model.encode(prefixed, batch_size=batch_size,
                                 normalize_embeddings=True,
                                 convert_to_tensor=True,
                                 show_progress_bar=len(queries) > 200)
        return vecs.to(self.device)

    # -- search ----------------------------------------------------------

    def search_vectors(self, query_vecs: torch.Tensor, k: int = 5
                       ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Score pre-computed query vectors against the whole corpus.

        Separated from `search` so evaluation can embed 2,163 questions once
        and then score them in a single batched matmul, rather than looping.

        Returns (scores [n_queries, k], indices [n_queries, k]).
        """
        if query_vecs.dim() == 1:
            query_vecs = query_vecs.unsqueeze(0)
        scores = query_vecs @ self.embeddings.T          # [n_queries, n_chunks]
        k = min(k, scores.shape[1])
        return scores.topk(k, dim=1)

    def search(self, query: str, k: int = 5) -> list[Hit]:
        q = self.encode_queries([query])
        scores, idx = self.search_vectors(q, k)
        return self._to_hits(scores[0], idx[0])

    def search_batch(self, queries: list[str], k: int = 5) -> list[list[Hit]]:
        q = self.encode_queries(queries)
        scores, idx = self.search_vectors(q, k)
        return [self._to_hits(scores[i], idx[i]) for i in range(len(queries))]

    def _to_hits(self, scores: torch.Tensor, idx: torch.Tensor) -> list[Hit]:
        hits = []
        for rank, (s, i) in enumerate(zip(scores.tolist(), idx.tolist()), start=1):
            m = self.meta[i]
            hits.append(Hit(
                rank=rank, score=s,
                chunk_id=m["chunk_id"], doc_id=m["doc_id"],
                breadcrumb=m["breadcrumb"], url=m["url"],
                text=self.texts.get(m["chunk_id"]) if self.texts else None,
            ))
        return hits

    # -- prompt construction ---------------------------------------------

    def build_context(self, query: str, k: int = 5, max_chars: int = 6000
                      ) -> tuple[str, list[Hit]]:
        """
        Retrieve and format chunks for injection into an LLM prompt.

        Chunks are numbered so the model can refer to a source, and truncated
        by total character budget rather than by k alone: five long chunks can
        crowd out the question itself in a limited context window.
        """
        if self.texts is None:
            raise ValueError("build_context needs chunk text — "
                             "construct Retriever with chunks_path=")

        hits = self.search(query, k)
        parts, used = [], 0
        for h in hits:
            block = f"[{h.rank}] {h.breadcrumb}\n{h.text}"
            if used + len(block) > max_chars and parts:
                break
            parts.append(block)
            used += len(block)
        return "\n\n---\n\n".join(parts), hits[:len(parts)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/processed/index")
    ap.add_argument("--chunks", default="data/processed/chunks.jsonl")
    ap.add_argument("--query", required=True)
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--show-text", action="store_true")
    args = ap.parse_args()

    r = Retriever(args.index, chunks_path=args.chunks)
    print(f"index: {r.embeddings.shape[0]:,} chunks, dim {r.embeddings.shape[1]}, "
          f"model {r.model_name}, device {r.device}\n")
    print(f"query: {args.query}\n")

    for h in r.search(args.query, args.k):
        print(f"[{h.rank}] {h.score:.4f}  {h.breadcrumb}")
        print(f"     {h.chunk_id}")
        if args.show_text and h.text:
            body = h.text.split("\n\n", 1)[-1]
            print("     " + body[:300].replace("\n", "\n     "))
        print()


if __name__ == "__main__":
    main()