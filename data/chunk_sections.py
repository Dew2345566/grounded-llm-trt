"""
Phase 1, Stage 3 — Turn sections into uniform, embedding-ready chunks.

The problem
-----------
parse_rst.py produced 4,707 sections ranging from 41 to 29,769 characters.
That distribution is useless for retrieval:

  * Sections longer than the embedding model's context get silently truncated,
    so their tail is invisible to search.
  * A single vector for a very long section is an average of several topics,
    which is close to meaningless — it matches everything weakly and nothing well.
  * Sections of 40 characters ("See the tutorial below.") carry no retrievable
    signal on their own.

So there are exactly three cases, and each needs a different policy:

  1. FITS   — emit as-is.
  2. TOO BIG — split, preferring natural boundaries over arbitrary cuts.
  3. TOO SMALL — merge with neighbours from the SAME document.

Design decisions worth defending in an interview
------------------------------------------------
* Token budget is measured with the *embedding* tokenizer (bge-base-en-v1.5),
  not Llama's. The chunk has to fit the model that will embed it. bge is a
  BERT-family model with a 512-token limit, so that is the hard ceiling.

* Every chunk is prefixed with its breadcrumb. A passage reading "run this
  command" is unretrievable; the same passage headed
  "Using colcon > Prerequisites > Install colcon" is not. The breadcrumb costs
  tokens, so it is subtracted from the content budget rather than ignored.

* Code blocks are treated as ATOMIC units. Half a shell command is worse than
  no shell command — splitting one produces two chunks that are both wrong.
  A code block that exceeds the budget on its own is split as a last resort and
  flagged (`code_split`) so it shows up in the statistics instead of hiding.

* Merging never crosses a document boundary. Two adjacent short sections from
  different pages are unrelated; merging them creates a chunk that is factually
  incoherent and will retrieve for queries it cannot answer.

* Overlap applies only when SPLITTING, never when merging. Overlap exists to
  stop a continuous passage from losing context at an arbitrary cut. A merge has
  no arbitrary cut — the boundary is a real section boundary — so overlap there
  would only duplicate text for no benefit.

Run:
    python data/chunk_sections.py \
        --in data/processed/sections.jsonl \
        --out data/processed/chunks.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

# bge-base-en-v1.5 is BERT-based: hard limit 512 tokens including [CLS]/[SEP].
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
HARD_LIMIT = 512
SPECIAL_TOKENS = 2          # [CLS] and [SEP]
SAFETY_MARGIN = 8           # tokenizer disagreements, joining whitespace
MAX_CHUNK_TOKENS = HARD_LIMIT - SPECIAL_TOKENS - SAFETY_MARGIN   # 502
OVERLAP_TOKENS = 64
MIN_CHUNK_TOKENS = 48       # below this, a chunk is merged into its neighbour


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def get_token_counter(model_name: str = EMBED_MODEL) -> tuple[Callable[[str], int], str]:
    """
    Return (counter, description).

    Prefers the real tokenizer. Falls back to a character heuristic so the
    script still runs offline — but the fallback is reported loudly, because
    chunk sizes computed from an estimate are not trustworthy.
    """
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name)

        def count(text: str) -> int:
            return len(tok.encode(text, add_special_tokens=False))

        return count, f"exact ({model_name})"
    except Exception as e:                                   # noqa: BLE001
        def count(text: str) -> int:
            # ~4 chars/token for English prose; code is denser, so bias down.
            return max(1, len(text) // 4)

        return count, f"ESTIMATED (chars/4) — tokenizer unavailable: {type(e).__name__}"


# ---------------------------------------------------------------------------
# Splitting text into atomic units
# ---------------------------------------------------------------------------

FENCE_RE = re.compile(r"^```")


@dataclass
class Unit:
    """An indivisible piece of a section: one code block, or one paragraph."""
    text: str
    is_code: bool
    n_tokens: int


def to_units(text: str, count: Callable[[str], int]) -> list[Unit]:
    """
    Break a section into units, keeping fenced code blocks whole.

    Prose is split on blank lines. This is the highest-quality boundary
    available: paragraph breaks are author-chosen semantic boundaries, unlike
    a cut at token 502.
    """
    units: list[Unit] = []
    buf: list[str] = []
    in_code = False

    def flush_prose() -> None:
        raw = "\n".join(buf).strip()
        buf.clear()
        if not raw:
            return
        for para in re.split(r"\n\s*\n", raw):
            para = para.strip()
            if para:
                units.append(Unit(para, False, count(para)))

    for line in text.splitlines():
        if FENCE_RE.match(line):
            if not in_code:
                flush_prose()
                buf.append(line)
                in_code = True
            else:
                buf.append(line)
                block = "\n".join(buf)
                buf.clear()
                in_code = False
                units.append(Unit(block, True, count(block)))
            continue
        buf.append(line)

    if in_code:
        # Unterminated fence: treat the remainder as code rather than losing it.
        block = "\n".join(buf)
        buf.clear()
        units.append(Unit(block, True, count(block)))
    else:
        flush_prose()

    return units


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def split_oversized_unit(unit: Unit, budget: int, count: Callable[[str], int]) -> list[Unit]:
    """
    Last resort for a single unit that exceeds the budget on its own.

    Prose: split on sentence boundaries, then on words if a sentence is still
    too long. Code: split on line boundaries — never mid-line, because half a
    command is actively misleading.
    """
    if unit.n_tokens <= budget:
        return [unit]

    pieces: list[str] = []

    if unit.is_code:
        lines = unit.text.splitlines()
        cur: list[str] = []
        for ln in lines:
            trial = "\n".join(cur + [ln])
            if cur and count(trial) > budget:
                pieces.append("\n".join(cur))
                cur = [ln]
            else:
                cur.append(ln)
        if cur:
            pieces.append("\n".join(cur))
    else:
        sentences = SENTENCE_RE.split(unit.text)
        cur = []
        for s in sentences:
            trial = " ".join(cur + [s])
            if cur and count(trial) > budget:
                pieces.append(" ".join(cur))
                cur = [s]
            else:
                cur.append(s)
        if cur:
            pieces.append(" ".join(cur))

        # A single sentence may still be too long (rare: giant tables/URLs).
        final: list[str] = []
        for p in pieces:
            if count(p) <= budget:
                final.append(p)
                continue
            words, cur_w = p.split(), []
            for w in words:
                trial = " ".join(cur_w + [w])
                if cur_w and count(trial) > budget:
                    final.append(" ".join(cur_w))
                    cur_w = [w]
                else:
                    cur_w.append(w)
            if cur_w:
                final.append(" ".join(cur_w))
        pieces = final

    return [Unit(p, unit.is_code, count(p)) for p in pieces if p.strip()]


# ---------------------------------------------------------------------------
# Packing units into chunks
# ---------------------------------------------------------------------------

def pack_units(units: list[Unit], budget: int, count: Callable[[str], int]) -> list[list[Unit]]:
    """
    Greedily fill chunks up to `budget`, then start a new one carrying an
    overlap tail.

    Overlap is applied at UNIT granularity: the trailing prose units of the
    previous chunk are repeated at the head of the next, up to OVERLAP_TOKENS.
    Whole units are used rather than a raw token slice so overlap never begins
    mid-sentence. Code units are not repeated — duplicating a command across two
    chunks would make it look like it must be run twice.
    """
    chunks: list[list[Unit]] = []
    cur: list[Unit] = []
    cur_tokens = 0

    for u in units:
        if cur and cur_tokens + u.n_tokens > budget:
            chunks.append(cur)
            # Build the overlap tail from the end of the chunk just closed.
            tail: list[Unit] = []
            tail_tokens = 0
            for prev in reversed(cur):
                if prev.is_code:
                    break
                if tail_tokens + prev.n_tokens > OVERLAP_TOKENS:
                    break
                tail.insert(0, prev)
                tail_tokens += prev.n_tokens
            cur = list(tail)
            cur_tokens = tail_tokens

        cur.append(u)
        cur_tokens += u.n_tokens

    if cur:
        chunks.append(cur)
    return chunks


def enforce_limit(
    groups: list[list[Unit]],
    breadcrumb: str,
    budget_total: int,
    count: Callable[[str], int],
) -> list[list[Unit]]:
    """
    Guarantee every group fits, by measuring the REAL assembled text.

    pack_units works from a running sum of per-unit token counts, which
    systematically under-counts: joining units inserts separators, and per-unit
    rounding compounds. The only trustworthy number is the token count of the
    string that will actually be embedded, so it is measured here and any
    overflowing group sheds trailing units into the following group.
    """
    queue = [list(g) for g in groups]
    fixed: list[list[Unit]] = []

    while queue:
        group = queue.pop(0)
        if not group:
            continue

        while len(group) > 1:
            text = make_text(breadcrumb, "\n\n".join(u.text for u in group))
            if count(text) <= budget_total:
                break
            # Push the last unit forward rather than dropping it.
            moved = group.pop()
            if queue:
                queue[0].insert(0, moved)
            else:
                queue.append([moved])

        fixed.append(group)

    return fixed


# ---------------------------------------------------------------------------
# Section -> chunks
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    breadcrumb: str
    url: str
    text: str            # what actually gets embedded (breadcrumb + content)
    content: str         # content without the breadcrumb prefix
    n_tokens: int
    has_code: bool
    code_split: bool     # True if a code block had to be broken
    merged_from: int     # how many sections were merged (1 = none)


def make_text(breadcrumb: str, content: str) -> str:
    return f"{breadcrumb}\n\n{content}"


def chunk_section(
    section: dict,
    count: Callable[[str], int],
    seq_start: int,
    merged_from: int = 1,
) -> list[Chunk]:
    """Turn one (possibly merged) section into one or more chunks."""
    breadcrumb = section["breadcrumb"]
    bc_tokens = count(breadcrumb) + 2          # +2 for the blank-line join
    budget = MAX_CHUNK_TOKENS - bc_tokens

    if budget < 64:
        # Pathological breadcrumb; truncate it rather than starve the content.
        breadcrumb = breadcrumb[-200:]
        bc_tokens = count(breadcrumb) + 2
        budget = MAX_CHUNK_TOKENS - bc_tokens

    units = to_units(section["text"], count)

    code_split = False
    expanded: list[Unit] = []
    for u in units:
        if u.n_tokens > budget:
            parts = split_oversized_unit(u, budget, count)
            if u.is_code and len(parts) > 1:
                code_split = True
            expanded.extend(parts)
        else:
            expanded.append(u)

    out: list[Chunk] = []
    groups = pack_units(expanded, budget, count)
    groups = enforce_limit(groups, breadcrumb, HARD_LIMIT - SPECIAL_TOKENS, count)
    for i, group in enumerate(groups):
        content = "\n\n".join(g.text for g in group).strip()
        if not content:
            continue
        text = make_text(breadcrumb, content)
        out.append(Chunk(
            chunk_id=f"{section['doc_id']}::{seq_start + i}",
            doc_id=section["doc_id"],
            breadcrumb=breadcrumb,
            url=section["url"],
            text=text,
            content=content,
            n_tokens=count(text),
            has_code="```" in content,
            code_split=code_split,
            merged_from=merged_from,
        ))
    return out


def merge_small_sections(sections: list[dict], count: Callable[[str], int]) -> list[tuple[dict, int]]:
    """
    Merge consecutive short sections within the same document.

    Returns (section, n_merged). The merged breadcrumb is the SHARED prefix of
    the merged sections' breadcrumbs, so the chunk does not claim to be about a
    subsection it only partly covers.
    """
    merged: list[tuple[dict, int]] = []
    buf: list[dict] = []

    def shared_breadcrumb(items: list[dict]) -> str:
        paths = [s["breadcrumb"].split(" > ") for s in items]
        common: list[str] = []
        for parts in zip(*paths):
            if len(set(parts)) == 1:
                common.append(parts[0])
            else:
                break
        return " > ".join(common) if common else items[0]["breadcrumb"]

    def flush() -> None:
        if not buf:
            return
        if len(buf) == 1:
            merged.append((buf[0], 1))
        else:
            merged.append(({
                "doc_id": buf[0]["doc_id"],
                "breadcrumb": shared_breadcrumb(buf),
                "url": buf[0]["url"],
                "text": "\n\n".join(s["text"] for s in buf),
            }, len(buf)))
        buf.clear()

    for sec in sections:
        n = count(sec["text"])
        if n >= MIN_CHUNK_TOKENS:
            flush()
            merged.append((sec, 1))
            continue

        # Short section: try to accumulate with other short neighbours.
        if buf and buf[0]["doc_id"] != sec["doc_id"]:
            flush()
        buf.append(sec)
        if count("\n\n".join(s["text"] for s in buf)) >= MIN_CHUNK_TOKENS * 3:
            flush()

    flush()
    return merged


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def report(chunks: list[Chunk], sections: list[dict], count: Callable[[str], int], mode: str) -> None:
    sizes = sorted(c.n_tokens for c in chunks)
    n = len(sizes)

    def pct(p: float) -> int:
        return sizes[min(int(n * p), n - 1)] if n else 0

    src_tokens = sum(count(s["text"]) for s in sections)
    out_tokens = sum(c.n_tokens for c in chunks)
    over = [c for c in chunks if c.n_tokens > HARD_LIMIT - SPECIAL_TOKENS]
    empty = [c for c in chunks if not c.content.strip()]
    with_code = sum(1 for c in chunks if c.has_code)
    code_split = sum(1 for c in chunks if c.code_split)
    merged = sum(1 for c in chunks if c.merged_from > 1)

    print(f"Token counting:     {mode}")
    print(f"Sections in:        {len(sections):,}")
    print(f"Chunks out:         {n:,}")
    print()
    print(f"Tokens  min/median: {sizes[0] if n else 0} / {statistics.median(sizes) if n else 0:.0f}")
    print(f"        p95/max:    {pct(0.95)} / {sizes[-1] if n else 0}")
    print(f"Chunks with code:   {with_code:,} ({with_code / max(n,1):.0%})")
    print(f"Merged chunks:      {merged:,}")
    print(f"Code blocks split:  {code_split:,}")
    print()
    print(f"Tokens in/out:      {src_tokens:,} -> {out_tokens:,} "
          f"({(out_tokens / max(src_tokens,1) - 1) * 100:+.1f}%)")
    print()
    print("SANITY CHECKS")
    print(f"  over hard limit:  {len(over)}   {'OK' if not over else 'FAIL'}")
    print(f"  empty content:    {len(empty)}   {'OK' if not empty else 'FAIL'}")
    ratio_ok = out_tokens >= src_tokens * 0.95
    print(f"  no content lost:  {'OK' if ratio_ok else 'FAIL — output shrank, check for dropped text'}")

    if over:
        print("\n  Worst offenders:")
        for c in sorted(over, key=lambda c: -c.n_tokens)[:3]:
            print(f"    {c.n_tokens} tok  {c.chunk_id}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/processed/sections.jsonl")
    ap.add_argument("--out", default="data/processed/chunks.jsonl")
    ap.add_argument("--preview", type=int, default=0)
    ap.add_argument("--min-tokens", type=int, default=MIN_CHUNK_TOKENS,
                    help="Drop chunks below this many tokens (stub/placeholder pages)")
    args = ap.parse_args()

    count, mode = get_token_counter()

    sections = [json.loads(l) for l in Path(args.inp).open(encoding="utf-8")]

    chunks: list[Chunk] = []
    for section, n_merged in merge_small_sections(sections, count):
        chunks.extend(chunk_section(section, count, seq_start=0, merged_from=n_merged))

     # Drop stub/placeholder chunks (e.g. "Coming Soon" pages).
    n_before = len(chunks)
    chunks = [c for c in chunks if c.n_tokens >= args.min_tokens]
    n_dropped = n_before - len(chunks)

    # Re-number chunk ids per document so they are stable and unique.
    per_doc: dict[str, int] = {}
    # Re-number chunk ids per document so they are stable and unique.
    per_doc: dict[str, int] = {}
    for c in chunks:
        i = per_doc.get(c.doc_id, 0)
        c.chunk_id = f"{c.doc_id}::{i}"
        per_doc[c.doc_id] = i + 1

    if args.preview:
        for c in chunks[:args.preview]:
            print("=" * 70)
            print(f"{c.chunk_id}  [{c.n_tokens} tok, code={c.has_code}, merged={c.merged_from}]")
            print("-" * 70)
            print(c.text[:700])
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    report(chunks, sections, count, mode)
    print(f"Dropped (stub, <{args.min_tokens} tok): {n_dropped:,}")
    print(f"\nOutput: {out_path}")


if __name__ == "__main__":
    main()