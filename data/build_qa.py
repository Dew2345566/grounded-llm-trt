"""
Phase 1, Stage 4 — Generate the instruction-tuning dataset from chunks.

Why generate from chunks rather than writing questions by hand
--------------------------------------------------------------
Because it hands us the retrieval ground truth for free. Every generated
question knows which chunk it came from, so `gold_chunk_id` is a label we did
not have to annotate. Without it, hit-rate@k and MRR are unmeasurable and
Phase 3 has no evaluation at all.

This single ordering decision — chunk first, then generate — is what makes the
project's retrieval metrics possible.

What makes a trainable question
-------------------------------
Three failure modes are designed against explicitly:

1. META-REFERENCES. "According to the passage, what does colcon do?" At
   inference time there is no passage. These teach the model to refer to
   context it will not have.

2. GLOBALLY ANSWERABLE QUESTIONS. "What is ROS2?" The base model already knows.
   Training on it wastes capacity, and for retrieval evaluation it is worse:
   the question fails to discriminate between chunks, so hit-rate becomes noise.

3. EXTRACTION DISGUISED AS ANSWERING. If the answer is mostly a verbatim copy of
   the chunk, the model is being trained as a highlighter.

Cost control
------------
One API call per chunk is not free, so three things are non-negotiable:
caching (never pay twice for the same chunk), resumability (a rate limit at
chunk 1,200 must not discard 1,200 chunks of work), and stratified sampling
(2,000 well-spread chunks beat 5,893 redundant ones).

Run:
    export GEMINI_API_KEY=...
    python data/build_qa.py --provider gemini --max-chunks 2000 --per-chunk 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

CACHE_DIR = Path("data/processed/.qa_cache")
BANNED_QUESTION_PHRASES = [
    "this document", "the document", "this passage", "the passage",
    "this section", "the section", "the above", "mentioned above",
    "according to the", "in the text", "this chunk", "the excerpt",
    "as described", "the following text", "this page", "the tutorial above",
]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You write training data for a ROS2 documentation assistant.

Given one passage of ROS2 documentation, write questions that a working ROS2 \
developer would genuinely type into a search bar, together with answers drawn \
strictly from the passage.

HARD RULES for questions:
- Must be answerable using ONLY the passage. No outside knowledge.
- Must be specific enough that this passage is the single best source. \
"What is ROS2?" is useless; "Which colcon flag skips packages that are already \
built?" is good.
- Must NEVER refer to the passage itself. Forbidden: "according to the \
document", "in this section", "the above", "as described". The question must \
stand alone with no context.
- Must sound like a real developer, not a quiz question. Natural phrasing, \
including how people actually type ("how do I...", "why does ... fail").
- Vary the shape: some how-to, some why, some which-flag/which-command, some \
troubleshooting.
- Each question must target a DIFFERENT fact, command, or concept in the \
passage. Never rephrase the same question twice. If the passage only supports \
one good question, return only one — fewer good questions is better than \
padding with rephrasings.

HARD RULES for answers:
- Answer the question directly. Do not restate the question.
- Reproduce commands, flags, file paths and code EXACTLY as written in the \
passage. These are the part users copy; altering them makes the answer wrong.
- Be complete but not padded. Two to six sentences, plus code blocks where the \
passage has them.
- Do not invent anything absent from the passage. If the passage lacks a \
detail, leave it out.
- Do not reference the passage. Write as if answering from your own knowledge.

If the passage cannot support good questions — it is a navigation stub, a table \
of contents, a placeholder like "Coming Soon", pure license text, or a bare \
list of links — set "suitable" to false and return an empty questions list. \
Do not force questions out of unsuitable material.

Return STRICT JSON only. No markdown fences, no commentary:
{"suitable": true, "questions": [{"question": "...", "answer": "..."}]}"""


def build_user_prompt(chunk: dict, n: int) -> str:
    return (
        f"Documentation section: {chunk['breadcrumb']}\n\n"
        f"--- PASSAGE START ---\n{chunk['content']}\n--- PASSAGE END ---\n\n"
        f"Write {n} question-answer pairs following the rules. JSON only."
    )


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

def make_provider(name: str, model: str | None) -> tuple[Callable[[str, str], str], str]:
    """
    Return (call_fn, model_name). call_fn(system, user) -> raw text response.

    Kept as a thin adapter so switching provider is a flag, not a rewrite. The
    generation prompt is the valuable artifact here; the vendor behind it is an
    implementation detail.
    """
    if name == "gemini":
        model = model or "gemini-2.0-flash"
        key = require_key("GEMINI_API_KEY")
        import urllib.request

        def call(system: str, user: str) -> str:
            url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
                   f"{model}:generateContent?key={key}")
            body = {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"parts": [{"text": user}]}],
                "generationConfig": {"temperature": 0.7, "maxOutputTokens": 2048,
                                     "responseMimeType": "application/json"},
            }
            req = urllib.request.Request(
                url, data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.loads(r.read())
            return data["candidates"][0]["content"]["parts"][0]["text"]

        return call, model

    if name == "groq":
        model = model or "llama-3.3-70b-versatile"
        key = require_key("GROQ_API_KEY")
        return _openai_compatible(
            "https://api.groq.com/openai/v1/chat/completions", key, model), model

    if name == "openai":
        model = model or "gpt-4o-mini"
        key = require_key("OPENAI_API_KEY")
        return _openai_compatible(
            "https://api.openai.com/v1/chat/completions", key, model), model

    if name == "anthropic":
        model = model or "claude-sonnet-4-6"
        key = require_key("ANTHROPIC_API_KEY")
        import urllib.request

        def call(system: str, user: str) -> str:
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=json.dumps({
                    "model": model, "max_tokens": 2048, "system": system,
                    "messages": [{"role": "user", "content": user}],
                }).encode(),
                headers={"Content-Type": "application/json",
                         "x-api-key": key,
                         "anthropic-version": "2023-06-01"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.loads(r.read())
            return data["content"][0]["text"]

        return call, model

    if name == "ollama":
        # Local model, e.g. on Kaggle or your own machine. Zero cost.
        model = model or "qwen2.5:7b-instruct"
        import urllib.request

        def call(system: str, user: str) -> str:
            req = urllib.request.Request(
                "http://localhost:11434/api/chat",
                data=json.dumps({
                    "model": model, "stream": False, "format": "json",
                    "messages": [{"role": "system", "content": system},
                                 {"role": "user", "content": user}],
                }).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read())["message"]["content"]

        return call, model

    raise SystemExit(f"Unknown provider: {name}")


def _openai_compatible(url: str, key: str, model: str) -> Callable[[str, str], str]:
    import urllib.request

    def call(system: str, user: str) -> str:
        req = urllib.request.Request(
            url,
            data=json.dumps({
                "model": model, "temperature": 0.7,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
            }).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]

    return call


def require_key(var: str) -> str:
    key = os.environ.get(var)
    if not key:
        raise SystemExit(
            f"{var} is not set.\n"
            f"  Windows CMD:  set {var}=your_key_here\n"
            f"  PowerShell:   $env:{var}='your_key_here'\n"
            f"  Linux/macOS:  export {var}=your_key_here")
    return key


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_response(raw: str) -> dict | None:
    """
    Parse the model's JSON, tolerating the usual deviations.

    Even with JSON mode requested, models occasionally wrap output in fences or
    add a sentence before the object. Recovering from that is cheaper than
    re-paying for the call.
    """
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fall back to the outermost {...} span.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@dataclass
class QAPair:
    qa_id: str
    question: str
    answer: str
    gold_chunk_id: str      # the retrieval ground-truth label
    doc_id: str
    breadcrumb: str
    url: str
    provider: str
    model: str


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


# Function words carry no topical signal, so they are excluded before
# comparing questions. Without this, "how do I ..." phrasing alone would make
# unrelated questions look similar.
STOPWORDS = {
    "how", "do", "i", "what", "the", "a", "an", "is", "are", "to", "in",
    "for", "my", "on", "of", "and", "with", "can", "you", "does", "when",
    "where", "which", "that", "this", "it", "be", "use", "using", "need",
}


def content_words(text: str) -> set[str]:
    return set(normalize(text).split()) - STOPWORDS


def jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def is_redundant_answer(answer: str, seen_in_chunk: list[set[str]],
                        threshold: float = 0.45) -> bool:
    """
    Detect two questions from the same chunk that resolve to the same fact.

    Measured on the pilot batch rather than guessed: for the RHEL pair
    ("What packages do I need on RHEL" / "How do I install the prerequisites"),
    QUESTION similarity was only 0.30 while ANSWER similarity was 0.467. The
    questions share almost no vocabulary despite pointing at one identical
    command, so question-level comparison cannot catch this and answer-level can.

    Scoped per chunk deliberately. A 0.45 threshold applied across the whole
    corpus would delete legitimately distinct pairs that happen to share
    technical vocabulary; within a single passage, two answers that overlap this
    much are describing the same thing.
    """
    words = content_words(answer)
    if not words:
        return False
    for prev in seen_in_chunk:
        if jaccard(words, prev) >= threshold:
            return True
    seen_in_chunk.append(words)
    return False


def overlap_ratio(answer: str, chunk_text: str) -> float:
    """
    Fraction of the answer's words that appear as a contiguous run in the chunk.

    Approximated with the longest shared word sequence, normalised by answer
    length. High values mean the "answer" is really an extract.
    """
    a = normalize(answer).split()
    c = normalize(chunk_text).split()
    if not a:
        return 1.0

    c_index = defaultdict(list)
    for i, w in enumerate(c):
        c_index[w].append(i)

    best = 0
    for i, w in enumerate(a):
        for j in c_index.get(w, ()):
            k = 0
            while (i + k < len(a) and j + k < len(c) and a[i + k] == c[j + k]):
                k += 1
            best = max(best, k)
    return best / len(a)


def validate(q: str, a: str, chunk: dict, max_overlap: float) -> str | None:
    """Return a rejection reason, or None if the pair is acceptable."""
    ql, al = q.strip(), a.strip()

    if len(ql) < 15:
        return "question_too_short"
    if len(ql) > 400:
        return "question_too_long"
    if not ql.endswith("?"):
        return "question_not_a_question"
    if len(al) < 30:
        return "answer_too_short"

    low_q = ql.lower()
    for phrase in BANNED_QUESTION_PHRASES:
        if phrase in low_q:
            return f"meta_reference:{phrase}"

    low_a = al.lower()
    if any(p in low_a for p in ("the passage", "this document", "the above",
                                "the section states")):
        return "answer_references_source"

    if any(p in low_a for p in ("i don't know", "not specified", "unclear from",
                                "cannot be determined")):
        return "answer_is_refusal"

    if overlap_ratio(al, chunk["content"]) > max_overlap:
        return "answer_is_extraction"

    return None


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def stratified_sample(chunks: list[dict], max_chunks: int, max_per_doc: int,
                      seed: int) -> list[dict]:
    """
    Spread the sample across documents instead of taking a flat random draw.

    A flat sample over-represents long pages: the installation guide alone has
    hundreds of chunks and would dominate, teaching the model a great deal about
    apt and nothing about lifecycle nodes. Capping per document buys topical
    coverage, which is what the fine-tune actually needs.
    """
    rng = random.Random(seed)
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        by_doc[c["doc_id"]].append(c)

    picked: list[dict] = []
    for doc in sorted(by_doc):
        group = by_doc[doc][:]
        rng.shuffle(group)
        # Prefer chunks with code: they yield concrete, verifiable questions.
        group.sort(key=lambda c: not c.get("has_code", False))
        picked.extend(group[:max_per_doc])

    rng.shuffle(picked)
    return picked[:max_chunks]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def cache_path(chunk_id: str, model: str, n: int) -> Path:
    key = hashlib.sha256(f"{chunk_id}|{model}|{n}".encode()).hexdigest()[:20]
    return CACHE_DIR / f"{key}.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/processed/chunks.jsonl")
    ap.add_argument("--out", default="data/processed/qa_raw.jsonl")
    ap.add_argument("--provider", default="gemini",
                    choices=["gemini", "groq", "openai", "anthropic", "ollama"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--per-chunk", type=int, default=2,
                    help="questions requested per chunk")
    ap.add_argument("--max-chunks", type=int, default=2000)
    ap.add_argument("--max-per-doc", type=int, default=12)
    ap.add_argument("--max-overlap", type=float, default=0.6,
                    help="reject answers whose longest verbatim run exceeds this "
                         "fraction of the answer")
    ap.add_argument("--sleep", type=float, default=0.5,
                    help="seconds between calls, for rate limits")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true",
                    help="show the sample and one prompt, make no API calls")
    args = ap.parse_args()

    chunks = [json.loads(l) for l in Path(args.inp).open(encoding="utf-8")]
    sample = stratified_sample(chunks, args.max_chunks, args.max_per_doc, args.seed)

    print(f"Corpus:   {len(chunks):,} chunks across "
          f"{len({c['doc_id'] for c in chunks}):,} documents")
    print(f"Sampled:  {len(sample):,} chunks from "
          f"{len({c['doc_id'] for c in sample}):,} documents "
          f"(max {args.max_per_doc}/doc)")
    print(f"Target:   ~{len(sample) * args.per_chunk:,} pairs before validation")

    if args.dry_run:
        print("\n--- SYSTEM PROMPT ---")
        print(SYSTEM_PROMPT)
        print("\n--- EXAMPLE USER PROMPT ---")
        print(build_user_prompt(sample[0], args.per_chunk)[:1500])
        print("\nDry run: no API calls made.")
        return

    call, model = make_provider(args.provider, args.model)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Provider: {args.provider} / {model}\n")

    pairs: list[QAPair] = []
    rejects: dict[str, int] = defaultdict(int)
    seen_questions: set[str] = set()

    n_unsuitable = n_cached = n_called = n_failed = 0
    t0 = time.time()

    for i, chunk in enumerate(sample, 1):
        cp = cache_path(chunk["chunk_id"], model, args.per_chunk)

        if cp.exists():
            raw = cp.read_text(encoding="utf-8")
            n_cached += 1
        else:
            try:
                raw = call(SYSTEM_PROMPT, build_user_prompt(chunk, args.per_chunk))
                cp.write_text(raw, encoding="utf-8")
                n_called += 1
                time.sleep(args.sleep)
            except Exception as e:                            # noqa: BLE001
                # Never lose completed work to one bad call.
                n_failed += 1
                print(f"  ! {chunk['chunk_id']}: {type(e).__name__}: {e}")
                if n_failed > 20 and n_failed > n_called:
                    print("Too many consecutive failures — stopping. "
                          "Progress is cached; re-run to resume.")
                    break
                continue

        data = parse_response(raw)
        if not data:
            rejects["unparseable_response"] += 1
            continue
        if not data.get("suitable", True):
            n_unsuitable += 1
            continue

        # Reset per chunk: redundancy is only checked within one passage.
        answers_this_chunk: list[set[str]] = []

        for item in data.get("questions", []):
            q = str(item.get("question", ""))
            a = str(item.get("answer", ""))

            reason = validate(q, a, chunk, args.max_overlap)
            if reason:
                rejects[reason] += 1
                continue

            key = normalize(q)
            if key in seen_questions:
                rejects["duplicate_question"] += 1
                continue
            seen_questions.add(key)

            if is_redundant_answer(a, answers_this_chunk):
                rejects["redundant_answer_same_chunk"] += 1
                continue

            pairs.append(QAPair(
                qa_id=f"{chunk['chunk_id']}::q{len(pairs)}",
                question=q.strip(),
                answer=a.strip(),
                gold_chunk_id=chunk["chunk_id"],
                doc_id=chunk["doc_id"],
                breadcrumb=chunk["breadcrumb"],
                url=chunk["url"],
                provider=args.provider,
                model=model,
            ))

        if i % 50 == 0:
            rate = i / max(time.time() - t0, 1e-9)
            eta = (len(sample) - i) / max(rate, 1e-9) / 60
            print(f"  {i}/{len(sample)}  pairs={len(pairs)}  "
                  f"cached={n_cached}  eta={eta:.0f}min")

    # ---------------- write + report ----------------
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")

    total_attempted = len(pairs) + sum(rejects.values())
    print(f"\n{'=' * 58}")
    print(f"Chunks processed:   {n_cached + n_called:,} "
          f"(cached {n_cached:,}, called {n_called:,}, failed {n_failed:,})")
    print(f"Unsuitable chunks:  {n_unsuitable:,} "
          f"({n_unsuitable / max(n_cached + n_called, 1):.0%})")
    print(f"Pairs accepted:     {len(pairs):,} of {total_attempted:,} "
          f"({len(pairs) / max(total_attempted, 1):.0%})")

    if rejects:
        print("\nRejections by reason:")
        for reason, n in sorted(rejects.items(), key=lambda kv: -kv[1]):
            print(f"  {n:5,}  {reason}")

    docs = len({p.doc_id for p in pairs})
    golds = len({p.gold_chunk_id for p in pairs})
    print(f"\nCoverage:  {docs:,} documents, {golds:,} distinct gold chunks")
    print(f"Output:    {out}")
    print("\nNEXT: read 30-50 pairs yourself. Automated checks cannot judge "
          "whether a question is actually useful.")


if __name__ == "__main__":
    main()