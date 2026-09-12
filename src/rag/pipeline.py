"""
Phase 3, Stage 4 — The full pipeline: retrieve, ground, generate.

    question -> retrieve top-k -> build prompt -> generate -> answer + sources

Three modes, one code path
--------------------------
    base      base Llama-3-8B, no adapter, no context
    tuned     + QLoRA adapter, no context
    rag       + QLoRA adapter + retrieved context

They share the same decoding settings and the same prompt scaffolding so the
ablation compares exactly one variable at a time. Running three separate
scripts would risk a difference in temperature or chat template silently
explaining the result instead of the thing being measured.

How hard to push the model toward the context
---------------------------------------------
This is the decision that determines whether RAG works at all.

Too soft ("here is some context") and the model answers from memory, ignoring
retrieved text even when it is correct — the failure this project exists to fix.
Too hard ("answer ONLY from the context, otherwise refuse") and it refuses
whenever retrieval is imperfect, which at hit@5 = 0.728 is roughly a quarter of
queries.

The prompt below takes the middle path: prefer the context, copy commands and
package names from it character-for-character, and say plainly when the context
does not cover the question. That last clause matters most — it converts a
silent hallucination into a visible gap.

Why there is no score threshold
-------------------------------
It is tempting to drop chunks below some similarity and fall back to the model's
own knowledge. Measurement argues against it: a score of 0.78 was an exact hit
for the colcon install query, while 0.72 was confidently useless for a query
about a specific build flag. Score magnitude tracks "similar to something", not
"answers the question", so a threshold would discard good context and keep bad.
Context is passed through and the scores are reported, leaving the judgement
visible rather than buried.

Run:
    python src/rag/pipeline.py --mode rag --question "how do I install colcon on Ubuntu"
    python src/rag/pipeline.py --compare --questions-file benchmarks/eval_questions.txt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retriever import Retriever, Hit  # noqa: E402

BASE_MODEL = "NousResearch/Meta-Llama-3-8B-Instruct"

SYSTEM_PLAIN = "You are a precise assistant for ROS2 development questions."

SYSTEM_GROUNDED = """You are a precise assistant for ROS2 development questions.

Documentation excerpts are provided below. Use them as your primary source.

- Copy commands, flags, package names and file paths from the excerpts EXACTLY. \
These are the parts users paste into a terminal; altering a single character \
makes the answer wrong.
- Prefer the excerpts over your own recollection wherever they disagree.
- If the excerpts do not cover the question, say so plainly and then answer \
from general knowledge, making clear which part is which.
- Answer directly and concisely. Do not restate the question or describe the \
excerpts."""


@dataclass
class Answer:
    question: str
    mode: str
    text: str
    sources: list[dict] = field(default_factory=list)
    latency_s: float = 0.0
    prompt_tokens: int = 0
    new_tokens: int = 0


class Pipeline:
    """
    Holds one base model and optionally one adapter.

    The adapter is attached with PeftModel and then toggled with
    `disable_adapter()` rather than loading the 8B weights twice: base and tuned
    differ by 42M LoRA parameters, so keeping two full copies in VRAM would cost
    ~5.5 GB for nothing.
    """

    def __init__(self, base_model: str = BASE_MODEL, adapter_path: str | None = None,
                 index_dir: str | None = None, chunks_path: str | None = None,
                 load_in_4bit: bool = True, device: str | None = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.has_adapter = False

        quant = None
        if load_in_4bit and self.device == "cuda":
            from transformers import BitsAndBytesConfig
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                # bf16 on Ampere+, fp16 on Turing. Blackwell (sm_120) supports
                # bf16; the T4 used for training does not.
                bnb_4bit_compute_dtype=(
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16),
            )

        self.tok = AutoTokenizer.from_pretrained(base_model)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quant,
            device_map={"": 0} if self.device == "cuda" else None,
        )
        self.model.eval()

        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model.eval()
            self.has_adapter = True

        self.retriever = None
        if index_dir:
            self.retriever = Retriever(index_dir, device=self.device,
                                       chunks_path=chunks_path)

    # -- prompting -------------------------------------------------------

    def _build_messages(self, question: str, context: str | None) -> list[dict]:
        if context:
            return [
                {"role": "system", "content": SYSTEM_GROUNDED},
                {"role": "user",
                 "content": f"Documentation excerpts:\n\n{context}\n\n"
                            f"---\n\nQuestion: {question}"},
            ]
        return [
            {"role": "system", "content": SYSTEM_PLAIN},
            {"role": "user", "content": question},
        ]

    def _encode(self, messages: list[dict]) -> torch.Tensor:
        enc = self.tok.apply_chat_template(messages, add_generation_prompt=True,
                                           return_tensors="pt", return_dict=True)
        # transformers 5.x returns BatchEncoding; 4.x returns a bare tensor.
        ids = enc["input_ids"] if not torch.is_tensor(enc) else enc
        return ids.to(self.model.device)

    # -- generation ------------------------------------------------------

    @torch.no_grad()
    def _generate(self, messages: list[dict], max_new_tokens: int) -> tuple[str, int, int]:
        ids = self._encode(messages)
        out = self.model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,            # greedy: the ablation must not be
                                        # confounded by sampling randomness
            pad_token_id=self.tok.pad_token_id,
        )
        new = out[0][ids.shape[1]:]
        return (self.tok.decode(new, skip_special_tokens=True).strip(),
                int(ids.shape[1]), int(new.shape[0]))

    def answer(self, question: str, mode: str = "rag", k: int = 5,
               max_new_tokens: int = 300, max_context_chars: int = 6000) -> Answer:
        if mode not in {"base", "tuned", "rag"}:
            raise ValueError(f"unknown mode: {mode}")
        if mode in {"tuned", "rag"} and not self.has_adapter:
            raise ValueError(f"mode '{mode}' needs an adapter — pass adapter_path")
        if mode == "rag" and self.retriever is None:
            raise ValueError("mode 'rag' needs an index — pass index_dir")

        context, hits = None, []
        if mode == "rag":
            context, hits = self.retriever.build_context(
                question, k=k, max_chars=max_context_chars)

        messages = self._build_messages(question, context)

        t0 = time.time()
        if mode == "base":
            # Same object, adapter switched off — no second copy of the weights.
            with self.model.disable_adapter():
                text, n_prompt, n_new = self._generate(messages, max_new_tokens)
        else:
            text, n_prompt, n_new = self._generate(messages, max_new_tokens)
        latency = time.time() - t0

        return Answer(
            question=question, mode=mode, text=text,
            sources=[{"rank": h.rank, "score": round(h.score, 4),
                      "breadcrumb": h.breadcrumb, "chunk_id": h.chunk_id,
                      "url": h.url} for h in hits],
            latency_s=round(latency, 2),
            prompt_tokens=n_prompt, new_tokens=n_new,
        )

    def compare(self, question: str, **kw) -> dict[str, Answer]:
        """All three modes on one question, for the ablation table."""
        modes = ["base"]
        if self.has_adapter:
            modes.append("tuned")
        if self.has_adapter and self.retriever is not None:
            modes.append("rag")
        return {m: self.answer(question, mode=m, **kw) for m in modes}


def print_answer(a: Answer, show_sources: bool = True) -> None:
    print(f"\n[{a.mode}]  {a.latency_s}s  "
          f"{a.prompt_tokens} prompt + {a.new_tokens} new tokens")
    print(a.text)
    if show_sources and a.sources:
        print("\n  sources:")
        for s in a.sources:
            print(f"    [{s['rank']}] {s['score']:.4f}  {s['breadcrumb'][:70]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--adapter", default=None,
                    help="path to the LoRA adapter (the best/ checkpoint)")
    ap.add_argument("--index", default="data/processed/index")
    ap.add_argument("--chunks", default="data/processed/chunks.jsonl")
    ap.add_argument("--question")
    ap.add_argument("--questions-file",
                    help="one question per line; runs all of them")
    ap.add_argument("--mode", default="rag", choices=["base", "tuned", "rag"])
    ap.add_argument("--compare", action="store_true",
                    help="run every available mode on each question")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--out", default=None, help="write results as JSON")
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()

    if not args.question and not args.questions_file:
        raise SystemExit("give --question or --questions-file")

    questions = ([args.question] if args.question else
                 [l.strip() for l in Path(args.questions_file).read_text(
                     encoding="utf-8").splitlines() if l.strip()])

    needs_index = args.compare or args.mode == "rag"
    pipe = Pipeline(
        base_model=args.base_model,
        adapter_path=args.adapter,
        index_dir=args.index if needs_index else None,
        chunks_path=args.chunks if needs_index else None,
        load_in_4bit=not args.no_4bit,
    )
    print(f"model:   {args.base_model}")
    print(f"adapter: {args.adapter or '(none)'}")
    print(f"index:   {len(pipe.retriever.meta):,} chunks"
          if pipe.retriever else "index:   (none)")

    results = []
    for q in questions:
        print("\n" + "=" * 72)
        print(f"Q: {q}")
        if args.compare:
            answers = pipe.compare(q, k=args.k, max_new_tokens=args.max_new_tokens)
            for a in answers.values():
                print_answer(a, show_sources=(a.mode == "rag"))
            results.append({m: asdict(a) for m, a in answers.items()})
        else:
            a = pipe.answer(q, mode=args.mode, k=args.k,
                            max_new_tokens=args.max_new_tokens)
            print_answer(a)
            results.append(asdict(a))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()