# Grounded Q&A LLM — Hardware-Accelerated ROS2 Assistant

[![CI](https://github.com/Dew2345566/grounded-llm-trt/actions/workflows/ci.yml/badge.svg)](https://github.com/Dew2345566/grounded-llm-trt/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A Llama-3-8B assistant fine-tuned with QLoRA on ROS2 documentation, grounded with a
> custom PyTorch RAG pipeline, and accelerated for inference with NVIDIA TensorRT-LLM.

**Status:** 🚧 In development — Phase 1 (Data Engineering)

---

## Why this project

Large language models answer ROS2 questions confidently and often incorrectly. This project
tests a full pipeline for fixing that: domain fine-tuning for style and terminology,
retrieval for factual grounding, and compiled inference so the result is fast enough to
actually use.

Every layer is built from primitives rather than framework abstractions — the training loop,
the retriever, and the evaluation harness are all hand-written, so each design decision is
inspectable and defensible.

## Architecture

```
ROS2 .rst docs
      │
      ├─► chunking + metadata ──► embeddings ──► PyTorch top-k retriever ─┐
      │                                                                   │
      └─► synthetic Q&A pairs ──► QLoRA fine-tune ──► merge ──► TRT-LLM ──┴─► grounded answer
```

## Results

*(populated as phases complete)*

### Inference benchmark

| Backend | Precision | Batch | Tokens/sec | TTFT (ms) |
|---|---|---|---|---|
| HF Transformers | FP16 | 1 | — | — |
| vLLM | FP16 | 1 | — | — |
| TensorRT-LLM | FP16 | 1 | — | — |
| TensorRT-LLM | INT8 | 1 | — | — |

Hardware: *TBD*

### Ablation

| Configuration | Retrieval hit@5 | Answer accuracy | Faithfulness |
|---|---|---|---|
| Base Llama-3-8B | n/a | — | — |
| + QLoRA fine-tune | n/a | — | — |
| + RAG grounding | — | — | — |

## Repository layout

| Path | Contents |
|---|---|
| `configs/` | YAML configs for training and RAG |
| `data/` | Dataset build scripts and processed artifacts |
| `src/training/` | Hand-written QLoRA training loop, adapter merge |
| `src/rag/` | Embedding, retrieval, and generation pipeline |
| `src/deploy/` | TensorRT-LLM export and serving |
| `src/eval/` | Retrieval and answer-quality evaluation |
| `benchmarks/` | Throughput and latency results |
| `tests/` | Unit tests for data and retrieval logic |

## Quickstart

```bash
git clone https://github.com/Dew2345566/grounded-llm-trt.git
cd grounded-llm-trt
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Design decisions

*(written up as decisions are made — see `docs/DESIGN.md`)*

## Limitations

*(honest accounting, maintained throughout — see `docs/LIMITATIONS.md`)*

## License

MIT

## Author

Dew — 3rd-year Robotics & AI Engineering, KMITL  
GitHub: [@Dew2345566](https://github.com/Dew2345566)
