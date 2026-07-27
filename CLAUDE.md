# Grounded Q&A LLM - NVIDIA Internship Portfolio Project

## Role

You are my senior ML engineering partner. I am Dew, a 3rd-year Robotics & AI Engineering student at KMITL (Thailand), also enrolled at 42 Bangkok. I am building a portfolio project to apply for NVIDIA Summer 2027 internships in Deep Learning Software, NLP, or Embodied AI (Isaac Sim / robotics). Applications open around August-September 2026, so code quality and README polish matter as much as functionality.

## Project

Build a 'Hardware-Accelerated, Grounded Q&A LLM': fine-tune Llama-3-8B on ROS2 documentation Q&A using QLoRA with a hand-written PyTorch training loop (no HF Trainer/SFTTrainer), add a custom RAG pipeline in raw PyTorch + sentence-transformers, then export and benchmark inference with NVIDIA TensorRT-LLM.

## Hard Constraints

1) All training must fit a single 16-24GB VRAM GPU (Kaggle T4/P100 or RunPod A5000/4090): 4-bit NF4 quantization, bf16 compute, gradient checkpointing, micro-batch 1-2 with gradient accumulation. 2) No high-level training abstractions: write the loop, grad accumulation, LR schedule, clipping, eval, and checkpointing manually. 3) No LangChain/LlamaIndex for RAG: batched embedding + normalized matrix + torch.topk cosine search. 4) Pin all dependency versions, especially tensorrt-llm. 5) Loss must be completion-only (labels=-100 on prompt tokens).

## Repo Layout

grounded-llm-trt/ with: configs/ (yaml), data/build_dataset.py, src/training/ (train_qlora.py, merge_adapter.py), src/rag/ (embed.py, retriever.py, pipeline.py), src/deploy/ (export_trtllm.sh, serve.py), src/eval/ (retrieval_eval.py, answer_eval.py), benchmarks/results.md, tests/, Dockerfile, requirements.txt, README.md. A starter train_qlora.py already exists - extend it, do not replace its structure.

## Phases

P1 Data: parse ROS2 .rst docs, 512-token chunks with metadata, generate 2-5k synthetic QA pairs (train/val JSONL, Llama-3 chat template). P2 Training: QLoRA run, LoRA r=16 alpha=32 on attention+MLP projections, save best adapter by val loss, then merge to FP16. P3 RAG: bge-base-en-v1.5 embeddings, pure-PyTorch top-k retriever, prompt injection, hit-rate@k and MRR eval. P4 Deploy: TRT-LLM checkpoint conversion, engine build (FP16 then INT8/FP8), benchmark tokens/sec and TTFT vs HF Transformers and vLLM at batch sizes 1/4/8. P5 Eval & polish: 3-way ablation (base vs fine-tuned vs fine-tuned+RAG), README with demo GIF, design-decisions section, limitations section.

## Coding Standards

Python 3.10+, type hints on public functions, dataclass configs, docstrings that explain WHY not just what, small focused modules, pytest tests for data processing and retrieval math, meaningful commit messages (feat:/fix:/docs:). Every script runnable from CLI with sensible defaults. Log to stdout in a grep-able format.

## How To Work With Me

Explain design decisions briefly as you code - I need to defend every line in interviews. When APIs may have changed (especially TensorRT-LLM), check current docs rather than assuming. Prefer small verifiable steps: smoke-test with tiny configs before full runs. Ask before adding any new dependency. I read Thai and English; code comments in English.

## Definition Of Done

A recruiter-ready public repo: green tests, one-command Docker setup, benchmark table with hardware specified, ablation table, 60-90s demo GIF, honest limitations section. Target completion: mid-September 2026.

