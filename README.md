# Inference engine

A guided learning baseline for single-request, full-prefix greedy generation.
The implementation is in `inference_engine/generate_naive.py`.

## Setup and execution

Use Python 3.10 or newer in a supported PyTorch environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m inference_engine.generate_naive --device cpu
```

The first run downloads the GPT-2 tokenizer and roughly 500 MB of model weights
from Hugging Face. Later runs reuse the local cache, but can still contact the
Hub. Once cached, `HF_HUB_OFFLINE=1` enables offline execution.

CPU is the default. Optional backend selections:

```bash
.venv/bin/python -m inference_engine.generate_naive --device mps
.venv/bin/python -m inference_engine.generate_naive --device cuda
```

MPS requires supported Apple hardware and PyTorch support. CUDA requires a
compatible NVIDIA GPU, driver, and CUDA-enabled PyTorch installation. An
unavailable requested backend fails explicitly. The future benchmarking target
is a rented CUDA GPU; no cloud provisioning is part of this repository.

## What runs

The script loads `openai-community/gpt2` in FP32 and evaluation mode. The prompt
is `The capital of France is`, and the generation budget is ten new tokens.
These are currently constants in `main()`.

Each iteration forwards the entire current prefix with `use_cache=False`, takes
the final position's logits, chooses the highest-scoring vocabulary ID, and
appends that ID. Gradient tracking is disabled. Generation stops on the token
budget, an end-of-sequence token, or the context boundary. Our context policy
keeps the total prompt-plus-output length within the model's position capacity;
a prompt already at capacity produces zero new tokens.

The output trace reports each forward call's input length and selected token.
The displayed output includes the original prompt. Special tokens are hidden
in that final text; a generated end-of-sequence token still counts toward the
generated length. This is greedy continuation, not a guarantee of factual or
complete answers.

## Metric definitions

| Metric | Definition |
| --- | --- |
| Prompt length | Number of tokenizer IDs in the input prompt. |
| Generated length | Final sequence length minus prompt length, including EOS if generated. |
| Total local request latency | Tokenization through final output decoding, with model and tokenizer already loaded. Includes input preparation and generation. Excludes diagnostic printing, network transport, and queueing. |
| Generation-loop latency | Elapsed time around the loop, including correctness checks, selection, appends, trace recording, and stopping decisions. Excludes input preparation and output decoding. |
| Generated tokens/sec | Generated length divided by generation-loop seconds. |
| Parameter storage | Sum of parameter element counts times bytes per element; not runtime or peak memory. |
| Peak CUDA allocated memory | PyTorch tensor-allocation high-water mark during the request, including resident model allocations; excludes loading-time peaks. |
| Peak CUDA reserved memory | CUDA caching-allocator high-water mark in the same window, including retained reusable blocks. Not whole-device memory usage. |

Memory units are MiB (1,048,576 bytes). CUDA peaks are reset after model loading,
before the request, and read after it. Resetting statistics does not free model
memory. CPU and MPS report peak GPU memory as N/A; no MPS peak is estimated.
Synchronization at timing boundaries waits for CUDA/MPS work to finish.

## Validation and limits

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Tests use controlled scores and tiny CPU tensors, without model downloads. They
check prefix growth, generation-budget stopping, EOS stopping, context stopping,
and a prompt already at capacity. They exercise the actual script loop but do
not validate real model numerics or GPU memory counters.

Real GPT-2 CPU and MPS runs have been checked locally with the same ten-token
continuation. CUDA execution and allocator peaks still require validation on
CUDA hardware. On that hardware, run with `--device cuda` and verify successful
generation and reported peaks before drawing performance conclusions.

This is a checked baseline, not a controlled performance study: there is no
warm-up, repeated-trial aggregation, batching, or concurrency. Diagnostic
assertions and scalar transfers add overhead. Run without Python's `-O` flag,
which would remove assertions and change the measured work. Device-wide
synchronization and allocator peaks are unsuitable for independent attribution
to concurrent requests. Model revision and dependencies are not exactly pinned;
record those, hardware, dtype, and workload before comparing future runs.
