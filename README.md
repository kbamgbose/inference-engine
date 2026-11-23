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
defaults to `The capital of France is`, and the generation budget defaults to
ten new tokens. Override them with:

```bash
.venv/bin/python -m inference_engine.generate_naive --device cpu \
  --prompt "The capital of France is" --max-new-tokens 20
```

The prompt must encode to at least one token. The budget must be nonnegative;
zero performs no generation forwards. EOS or the context limit can produce
fewer tokens than requested, so compare actual generated lengths as well as
budgets. `--inspect-kv-cache` instead performs its fixed one-continuation check,
independent of the budget, and requires room for that continuation.

Use `--warmup-runs 3` to execute three requests with the same prompt, budget,
and cache mode before measuring one request. The default is zero. Warm-up
includes tokenization, generation, and output decoding; its timings are discarded.
Each run creates a fresh KV cache and releases its result before the next run.
CUDA peak statistics are reset afterward, but the allocator may retain reserved
blocks from warm-up. No allocator cache flush is performed. Warm-up counts must
be nonnegative and cannot be combined with `--inspect-kv-cache`. A request that
performs no forwards does not warm model execution. A chosen warm-up count does
not by itself establish stable performance.

Use `--trials 5` to measure five sequential requests after warm-up. The default
is one and the count must be positive. For example, `--warmup-runs 2 --trials 5`
runs seven requests with one model load and reports only the final five. Each
trial starts from the original prompt with a fresh cache, resets CUDA peaks,
and releases its output/cache before the next trial. Allocator reservations may
persist. Each trial prints its own metrics; saved trials can be summarized offline.
Printing is outside request timers but introduces gaps between
trials, so this is not a sustained-load or concurrent-serving measurement.
Cache inspection supports only the default trial count.

Add `--results-json benchmarks/trials.json` to save individual measured trials
after all trials finish. The parent directory must exist and the file must be
new; existing files are never overwritten. This option cannot be combined with
cache inspection. JSON schema version 1 contains basic run settings and a
`trials` list, excluding warm-up. Each trial stores actual token counts, stopping
reason, request/generation/first-token times, inter-token gaps, generation
tokens/sec, and CUDA allocated/reserved peaks. Times are in seconds and memory
in bytes, as named in the fields. Unavailable measurements are `null`; inter-token
gaps are an empty list when fewer than two tokens were generated. Values retain
their numerical precision instead of console rounding.

Saving occurs outside request timers. Records retain no tensors or KV caches.
No partial results are saved if execution fails before the final write; a failure
during writing can leave an incomplete file. Prompt text and output content are
omitted from JSON, but remain in console diagnostics.

The `metadata` field records Python/PyTorch/Transformers versions, PyTorch's CUDA
build version (not the installed driver version), OS/CPU information, PyTorch CPU
thread settings, assertion status, and evaluation/training mode. CUDA runs also
record GPU name, memory capacity, compute capability, and multiprocessor count.
Unavailable hardware details are null; exact Apple GPU identification is not
collected. Metadata collection is outside request timers.

`--revision` selects the same model/tokenizer repository revision, defaulting to
`main`. For comparisons, supply a full 40-character commit hash to pin artifacts.
The report records the requested revision and whether it has commit-hash form;
it does not claim to resolve a branch or tag to an immutable snapshot.
SHA-256 fingerprints identify the UTF-8 prompt and token IDs encoded as compact
JSON (no separator whitespace). They support input matching, not reconstruction
or anonymization; retain the workload separately. Batch size, greedy decoding,
EOS ID, and tokenizer class are also recorded. Driver version, clocks, background
load, and project source revision remain outside this metadata record.

Summarize a saved file without loading a model:

```bash
.venv/bin/python -m inference_engine.summarize_benchmark benchmarks/trials.json
```

The command prints JSON with sample counts and p50/p99 for request, generation,
first-token, and pooled inter-token times, all in seconds. Percentiles linearly
interpolate sorted samples at index `(n - 1) * q`, where q is 0.50 or 0.99.
Missing first-token times are excluded; empty distributions have null percentiles.
Inter-token samples pool all individual gaps, so longer outputs contribute more
observations. These are empirical percentiles, not reliable estimates of rare
latency events from a small trial count.

Combined generation throughput is total generated tokens divided by total
generation-loop seconds, not the arithmetic mean of per-trial rates. It excludes
input preparation, output decoding, and gaps between trials. The summary also
reports output-length range and stopping-reason counts to expose early stopping.
The source file is unchanged; raw per-trial memory measurements remain there.

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

Optional `--nvtx` adds profiler labels during normal `--device cuda` generation.
It cannot be combined with cache inspection. Outer ranges `warmup/1`, `trial/1`,
etc. cover input preparation through final text decoding, excluding diagnostic
printing. Measured-trial ranges also exclude the preceding device synchronization
and CUDA peak reset. Nested `prefill`, `decode`, or `full_prefix` ranges cover
model calls only; token selection and loop checks are outside those inner ranges.
No extra synchronization is added. These are CPU scopes: asynchronously launched
GPU work can outlive an inner range, so correlate kernels with their launching
calls instead of treating range width as GPU execution time.

Labels are disabled by default and saved reports record `nvtx_enabled`. Keep
baseline performance runs unprofiled and without `--nvtx`. The disabled path uses
a no-op Python context manager; enabling labels adds instrumentation overhead.
Future Nsight Systems captures should include `--trace=cuda,nvtx` and
`--discard-environment=true`. Labels contain only phase names and counters, but
other trace content still needs review before sharing. Local tests can verify
label nesting with mocked NVTX calls; actual trace visibility requires CUDA.

| Metric | Definition |
| --- | --- |
| Prompt length | Number of tokenizer IDs in the input prompt. |
| Generated length | Final sequence length minus prompt length, including EOS if generated. |
| Total local request latency | Tokenization through final output decoding, with model and tokenizer already loaded. Includes input preparation and generation. Excludes diagnostic printing, network transport, and queueing. |
| Local time to first token | Same request start through the first selected token ID becoming available as a Python integer. Includes tokenization, input preparation, the first forward, checks, and selection; excludes output text decoding, queueing, and transport. Reports N/A if no token is generated. This is not prefill-forward latency alone. |
| Generation-loop latency | Elapsed time around the loop, including correctness checks, selection, appends, trace recording, and stopping decisions. Excludes input preparation and output decoding. |
| Local inter-token gaps / mean latency | Time between successive selected token IDs becoming available as Python integers. G generated tokens give G−1 gaps; the mean is N/A for fewer than two tokens. Includes forward calls, checks, selection, and intervening loop overhead; excludes initial first-token wait and output text decoding/transport. EOS counts as a generated token. These are host-observed intervals, not isolated GPU-kernel durations. |
| Generated tokens/sec | Generated length divided by generation-loop seconds. |
| Parameter storage | Sum of parameter element counts times bytes per element; not runtime or peak memory. |
| Peak CUDA allocated memory | PyTorch tensor-allocation high-water mark during the request, including resident model allocations; excludes loading-time peaks. |
| Peak CUDA reserved memory | CUDA caching-allocator high-water mark in the same window, including retained reusable blocks. Not whole-device memory usage. |

Memory units are MiB (1,048,576 bytes). CUDA peaks are reset after model loading,
before the request, and read after it. Resetting statistics does not free model
memory. CPU and MPS report peak GPU memory as N/A; no MPS peak is estimated.
Synchronization at timing boundaries waits for CUDA/MPS work to finish.

## Validation and limits

Prepare a focused CUDA capture locally, without a GPU or Nsight installation:

```bash
bash scripts/profile_cuda.sh cached --dry-run
bash scripts/profile_cuda.sh uncached --dry-run
```

Dry runs only print the command. During a later GPU session, activate the project
environment and run `bash scripts/profile_cuda.sh cached` (or `uncached`). The
script uses `python` from that environment; `PROFILE_PYTHON` can select another
interpreter. It does not provision a Pod or install dependencies. The model must
be cached or downloadable on the execution host.

The fixed workload matches the single-request comparison: pinned GPT-2 FP32,
prompt `The capital of France is`, ten output tokens, three warm-ups, one trial.
Nsight captures only `trial/1`, with CUDA and NVTX tracing, CPU sampling disabled,
and environment collection disabled. `NSYS_NVTX_PROFILER_REGISTER_ONLY=0` enables
capture-trigger matching for PyTorch's ordinary NVTX strings; this adds profiler
overhead. No unprofiled benchmark JSON is produced by this capture script.

Each actual run creates a unique `benchmarks/profile_MODE.XXXXXX` directory with
`trace.nsys-rep`, `capture.log`, and `command.txt`. Failures return a nonzero exit
code and retain the log; a missing/empty expected report also fails. File existence
does not establish that the trace contains the expected activity. Review it before
sharing. Environment suppression is not general secret sanitization. Output lives
on the execution machine: this script does not back it up or guarantee persistence
when a cloud instance stops.

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

This is a checked baseline with optional warm-up, not a controlled performance
study: offline trial aggregation is available, but batching and concurrency are not. Diagnostic
assertions and scalar transfers add overhead. Run without Python's `-O` flag,
which would remove assertions and change the measured work. Device-wide
synchronization and allocator peaks are unsuitable for independent attribution
to concurrent requests. Model revision and dependencies are not exactly pinned;
record those, hardware, dtype, and workload before comparing future runs.
