# Local cached-versus-uncached comparison

Completed 2026-11-21. This run exercises the local single-request benchmark
workflow on MPS. It is not a CUDA result or a concurrent-serving measurement.

## Workload and environment

- Model: `openai-community/gpt2`, FP32, evaluation mode.
- Model and tokenizer revision: `607a30d783dfa663caf39e06633721c8d4cfcd7e`.
- Prompt: `The capital of France is` (5 tokens).
- Output budget: 10 tokens; every trial generated all 10.
- Batch size: 1; greedy decoding; assertions enabled.
- Each mode: 3 warm-up requests, then 20 sequential measured trials.
- Order: uncached first, cached second, in separate processes with one model load each.
- Environment: Darwin 24.5.0, arm64, 8 logical CPUs, MPS backend.
- Python 3.12.3, PyTorch 2.14.1, Transformers 5.20.2.
- PyTorch CPU threads: 4 intra-op, 8 inter-op.

No paid resources were used. The exact Apple GPU model, clocks, and background
load were not recorded or controlled. CUDA memory counters are unavailable here.

## Results

| Metric | Uncached | Cached |
| --- | ---: | ---: |
| Request p50 (ms) | 257.41 | 161.42 |
| Request p99 (ms) | 362.08 | 209.68 |
| Generation-loop p50 (ms) | 256.62 | 160.61 |
| Generation-loop p99 (ms) | 361.10 | 206.67 |
| First-token p50 (ms) | 22.13 | 19.64 |
| First-token p99 (ms) | 27.55 | 36.61 |
| Pooled inter-token p50 (ms) | 25.58 | 15.68 |
| Pooled inter-token p99 (ms) | 54.78 | 20.84 |
| Combined generation tokens/sec | 37.96 | 61.83 |
| Peak CUDA allocated/reserved memory | unavailable | unavailable |

Each request-level distribution contains 20 observations. Each pooled token-gap
distribution contains 180 observations (9 gaps per request). Percentiles use
linear interpolation at index `(n - 1) * q`. These small, related samples do not
establish production tail latency.

Throughput is 200 generated tokens divided by total generation-loop seconds:
5.268885 seconds uncached and 3.234825 seconds cached. The cached rate is about
1.63 times the uncached rate for this workload. It excludes input preparation,
output text decoding, model loading, and gaps between requests.

Cached first-token p99 was higher, despite lower median first-token latency.
Both modes process the whole prompt on their first forward; later decode reuse
does not explain an improvement in that initial forward. These measurements do
not establish the cause of the first-token difference.

## Correctness and comparability checks

The JSON files have identical model, dtype, device, warm-up count, trial count,
budget, and metadata, including prompt/token fingerprints and pinned revision.
All trials stopped at the generation limit with 5 prompt and 10 generated tokens.

All 40 measured generated-token sequences, extracted from the console logs, were:

```text
[262, 3139, 286, 262, 4141, 2066, 11, 290, 262, 3139]
```

Every uncached trial forwarded `[5, 6, 7, 8, 9, 10, 11, 12, 13, 14]` token
positions per call; every cached trial forwarded `[5, 1, 1, 1, 1, 1, 1, 1, 1, 1]`.
This is 95 versus 14 forwarded positions, not a predicted latency ratio:
forward-call cost is not constant per position. Matching greedy IDs checks this
workload's output behavior, not equality of every floating-point logit.

## Reproduction

Run from the repository root with the model already cached. Use new output
filenames when repeating: the JSON writer refuses overwrites. The commands below
use a `repeat_` prefix so they do not replace the checked-in artifacts.

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m inference_engine.generate_naive \
  --device mps --revision 607a30d783dfa663caf39e06633721c8d4cfcd7e \
  --prompt 'The capital of France is' --max-new-tokens 10 \
  --warmup-runs 3 --trials 20 \
  --results-json benchmarks/repeat_mps_uncached.json

HF_HUB_OFFLINE=1 .venv/bin/python -m inference_engine.generate_naive \
  --device mps --revision 607a30d783dfa663caf39e06633721c8d4cfcd7e \
  --prompt 'The capital of France is' --max-new-tokens 10 \
  --warmup-runs 3 --trials 20 --use-kv-cache \
  --results-json benchmarks/repeat_mps_cached.json

.venv/bin/python -m inference_engine.summarize_benchmark benchmarks/local_mps_uncached.json
.venv/bin/python -m inference_engine.summarize_benchmark benchmarks/local_mps_cached.json
```

## Artifacts and limits

- Raw trials: [uncached](local_mps_uncached.json), [cached](local_mps_cached.json).
- Summaries: [uncached](local_mps_uncached_summary.json), [cached](local_mps_cached_summary.json).
- Console logs with token IDs: [uncached](local_mps_uncached.log), [cached](local_mps_cached.log).

This is one short prompt, one budget, and one execution order. Model loading and
warm-up are excluded; checks and scalar transfers remain inside generation timing.
Console printing occurs between requests. Ordering, thermal conditions, and
background activity could influence results. No profiler was attached, and no
claim about hardware saturation or the source of memory stalls follows from
these timings. Longer contexts, batching, CUDA memory measurements, profiling,
and concurrent load remain separate experiments.
