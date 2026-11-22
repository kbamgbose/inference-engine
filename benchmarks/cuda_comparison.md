# CUDA cached-versus-uncached comparison

For this short, single-request RTX 4090 workload, caching reduced peak allocated
memory by 6.89 MiB but did not improve measured throughput. This is a baseline
observation, not a diagnosis of the execution bottleneck.

## Evidence and workload

Raw results: [uncached](cuda_uncached.json), [cached](cuda_cached.json).
Derived summaries: [uncached](cuda_uncached_summary.json),
[cached](cuda_cached_summary.json). Summaries were regenerated from the saved
JSON with `inference_engine.summarize_benchmark`; no GPU rerun was performed.

- Model: `openai-community/gpt2`, FP32, evaluation mode.
- Model/tokenizer revision: `607a30d783dfa663caf39e06633721c8d4cfcd7e`.
- Prompt used in the session: `The capital of France is` (5 tokens).
- Greedy decoding, batch size 1, output budget 10 tokens.
- Each mode: 3 warm-ups followed by 20 sequential measured requests.
- Every measured request generated 10 tokens and stopped at the budget.
- Session execution order: uncached, then cached; separate processes.
- No profiler attached to these benchmark runs.

The two files have identical recorded metadata, model ID, dtype, device,
warm-up count, trial count, and output budget. Cache mode differs as intended.
Prompt and token-ID fingerprints match. Recorded environment:

- NVIDIA GeForce RTX 4090; compute capability 8.9; 128 multiprocessors.
- Device-reported memory: 25,252,724,736 bytes.
- Linux 6.8.0-138-generic, x86_64; 64 visible logical CPUs.
- Python 3.12.3; PyTorch 2.8.0+cu129; Transformers 5.24.0.
- PyTorch CUDA build 12.9 (this does not identify the installed driver version).
- PyTorch intra-op/inter-op threads: 32/64; Python assertions enabled.

Visible CPU count is not proof that all CPUs were dedicated to this Pod.
GPU clocks, background activity, host CPU allocation, and source commit were not
recorded in these result files.

## Measurements

| Metric | Uncached | Cached |
| --- | ---: | ---: |
| Request p50 (ms) | 66.41 | 69.12 |
| Request p99 (ms) | 77.57 | 71.76 |
| Generation-loop p50 (ms) | 66.14 | 68.88 |
| Generation-loop p99 (ms) | 77.20 | 71.40 |
| First-token p50 (ms) | 6.80 | 7.62 |
| First-token p99 (ms) | 7.87 | 8.63 |
| Pooled inter-token p50 (ms) | 6.59 | 6.64 |
| Pooled inter-token p99 (ms) | 8.47 | 8.39 |
| Combined generation tokens/sec | 147.93 | 145.00 |
| Peak CUDA allocated memory (MiB) | 494.09 | 487.20 |
| Peak CUDA reserved memory (MiB) | 532.00 | 532.00 |

Each request-level distribution has 20 samples; pooled inter-token latency has
180 gaps per mode. Percentiles linearly interpolate at index `(n - 1) * q`.
These are descriptive statistics from a small sample, not established service
tail-latency estimates. Gaps within a request are not independent requests.

Each mode generated 200 tokens. Total generation-loop time was 1.352014461 seconds
uncached and 1.379278647 seconds cached. Throughput is total tokens divided by
that total time, giving approximately 1.98% lower throughput for this cached run.
Generation timing includes checks, selection, scalar transfers, and loop work.
It excludes tokenization, final text decoding, and time between requests.

Allocated and reserved peaks were constant across all 20 trials within each mode.
The allocated difference is 6.88720703125 MiB. These peaks include resident model
weights and temporary tensors; they are not just KV-cache sizes. The earlier
cached smoke test separately verified a final KV payload of 1,032,192 bytes for
14 cached positions. The fifteenth sequence token was selected but not forwarded.

## Interpretation and limits

Although caching retains keys and values, forwarding fewer positions can reduce
temporary activations and logits. The measured allocated peak shows the net
reduction here; it does not attribute bytes to individual operations. Reserved
memory measures the allocator pool and need not shrink with the live allocation
peak. Neither counter is a measurement of all memory used by the entire device.

The data does not demonstrate a throughput benefit for this five-token prompt
and ten-token output. It also does not establish that caching is inherently
slower. Only one execution order and one workload were measured. Median token
gaps are nearly equal. CPU dispatch, synchronization, kernel execution, and cache
maintenance are possible contributors, but their relative costs remain unmeasured.
The first forward processes the entire prompt in both modes; later token reuse
cannot explain first-token differences by itself.

Earlier session output showed a successful CUDA cached-versus-full-prefix logit
comparison for one continuation, matching greedy IDs, and a full cached smoke
test producing the expected ten-token continuation. The uncached twenty-trial
console output also showed that continuation consistently. The saved JSON files
do not contain generated IDs, so they do not prove token equality for all cached
measured trials. Output lengths and stop reasons were verified from the JSON.

The MPS comparison used different PyTorch and Transformers versions. Cross-backend
timing differences therefore cannot be attributed solely to GPU hardware.

## Recompute the summaries

These commands run on CPU and load no model:

```bash
python -m inference_engine.summarize_benchmark benchmarks/cuda_uncached.json
python -m inference_engine.summarize_benchmark benchmarks/cuda_cached.json
```

## Next investigation

Profiling remains unfinished. A future focused Nsight Systems capture should
distinguish loading, warm-up, prefill, and decode before attributing timeline gaps.
The session's original reports were blocked from GitHub because they contained
a RunPod API key; they are not evidence used for this comparison. Future captures
must disable environment collection (`--discard-environment=true`) and be checked
before sharing. That option addresses environment capture, not every possible
source of sensitive trace content.

Use the timeline to select further measurements, including Nsight Compute for
specific kernels if needed. No conclusion about hardware saturation, launch
overhead, or memory stalls is established by the benchmark alone. Fixed batching,
concurrent scheduling, and quantization remain later experiments.
