# KV cache memory model

## Bytes per cached token

This section calculates the key/value tensor payload for one cached token in
one sequence. It uses the GPT-2 cache structure inspected by
`inference_engine/generate_naive.py --inspect-kv-cache`: twelve layers, with
first-layer key and value tensors shaped `(1, 12, 5, 64)` for our five-token
prompt, and dtype `torch.float32`.

The dimensions describe one sequence, twelve heads, five token positions, and
sixty-four numbers per head vector. To calculate bytes for just one token,
count the entries at one token position rather than multiplying by all five.

### Count one token at one layer

Each head retains a key vector containing 64 numbers and a value vector
containing 64 numbers. Across twelve heads:

```text
12 heads × (64 key values + 64 value values) = 1,536 numbers
```

FP32 stores each number in 32 bits, or four bytes:

```text
1,536 numbers × 4 bytes/number = 6,144 bytes per token per layer
```

### Include all layers

Every attention layer computes and retains its own keys and values. They are
not shared between layers. For twelve layers:

```text
6,144 bytes/token/layer × 12 layers = 73,728 bytes/token
73,728 / 1,024 = 72 KiB/token
```

Our result is **72 KiB per cached token, per sequence**. KiB means 1,024 bytes;
it is different from a decimal kB, which means 1,000 bytes.

### General formula and assumptions

Let L be the number of cached attention layers, H_kv the number of key/value
heads per layer, D_h the number of numbers in each head vector, and s the bytes
per cached number. With uniform layers and equal-size, equal-dtype keys and
values:

```text
bytes/token/sequence = 2 × L × H_kv × D_h × s
```

The factor 2 counts keys and values. For our GPT-2 configuration:

```text
2 × 12 × 12 × 64 × 4 = 73,728 bytes/token/sequence
```

Use the number of stored KV heads, not automatically the number of query heads.
Those counts are equal in our GPT-2 model. If layer layouts differ, calculate
each layer's key and value bytes separately and sum them instead of multiplying
one layer's size by L.

Use the actual cache dtype. Model weight precision alone does not establish
cache precision. If the same cache vectors were stored with two bytes per
number, their payload would be 36 KiB/token/sequence. Our implementation uses
FP32; this is a hypothetical precision comparison, not a code change.

### What this number measures

This is a calculated tensor payload, not measured peak GPU memory. It excludes
model weights, temporary tensors, allocator rounding, reserved reusable blocks,
and runtime overhead. It also assumes ordinary unquantized key/value tensors;
compressed representations can need additional metadata.

This distinction matters for capacity planning: cache payload consumes memory
alongside other allocations, so it cannot by itself establish whether an
entire inference workload fits on a GPU.

The cache shapes and dtype have been inspected. The tensor-payload comparison
is recorded in the validation section below.

## Growth with sequence length

Let S be the number of positions already processed and retained in the cache.
For one sequence in our GPT-2 FP32 configuration:

```text
cache payload in bytes = S × 73,728
```

This is linear growth: each additional cached position adds 73,728 bytes (72
KiB), regardless of how many positions are already cached. Doubling the cached
length doubles the logical payload under these assumptions.

| Cached positions S | Payload in bytes | Payload in MiB |
| ---: | ---: | ---: |
| 5 | 368,640 | 0.3515625 |
| 6 | 442,368 | 0.421875 |
| 14 | 1,032,192 | 0.984375 |
| 128 | 9,437,184 | 9 |
| 512 | 37,748,736 | 36 |
| 1,024 | 75,497,472 | 72 |

One MiB is 1,048,576 bytes. These are calculated payload sizes, not measured
GPU peaks. The 1,024-position row illustrates payload at GPT-2's position
capacity; it does not override the script's stopping policy.

### Processed positions versus selected output tokens

Our prompt has five tokens. The first forward pass processes and caches those
five positions, then supplies scores from which we select the first output
token. Selecting its ID does not compute its keys and values.

The second forward pass processes that selected token. The cache now contains
six positions, and the new scores let us select the second output token.
Generation repeats this pattern, stopping immediately after selecting the last
requested token (or EOS).

For a prompt of P tokens and G selected output tokens, with G >= 1:

```text
output sequence length = P + G
cached positions       = P + G - 1
cache payload          = (P + G - 1) × 73,728 bytes
```

For our P=5, G=10 run, the output contains fifteen IDs but the cache contains
fourteen positions. Its calculated payload is 1,032,192 bytes, or 0.984375 MiB.
The final output token has not been forwarded through the model. This counting
also applies if EOS causes an early stop. If no forward pass runs, there is no
populated cache; do not apply the G >= 1 formula in that case.

### Implications for serving requests

The model weights remain the same while a request's cache grows. For independent
sequences with cached lengths S_1 through S_N and no shared prefixes, their
combined logical K/V payload is:

```text
combined payload = 73,728 × (S_1 + ... + S_N) bytes
```

This assumes only those positions are stored; padding, reserved capacity, and
allocator overhead can increase physical memory consumption. Our current code
serves one sequence, so this multi-request equation is a planning extension,
not an implemented batching feature.

A server cannot assume that requests which fit now will still fit after more
generation. With N active sequences, one additional processed token for each
adds N × 72 KiB of logical payload in this configuration. Admission decisions
must account for possible future growth and leave room for weights, temporary
computation, and other memory consumers. Completed or cancelled requests must
eventually release their cache ownership so memory can be reused.

Linear storage growth does not imply constant decoding time. A new token still
attends to the retained keys and values, so attention must handle a larger
context as the sequence grows. The memory equation alone predicts neither
execution latency nor peak allocator memory.

## Validation against actual cache tensors

After generation and after stopping both timers, the script walks through every
cache layer and sums `tensor.numel() * tensor.element_size()` for its keys and
values. This uses the actual tensors' sizes and dtypes, without copying their
contents to the CPU.

Independently, it calculates bytes per token from GPT-2's layer count, head count,
head dimension, and the first key tensor's bytes per element. Multiplying by
batch size and cached positions gives the expected payload. This formula relies
on GPT-2's uniform key/value layout and precision. An assertion requires the
tensor-derived and formula-derived totals to match exactly.

Validated on MPS with FP32 GPT-2 on 2026-10-25:

```bash
.venv/bin/python -m inference_engine.generate_naive --device mps --use-kv-cache
```

```text
Final cached positions: 14
KV bytes/token/sequence: 73,728
KV payload from tensors: 1,032,192 bytes
KV payload from formula: 1,032,192 bytes
KV tensor payload: 0.984375 MiB
```

The ten-token continuation matched our baseline. The equality verifies logical
payload for this run, not every possible sequence length, peak GPU allocations,
or a performance speedup. It counts tensor elements rather than deduplicating
underlying storage or measuring allocator-reserved memory.

CUDA validation remains pending on rented hardware. Cached single-token CPU
execution previously encountered a native macOS bus error; this validation used
MPS and does not resolve that separate backend issue.
