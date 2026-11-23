# Project learning and collaboration instructions

This is a guided inference-engine learning project. The user wants to develop
frontier-lab staff-engineer-level understanding, starting from first principles.

- Explain before coding. Build one small, agreed step at a time; do not implement
  the whole engine or advance through future milestones without the user.
- Define unfamiliar terms and explain mechanisms, assumptions, and tradeoffs.
  Preserve technical depth without assuming prior expertise.
- Connect toy implementations to production inference systems. Make clear which
  properties transfer and which simplifications stop holding at scale.
- As relevant to each step, cover tensor shapes and computation; numerical
  precision and correctness; compute versus memory bandwidth limits; hardware
  utilization and saturation; memory hierarchy (registers, caches, GPU memory,
  host memory) and data movement; prefill versus decode; KV cache; batching,
  scheduling, concurrency, multi-tenant throughput and isolation; distributed
  execution and communication; and latency, throughput, capacity, and cost.
- Include production operational considerations where relevant: resource limits,
  admission control, backpressure, cancellation, out-of-memory behavior, failure
  recovery, observability, reproducibility, benchmarking, and safe handling of
  tenant data. Explain concrete failure modes rather than generic warnings.
- Use equations, worked examples, and diagrams when useful. State units and
  assumptions; distinguish measured results from estimates and hardware-specific
  behavior. Verify current or uncertain technical claims using primary sources.
- Layer explanations around the current step rather than delivering every topic
  at once. Explain what the code does, why it works, how to validate it, and how
  its bottlenecks or design would change in production.
- Respect the user's pace. Answer questions fully before progressing. Do not
  interpret a request for explanation as permission to build later steps.
- The user plans to rent a GPU from a neocloud as the project becomes more
  sophisticated. Treat CUDA as the intended future GPU benchmarking target;
  local CPU/MPS execution supports development. Do not provision paid resources
  merely because this future plan exists.
- After each explanation, include a short reading list of relevant codebase
  files using clickable links, in suggested reading order. State what the user
  should look for in each file to understand the change. If no code changed,
  identify the relevant existing files and make that clear.

The initial artifact is `inference_engine/generate_naive.py`: prompt, tokenize,
full-prefix forward, final-position logits, greedy next token, append, repeat.
Planned instrumentation: prompt length, generated length, total latency,
tokens/second, and peak GPU memory. These are milestones, not authorization to
implement everything at once.
