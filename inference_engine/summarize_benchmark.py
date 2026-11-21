"""Summarize saved single-request trials without loading a model."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path


def distribution(values):
    """Linear-interpolated percentiles; missing observations are excluded."""
    samples = sorted(value for value in values if value is not None)
    if any(not math.isfinite(value) or value < 0 for value in samples):
        raise ValueError("timings must be finite and nonnegative")

    def percentile(q):
        if not samples:
            return None
        index = (len(samples) - 1) * q
        lower = math.floor(index)
        upper = math.ceil(index)
        return samples[lower] + (samples[upper] - samples[lower]) * (index - lower)

    return {"samples": len(samples), "p50": percentile(0.50), "p99": percentile(0.99)}


def summarize(report):
    if report["schema_version"] != 1:
        raise ValueError("only benchmark schema version 1 is supported")
    trials = report["trials"]
    if not trials or len(trials) != report["requested_trials"]:
        raise ValueError("expected a nonempty, complete set of measured trials")
    for trial in trials:
        count = trial["generated_tokens"]
        if type(count) is not int or count < 0:
            raise ValueError("generated_tokens must be a nonnegative integer")
        if len(trial["inter_token_seconds"]) != max(0, count - 1):
            raise ValueError("inter-token gap count disagrees with generated length")
        if (trial["first_token_seconds"] is None) != (count == 0):
            raise ValueError("first-token availability disagrees with generated length")
        if not 0 < trial["generation_seconds"] <= trial["request_seconds"]:
            raise ValueError("expected 0 < generation_seconds <= request_seconds")

    total_tokens = sum(trial["generated_tokens"] for trial in trials)
    total_seconds = sum(trial["generation_seconds"] for trial in trials)
    return {
        "measured_trials": len(trials),
        "percentile_method": "linear interpolation at index (n - 1) * q",
        "request_seconds": distribution(trial["request_seconds"] for trial in trials),
        "generation_seconds": distribution(trial["generation_seconds"] for trial in trials),
        "first_token_seconds": distribution(trial["first_token_seconds"] for trial in trials),
        "pooled_inter_token_seconds": distribution(
            gap for trial in trials for gap in trial["inter_token_seconds"]
        ),
        "total_generated_tokens": total_tokens,
        "total_generation_seconds": total_seconds,
        "combined_generation_tokens_per_second": total_tokens / total_seconds,
        "generated_tokens_min": min(trial["generated_tokens"] for trial in trials),
        "generated_tokens_max": max(trial["generated_tokens"] for trial in trials),
        "stop_reason_counts": dict(Counter(trial["stop_reason"] for trial in trials)),
        "interpretation": "Empirical percentiles; small sample counts do not establish tail latency. "
        "Throughput excludes preparation, output decoding, and time between trials.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_json", type=Path)
    args = parser.parse_args()
    try:
        with args.results_json.open(encoding="utf-8") as source:
            summary = summarize(json.load(source))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(f"cannot summarize results: {error}")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
