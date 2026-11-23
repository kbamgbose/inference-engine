#!/usr/bin/env bash
# Prepare locally with --dry-run; execute only on an already configured CUDA host.
set -euo pipefail

mode=cached
dry_run=false
if [[ ${1:-} == cached || ${1:-} == uncached ]]; then
    mode=$1
    shift
fi
if [[ ${1:-} == --dry-run ]]; then
    dry_run=true
    shift
fi
if [[ $# -ne 0 ]]; then
    echo "Usage: bash scripts/profile_cuda.sh [cached|uncached] [--dry-run]" >&2
    exit 2
fi

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
python_bin=${PROFILE_PYTHON:-python}
output_dir="benchmarks/profile_${mode}.PREVIEW"

if [[ $dry_run == false ]]; then
    command -v nsys >/dev/null || { echo "Nsight Systems (nsys) is required." >&2; exit 1; }
    "$python_bin" -c 'import torch, transformers; assert torch.cuda.is_available(), "CUDA is unavailable"'
    mkdir -p benchmarks
    output_dir=$(mktemp -d "benchmarks/profile_${mode}.XXXXXX")
fi

app_args=(
    -m inference_engine.generate_naive
    --device cuda --nvtx --warmup-runs 3 --trials 1
    --revision 607a30d783dfa663caf39e06633721c8d4cfcd7e
    --prompt 'The capital of France is' --max-new-tokens 10
)
if [[ $mode == cached ]]; then
    app_args+=(--use-kv-cache)
fi

capture=(
    nsys profile
    --trace=cuda,nvtx --sample=none --cpuctxsw=none
    --discard-environment=true
    --capture-range=nvtx --nvtx-capture=trial/1 --capture-range-end=stop
    -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0
    -o "$output_dir/trace"
    "$python_bin" "${app_args[@]}"
)

if [[ $dry_run == true ]]; then
    echo "Preview only: no model, profiler, or output directory is created."
    printf '%q ' "${capture[@]}"
    printf '\n'
    exit 0
fi

# Record only explicit command arguments, not inherited environment variables.
printf '%q ' "${capture[@]}" > "$output_dir/command.txt"
printf '\n' >> "$output_dir/command.txt"
echo "Capturing $mode request; output directory: $output_dir"
if "${capture[@]}" > "$output_dir/capture.log" 2>&1; then
    if [[ ! -s "$output_dir/trace.nsys-rep" ]]; then
        echo "Profiler exited successfully but the expected report is missing or empty." >&2
        tail -n 25 "$output_dir/capture.log" >&2
        exit 1
    fi
else
    status=$?
    echo "Capture failed (exit $status). Log: $output_dir/capture.log" >&2
    tail -n 25 "$output_dir/capture.log" >&2
    exit "$status"
fi
tail -n 25 "$output_dir/capture.log"
echo "Report saved: $output_dir/trace.nsys-rep"
echo "Inspect trace contents before sharing; capture timing is not baseline timing."
