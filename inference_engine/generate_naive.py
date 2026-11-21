import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from inference_engine.benchmark_metadata import collect_metadata


def synchronize(device):
    """Wait for accelerator work before reading a wall-clock timer."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def prefill(model, prompt_ids):
    """Process a complete unpadded prompt and return logits plus a fresh KV cache.

    The caller supplies an evaluation-mode model and an inference-mode context.
    Token selection remains the caller's responsibility.
    """
    return model(input_ids=prompt_ids, use_cache=True, return_dict=True)


def decode_step(model, token_ids, cache):
    """Process one uncached token per sequence using the existing KV cache.

    For our single unpadded request, token_ids has shape [1, 1]. The model
    updates the cache in place and returns logits plus the updated cache.
    The caller supplies evaluation/inference mode and handles token selection.
    """
    return model(
        input_ids=token_ids, past_key_values=cache,
        use_cache=True, return_dict=True,
    )


@dataclass
class GenerationResult:
    """One request's output and diagnostics; retaining cache retains its tensors."""

    output_ids: torch.Tensor
    cache: object | None
    stop_reason: str
    trace: list[tuple[int, int, int]]
    forwarded_lengths: list[int]
    first_token_seconds: float | None
    inter_token_seconds: list[float]
    generation_seconds: float


def generate_tokens(
    model, input_tensor, *, max_new_tokens, use_kv_cache, eos_token_id, request_start,
):
    """Generate for one unpadded request with fresh state and an already loaded model.

    Caller supplies an evaluation-mode model, nonempty [1, P] token IDs on its
    device, a nonnegative budget, and the perf_counter timestamp from before
    tokenization. The input tensor is not modified in place. This function owns
    inference mode and generation timing; it neither loads weights nor prints.
    """
    device = input_tensor.device
    stop_reason = "generation limit"
    trace = []
    cache = None
    forwarded_lengths = []
    first_token_seconds = None
    previous_token_time = None
    inter_token_seconds = []

    # Finish input preparation before timing generation on asynchronous devices.
    synchronize(device)
    # Loop timing includes correctness checks; excludes loading and text I/O.
    generation_start = perf_counter()
    with torch.inference_mode():
        for step in range(max_new_tokens):
            prefix_length = input_tensor.shape[1]
            if prefix_length >= model.config.max_position_embeddings:
                stop_reason = "context limit"
                break

            if use_kv_cache:
                # Prefill once; then forward only the token not yet in the cache.
                if cache is None:
                    model_input = input_tensor
                    outputs = prefill(model, model_input)
                else:
                    model_input = input_tensor[:, -1:]
                    outputs = decode_step(model, model_input, cache)
                cache = outputs.past_key_values
                assert cache is not None
                assert cache.get_seq_length() == prefix_length
            else:
                model_input = input_tensor
                outputs = model(input_ids=model_input, use_cache=False, return_dict=True)
                assert outputs.past_key_values is None
            forwarded_lengths.append(model_input.shape[1])
            logits = outputs.logits
            expected_shape = (1, model_input.shape[1], model.config.vocab_size)
            assert tuple(logits.shape) == expected_shape
            assert torch.isfinite(logits).all().item(), "Logits contain NaN or infinity"
            assert not logits.requires_grad

            # The last input position predicts the token after the current prefix.
            next_token_logits = logits[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            input_tensor = torch.cat((input_tensor, next_token), dim=1)

            next_token_id = next_token.item()
            token_time = perf_counter()
            if first_token_seconds is None:
                # The first selected ID is now available to the host.
                first_token_seconds = token_time - request_start
            if previous_token_time is not None:
                inter_token_seconds.append(token_time - previous_token_time)
            previous_token_time = token_time
            trace.append((step + 1, prefix_length, next_token_id))

            if next_token_id == eos_token_id:
                stop_reason = "end of sequence"
                break
    synchronize(device)
    generation_seconds = perf_counter() - generation_start

    return GenerationResult(
        output_ids=input_tensor,
        cache=cache,
        stop_reason=stop_reason,
        trace=trace,
        forwarded_lengths=forwarded_lengths,
        first_token_seconds=first_token_seconds,
        inter_token_seconds=inter_token_seconds,
        generation_seconds=generation_seconds,
    )


def main():
    parser = argparse.ArgumentParser(description="Naive full-prefix greedy generation")
    parser.add_argument(
        "--revision", default="main",
        help="Shared model/tokenizer revision; use a full commit hash to pin artifacts",
    )
    parser.add_argument(
        "--prompt", default="The capital of France is",
        help="Nonempty input text (default: The capital of France is)",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=10,
        help="Maximum generated tokens, including EOS; must be nonnegative (default: 10)",
    )
    parser.add_argument(
        "--use-kv-cache", action="store_true",
        help="Reuse attention keys and values during generation",
    )
    parser.add_argument(
        "--warmup-runs", type=int, default=0,
        help="Unreported requests before measured trials (default: 0)",
    )
    parser.add_argument(
        "--trials", type=int, default=1,
        help="Sequential measured requests after warm-up; must be positive (default: 1)",
    )
    parser.add_argument(
        "--results-json", type=Path,
        help="Save measured trial records to a new JSON file after all trials finish",
    )
    parser.add_argument(
        "--inspect-kv-cache", action="store_true",
        help="Inspect the prompt cache and verify one cached continuation, then exit",
    )
    parser.add_argument(
        "--device", choices=("cpu", "cuda", "mps"), default="cpu",
        help="Execution backend (default: cpu)",
    )
    args = parser.parse_args()
    if not args.prompt:
        parser.error("--prompt must not be empty")
    if args.max_new_tokens < 0:
        parser.error("--max-new-tokens must be nonnegative")
    if args.warmup_runs < 0:
        parser.error("--warmup-runs must be nonnegative")
    if args.trials < 1:
        parser.error("--trials must be positive")
    if args.results_json is not None:
        if args.inspect_kv_cache:
            parser.error("--results-json cannot be combined with --inspect-kv-cache")
        if args.results_json.exists():
            parser.error("--results-json must name a new file")
        if not args.results_json.parent.is_dir():
            parser.error("--results-json parent directory must exist")
    if args.inspect_kv_cache and args.trials != 1:
        parser.error("--trials cannot be changed with --inspect-kv-cache")
    if args.inspect_kv_cache and args.warmup_runs:
        parser.error("--warmup-runs cannot be combined with --inspect-kv-cache")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS was requested but is not available")
    device = torch.device(args.device)

    prompt = args.prompt
    model_id = "openai-community/gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=args.revision)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=args.revision,
        dtype=torch.float32,
        use_safetensors=True,
    ).to(device)
    model.eval()

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    print(f"Model: {model_id}")
    print(f"Device: {model.device}")
    print(f"Parameter dtype: {model.dtype}")
    print(f"Parameters: {parameter_count:,}")
    # This excludes buffers, activations, and runtime overhead; it is not peak memory.
    print(f"Parameter storage: {parameter_bytes / 1024**2:.2f} MiB")

    if args.inspect_kv_cache:
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if not 0 < len(prompt_ids) < model.config.max_position_embeddings:
            parser.error("cache inspection requires a nonempty token sequence and room for one continuation")
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            outputs = prefill(model, prompt_tensor)
        cache = outputs.past_key_values
        assert cache is not None
        assert len(cache.layers) == model.config.n_layer
        assert cache.get_seq_length() == len(prompt_ids)
        first_layer = cache.layers[0]
        expected_shape = (
            1, model.config.n_head, len(prompt_ids),
            model.config.n_embd // model.config.n_head,
        )
        assert tuple(first_layer.keys.shape) == expected_shape
        assert tuple(first_layer.values.shape) == expected_shape
        print(f"Cache type: {type(cache).__name__}")
        print(f"Cached layers: {len(cache.layers)}")
        print(f"Cached token positions: {cache.get_seq_length()}")
        print(f"First-layer keys shape: {tuple(first_layer.keys.shape)}")
        print(f"First-layer values shape: {tuple(first_layer.values.shape)}")
        print(f"Cache dtype: {first_layer.keys.dtype}")

        # Reuse the prompt cache for exactly one new token, then compare scores.
        with torch.inference_mode():
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            cached_outputs = decode_step(model, next_token, cache)
            full_prefix = torch.cat((prompt_tensor, next_token), dim=1)
            reference = model(input_ids=full_prefix, use_cache=False, return_dict=True)
            cached_scores = cached_outputs.logits[:, -1, :]
            reference_scores = reference.logits[:, -1, :]
            torch.testing.assert_close(
                cached_scores, reference_scores, rtol=1e-4, atol=1e-4
            )
            assert torch.equal(cached_scores.argmax(-1), reference_scores.argmax(-1))
        cache = cached_outputs.past_key_values
        assert cache.get_seq_length() == len(prompt_ids) + 1
        print(f"Token fed into cached call: {tokenizer.decode(next_token[0].tolist())!r}")
        print(f"Cached call input length: {next_token.shape[1]}")
        print(f"Cached token positions after reuse: {cache.get_seq_length()}")
        print("Cached and full-prefix scores agree within tolerance; greedy IDs match")
        return

    for _ in range(args.warmup_runs):
        warmup_start = perf_counter()
        warmup_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if not warmup_ids:
            parser.error("--prompt must encode to at least one token")
        warmup_input = torch.tensor([warmup_ids], dtype=torch.long, device=device)
        warmup_result = generate_tokens(
            model, warmup_input, max_new_tokens=args.max_new_tokens,
            use_kv_cache=args.use_kv_cache, eos_token_id=tokenizer.eos_token_id,
            request_start=warmup_start,
        )
        tokenizer.decode(
            warmup_result.output_ids[0].tolist(),
            skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )
        # Release request tensors/cache; keep the model and reusable allocator blocks.
        del warmup_result, warmup_input, warmup_ids
    print(f"Warm-up requests completed: {args.warmup_runs}")

    trial_records = []
    for trial_index in range(args.trials):
        print(f"Measured trial: {trial_index + 1}/{args.trials}")
        # Local request timing: tokenization through final decoding, with model loaded.
        synchronize(device)
        if device.type == "cuda":
            # Reset the observation window, not allocations: resident weights count.
            torch.cuda.reset_peak_memory_stats(device)
        request_start = perf_counter()
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if not input_ids:
            parser.error("--prompt must encode to at least one token")
        # Add the batch dimension: one sequence containing all prompt token IDs.
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=model.device)
        prompt_length = input_tensor.shape[1]

        result = generate_tokens(
            model, input_tensor, max_new_tokens=args.max_new_tokens,
            use_kv_cache=args.use_kv_cache, eos_token_id=tokenizer.eos_token_id,
            request_start=request_start,
        )
        input_tensor = result.output_ids
        cache = result.cache
        stop_reason = result.stop_reason
        trace = result.trace
        forwarded_lengths = result.forwarded_lengths
        first_token_seconds = result.first_token_seconds
        inter_token_seconds = result.inter_token_seconds
        generation_seconds = result.generation_seconds

        output_text = tokenizer.decode(
            input_tensor[0].tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        request_seconds = perf_counter() - request_start

        # CUDA allocator peaks during the request, including the resident model.
        peak_allocated_bytes = None
        peak_reserved_bytes = None
        if device.type == "cuda":
            peak_allocated_bytes = torch.cuda.max_memory_allocated(device)
            peak_reserved_bytes = torch.cuda.max_memory_reserved(device)

        generated_length = input_tensor.shape[1] - prompt_length
        tokens_per_second = generated_length / generation_seconds
        assert 0 < generation_seconds <= request_seconds

        if args.results_json is not None:
            # Keep only host data; never retain a trial's tensors or KV cache.
            trial_records.append({
                "trial": trial_index + 1,
                "prompt_tokens": prompt_length,
                "generated_tokens": generated_length,
                "stop_reason": stop_reason,
                "request_seconds": request_seconds,
                "generation_seconds": generation_seconds,
                "first_token_seconds": first_token_seconds,
                "inter_token_seconds": list(inter_token_seconds),
                "generation_tokens_per_second": tokens_per_second,
                "peak_cuda_allocated_bytes": peak_allocated_bytes,
                "peak_cuda_reserved_bytes": peak_reserved_bytes,
            })

        # Decode and print diagnostics after stopping both timers.
        print(f"KV cache enabled: {args.use_kv_cache}")
        print(f"Tokens forwarded per call: {forwarded_lengths}")
        if cache is not None:
            cached_positions = cache.get_seq_length()
            # Count logical tensor payload after timing, independently of the formula.
            actual_cache_bytes = sum(
                tensor.numel() * tensor.element_size()
                for layer in cache.layers
                for tensor in (layer.keys, layer.values)
            )
            bytes_per_value = cache.layers[0].keys.element_size()
            head_dimension = model.config.n_embd // model.config.n_head
            bytes_per_token = (
                2 * model.config.n_layer * model.config.n_head
                * head_dimension * bytes_per_value
            )
            # This formula assumes GPT-2's uniform K/V layout and precision.
            expected_cache_bytes = input_tensor.shape[0] * cached_positions * bytes_per_token
            assert actual_cache_bytes == expected_cache_bytes, "KV payload disagrees with formula"
            print(f"Final cached positions: {cached_positions}")
            print(f"KV bytes/token/sequence: {bytes_per_token:,}")
            print(f"KV payload from tensors: {actual_cache_bytes:,} bytes")
            print(f"KV payload from formula: {expected_cache_bytes:,} bytes")
            print(f"KV tensor payload: {actual_cache_bytes / 1024**2:.6f} MiB")
        print(f"Prompt: {prompt}")
        print(f"Tokens: {tokenizer.convert_ids_to_tokens(input_ids)}")
        print(f"Token IDs: {input_ids}")
        print(f"Prompt length: {prompt_length} tokens")
        print(f"Decoded: {tokenizer.decode(input_ids, clean_up_tokenization_spaces=False)}")
        for step, prefix_length, token_id in trace:
            token_text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
            print(f"Step {step}: prefix={prefix_length}, token_id={token_id}, text={token_text!r}")

        print(f"Generated length: {generated_length} tokens")
        print(f"Total local request latency: {request_seconds:.4f} seconds")
        if first_token_seconds is not None:
            print(f"Local time to first token: {first_token_seconds:.4f} seconds")
        else:
            print("Local time to first token: N/A (no token generated)")
        if inter_token_seconds:
            gaps_ms = ", ".join(f"{seconds * 1000:.3f}" for seconds in inter_token_seconds)
            mean_gap_ms = sum(inter_token_seconds) / len(inter_token_seconds) * 1000
            print(f"Local inter-token gaps (ms): [{gaps_ms}]")
            print(f"Mean local inter-token latency: {mean_gap_ms:.3f} ms")
        else:
            print("Mean local inter-token latency: N/A (fewer than two tokens generated)")
        print(f"Generation-loop latency: {generation_seconds:.4f} seconds")
        print(f"Generated tokens/sec (generation loop): {tokens_per_second:.2f}")
        if peak_allocated_bytes is not None:
            print(f"Peak CUDA allocated memory (request): {peak_allocated_bytes / 1024**2:.2f} MiB")
            print(f"Peak CUDA reserved memory (request): {peak_reserved_bytes / 1024**2:.2f} MiB")
        else:
            print(f"Peak GPU memory: N/A (CUDA allocator counters require --device cuda; using {device.type})")
        print(f"Stop reason: {stop_reason}")
        print(f"Output: {output_text}")
        # Drop all references to this request's output/cache before the next peak reset.
        del result, input_tensor, cache

    if args.results_json is not None:
        report = {
            "schema_version": 1,
            "model_id": model_id,
            "dtype": str(model.dtype),
            "device": str(device),
            "use_kv_cache": args.use_kv_cache,
            "warmup_runs": args.warmup_runs,
            "requested_trials": args.trials,
            "max_new_tokens": args.max_new_tokens,
            "metadata": collect_metadata(
                model, tokenizer, device, prompt, input_ids, args.revision,
            ),
            "trials": trial_records,
        }
        # Exclusive creation also protects against a file appearing during the run.
        with args.results_json.open("x", encoding="utf-8") as results_file:
            json.dump(report, results_file, indent=2, allow_nan=False)
            results_file.write("\n")
        print(f"Saved trial results: {args.results_json}")


if __name__ == "__main__":
    main()
