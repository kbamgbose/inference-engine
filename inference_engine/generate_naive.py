import argparse
from time import perf_counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def synchronize(device):
    """Wait for accelerator work before reading a wall-clock timer."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def main():
    parser = argparse.ArgumentParser(description="Naive full-prefix greedy generation")
    parser.add_argument(
        "--device", choices=("cpu", "cuda", "mps"), default="cpu",
        help="Execution backend (default: cpu)",
    )
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS was requested but is not available")
    device = torch.device(args.device)

    prompt = "The capital of France is"
    model_id = "openai-community/gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
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

    max_new_tokens = 10
    stop_reason = "generation limit"
    trace = []

    # Local request timing: tokenization through final decoding, with model loaded.
    synchronize(device)
    if device.type == "cuda":
        # Reset the observation window, not allocations: resident weights count.
        torch.cuda.reset_peak_memory_stats(device)
    request_start = perf_counter()
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    # Add the batch dimension: one sequence containing all prompt token IDs.
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=model.device)
    prompt_length = input_tensor.shape[1]

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

            outputs = model(input_ids=input_tensor, use_cache=False, return_dict=True)
            logits = outputs.logits
            expected_shape = (1, prefix_length, model.config.vocab_size)
            assert tuple(logits.shape) == expected_shape
            assert torch.isfinite(logits).all().item(), "Logits contain NaN or infinity"
            assert not logits.requires_grad
            assert outputs.past_key_values is None

            # The last input position predicts the token after the current prefix.
            next_token_logits = logits[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            input_tensor = torch.cat((input_tensor, next_token), dim=1)

            next_token_id = next_token.item()
            trace.append((step + 1, prefix_length, next_token_id))

            if next_token_id == tokenizer.eos_token_id:
                stop_reason = "end of sequence"
                break
    synchronize(device)
    generation_seconds = perf_counter() - generation_start

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

    # Decode and print diagnostics after stopping both timers.
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
    print(f"Generation-loop latency: {generation_seconds:.4f} seconds")
    print(f"Generated tokens/sec (generation loop): {tokens_per_second:.2f}")
    if peak_allocated_bytes is not None:
        print(f"Peak CUDA allocated memory (request): {peak_allocated_bytes / 1024**2:.2f} MiB")
        print(f"Peak CUDA reserved memory (request): {peak_reserved_bytes / 1024**2:.2f} MiB")
    else:
        print(f"Peak GPU memory: N/A (CUDA allocator counters require --device cuda; using {device.type})")
    print(f"Stop reason: {stop_reason}")
    print(f"Output: {output_text}")


if __name__ == "__main__":
    main()
