"""Execution context for interpreting saved benchmark measurements."""

import hashlib
import json
import os
import platform
import re

import torch
import transformers


def collect_metadata(model, tokenizer, device, prompt, input_ids, revision):
    """Collect outside timing; hashes identify inputs but do not reconstruct them."""
    token_bytes = json.dumps(input_ids, separators=(",", ":")).encode("utf-8")
    gpu = None
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        gpu = {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
            "multiprocessor_count": properties.multi_processor_count,
        }
    return {
        "software": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "transformers": transformers.__version__,
            "torch_cuda_build": torch.version.cuda,
        },
        "hardware": {
            "os": platform.system(),
            "os_release": platform.release(),
            "architecture": platform.machine(),
            "processor": platform.processor() or None,
            "logical_cpu_count": os.cpu_count(),
            "cuda_gpu": gpu,
        },
        "execution": {
            "torch_intraop_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "python_assertions_enabled": __debug__,
            "model_training": model.training if hasattr(model, "training") else None,
        },
        "artifacts": {
            "requested_revision": revision,
            "revision_is_commit_hash": re.fullmatch(r"[0-9a-fA-F]{40}", revision) is not None,
            "tokenizer_class": type(tokenizer).__name__,
        },
        "workload": {
            "prompt_utf8_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "token_ids_compact_json_sha256": hashlib.sha256(token_bytes).hexdigest(),
            "prompt_tokens": len(input_ids),
            "batch_size": 1,
            "decoding": "greedy",
            "eos_token_id": tokenizer.eos_token_id,
        },
    }
