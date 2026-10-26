"""Check cached execution against the real script's full-prefix path."""

import io
from contextlib import redirect_stdout
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from inference_engine import generate_naive


class KVCacheTests(unittest.TestCase):
    def test_cached_logits_match_full_prefix_at_every_step(self):
        # Real transformer operations, small enough for an offline CPU test.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            model = GPT2LMHeadModel(GPT2Config(
                vocab_size=32, n_positions=32, n_embd=16, n_layer=2, n_head=2,
                bos_token_id=0, eos_token_id=31,
                resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
            )).eval()

        tokenizer = SimpleNamespace(
            # Exercise all ten generation steps, irrespective of selected IDs.
            eos_token_id=None,
            encode=lambda *args, **kwargs: [1, 2, 3, 4, 5],
            decode=lambda ids, **kwargs: " ".join(map(str, ids)),
            convert_ids_to_tokens=lambda ids: list(map(str, ids)),
        )

        def run_script(use_cache):
            scores = []
            forwarded_lengths = []

            def capture(module, args, kwargs, output):
                forwarded_lengths.append(kwargs["input_ids"].shape[1])
                # Copy now so later calls cannot change the recorded result.
                scores.append(output.logits[:, -1, :].detach().clone())

            handle = model.register_forward_hook(capture, with_kwargs=True)
            argv = ["generate_naive", "--device", "cpu"]
            if use_cache:
                argv.append("--use-kv-cache")
            try:
                with (
                    patch("sys.argv", argv),
                    patch.object(generate_naive.AutoTokenizer, "from_pretrained", return_value=tokenizer),
                    patch.object(generate_naive.AutoModelForCausalLM, "from_pretrained", return_value=model),
                    redirect_stdout(io.StringIO()),
                ):
                    generate_naive.main()
            finally:
                handle.remove()
            return scores, forwarded_lengths

        reference, full_lengths = run_script(use_cache=False)
        cached, cached_lengths = run_script(use_cache=True)
        self.assertEqual(full_lengths, list(range(5, 15)))
        self.assertEqual(cached_lengths, [5] + [1] * 9)
        self.assertEqual(len(reference), 10)
        self.assertEqual(len(cached), 10)
        for step, (actual, expected) in enumerate(zip(cached, reference)):
            with self.subTest(step=step + 1):
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
                self.assertTrue(torch.equal(actual.argmax(-1), expected.argmax(-1)))


if __name__ == "__main__":
    unittest.main()
