"""Check NVTX scope behavior locally; these do not validate a CUDA trace."""

from time import perf_counter
import unittest
from unittest.mock import patch

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from inference_engine.generate_naive import generate_tokens, profile_range


class ProfilingRangeTests(unittest.TestCase):
    def test_disabled_range_does_not_call_nvtx(self):
        with patch.object(torch.cuda.nvtx, "range") as nvtx:
            with profile_range("trial/1", False):
                pass
        nvtx.assert_not_called()

    def test_range_is_closed_on_failure(self):
        with (
            patch.object(torch.cuda.nvtx, "range_push") as push,
            patch.object(torch.cuda.nvtx, "range_pop") as pop,
        ):
            with self.assertRaisesRegex(RuntimeError, "forward failed"):
                with profile_range("decode", True):
                    raise RuntimeError("forward failed")
        push.assert_called_once_with("decode")
        pop.assert_called_once_with()

    def test_cached_and_full_prefix_labels_preserve_output(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            model = GPT2LMHeadModel(GPT2Config(
                vocab_size=32, n_positions=16, n_embd=16, n_layer=2, n_head=2,
                bos_token_id=0, eos_token_id=31,
            )).eval()
        prompt = torch.tensor([[1, 2, 3]])
        results = []
        for cached, phases in ((True, ["prefill", "decode", "decode"]),
                               (False, ["full_prefix"] * 3)):
            events = []
            with (
                patch.object(torch.cuda.nvtx, "range_push", side_effect=lambda name: events.append(name)),
                patch.object(torch.cuda.nvtx, "range_pop", side_effect=lambda: events.append("pop")),
            ):
                with profile_range("trial/1", True):
                    results.append(generate_tokens(
                        model, prompt, max_new_tokens=3, use_kv_cache=cached,
                        eos_token_id=None, request_start=perf_counter(), nvtx=True,
                    ))
            expected = ["trial/1"]
            for phase in phases:
                expected.extend([phase, "pop"])
            self.assertEqual(events, expected + ["pop"])
        self.assertTrue(torch.equal(results[0].output_ids, results[1].output_ids))
        self.assertEqual(results[0].forwarded_lengths, [3, 1, 1])
        self.assertEqual(results[1].forwarded_lengths, [3, 4, 5])
