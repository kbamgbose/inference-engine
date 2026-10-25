"""Exercise the real loop with tiny tensors and controlled next-token scores."""

import io
from contextlib import redirect_stdout
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from inference_engine import generate_naive


class ControlledModel:
    def __init__(self, capacity, selected_token):
        self.config = SimpleNamespace(max_position_embeddings=capacity, vocab_size=4)
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.selected_token = selected_token
        self.prefixes = []

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def parameters(self):
        return iter([torch.zeros(1)])

    def __call__(self, input_ids, use_cache, return_dict):
        assert not use_cache
        assert return_dict
        assert torch.is_inference_mode_enabled()
        self.prefixes.append(input_ids[0].tolist())
        scores = torch.zeros(1, input_ids.shape[1], self.config.vocab_size)
        scores[:, :, self.selected_token] = 10.0
        return SimpleNamespace(logits=scores, past_key_values=None)


class GenerationTests(unittest.TestCase):
    def run_case(self, capacity=32, selected_token=2):
        model = ControlledModel(capacity, selected_token)
        tokenizer = SimpleNamespace(
            eos_token_id=3,
            encode=lambda *args, **kwargs: [1] * 5,
            decode=lambda ids, **kwargs: " ".join(map(str, ids)),
            convert_ids_to_tokens=lambda ids: list(map(str, ids)),
        )
        output = io.StringIO()
        with (
            patch("sys.argv", ["generate_naive", "--device", "cpu"]),
            patch.object(generate_naive.AutoTokenizer, "from_pretrained", return_value=tokenizer),
            patch.object(generate_naive.AutoModelForCausalLM, "from_pretrained", return_value=model),
            redirect_stdout(output),
        ):
            generate_naive.main()
        return model, output.getvalue()

    def test_generation_limit_and_prefix_growth(self):
        model, output = self.run_case()
        self.assertEqual(len(model.prefixes), 10)
        for step, prefix in enumerate(model.prefixes):
            self.assertEqual(prefix, [1] * 5 + [2] * step)
        self.assertIn("Generated length: 10 tokens", output)
        self.assertIn("Stop reason: generation limit", output)
        self.assertIn("Peak GPU memory: N/A", output)

    def test_eos_is_counted_and_stops_generation(self):
        model, output = self.run_case(selected_token=3)
        self.assertEqual(len(model.prefixes), 1)
        self.assertIn("Generated length: 1 tokens", output)
        self.assertIn("Stop reason: end of sequence", output)

    def test_context_capacity_stops_before_another_forward(self):
        model, output = self.run_case(capacity=7)
        self.assertEqual([len(prefix) for prefix in model.prefixes], [5, 6])
        self.assertIn("Generated length: 2 tokens", output)
        self.assertIn("Stop reason: context limit", output)

    def test_prompt_at_capacity_performs_no_forward(self):
        model, output = self.run_case(capacity=5)
        self.assertEqual(model.prefixes, [])
        self.assertIn("Generated length: 0 tokens", output)
        self.assertIn("Stop reason: context limit", output)


if __name__ == "__main__":
    unittest.main()
