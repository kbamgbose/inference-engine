import unittest

from inference_engine.summarize_benchmark import distribution, summarize


def trial(tokens, seconds):
    return {
        "generated_tokens": tokens,
        "generation_seconds": seconds,
        "request_seconds": seconds + 0.1,
        "first_token_seconds": 0.1 if tokens else None,
        "inter_token_seconds": [0.01] * max(0, tokens - 1),
        "stop_reason": "generation limit",
    }


class SummaryTests(unittest.TestCase):
    def test_interpolated_percentiles_and_missing_observations(self):
        result = distribution([0.300, None, 0.100, 0.102])
        self.assertEqual(result["samples"], 3)
        self.assertAlmostEqual(result["p50"], 0.102)
        self.assertAlmostEqual(result["p99"], 0.29604)
        self.assertEqual(distribution([]), {"samples": 0, "p50": None, "p99": None})
        self.assertEqual(distribution([2]), {"samples": 1, "p50": 2, "p99": 2})

    def test_throughput_uses_total_work_over_total_time(self):
        result = summarize({"schema_version": 1, "requested_trials": 2,
                            "trials": [trial(10, 1), trial(10, 3)]})
        self.assertEqual(result["combined_generation_tokens_per_second"], 5)
        self.assertEqual(result["pooled_inter_token_seconds"]["samples"], 18)

    def test_zero_token_trial_is_not_a_first_token_observation(self):
        result = summarize({"schema_version": 1, "requested_trials": 2,
                            "trials": [trial(0, 1), trial(1, 1)]})
        self.assertEqual(result["first_token_seconds"]["samples"], 1)
        self.assertIsNone(result["pooled_inter_token_seconds"]["p99"])
        self.assertEqual(result["combined_generation_tokens_per_second"], 0.5)

    def test_incomplete_trials_are_rejected(self):
        with self.assertRaises(ValueError):
            summarize({"schema_version": 1, "requested_trials": 2,
                       "trials": [trial(1, 1)]})
