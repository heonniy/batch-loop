"""Pure CPU path matching and gate-summary tests."""
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from dynamic_sd.gate_probe_common import load_json, sha256_file, summarize, validate_trace


class GateCommonTest(unittest.TestCase):
    def test_read_gzip(self):
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "test.json.gz"
            with gzip.open(f, "wt", encoding="utf-8") as out:
                json.dump({"some": [1, 2]}, out)
            self.assertEqual(load_json(f), {"some": [1, 2]})
            self.assertEqual(len(sha256_file(f)), 64)

    def test_reference_shape(self):
        example = {"idx": 2, "prefix_len": 32,
                   "tokens_ar": [21, 22, 23], "depths_ar": [2, 4]}
        self.assertEqual(validate_trace(example, max_positions=1, max_depth=4), 1)
        for bad in (dict(example, depths_ar=[2]),
                    dict(example, depths_ar=[2, 5]),
                    dict(example, depths_ar=[0, 1])):
            with self.assertRaises(ValueError):
                validate_trace(bad, max_depth=4)

    def test_paired_deltas(self):
        base = dict(idx=1, repeat=0, input_digest="same", predicted_digest="same",
                    depths=[2, 2], decode_positions=2, depths_match_reference=True,
                    prefill_first_match_reference=True,
                    prediction_match_fraction_to_reference=1.0)
        data = [
            dict(base, mode="replay", decode_wall_ms=10.),
            dict(base, mode="shadow_gpu", decode_wall_ms=12., gate_event_ms=1.),
            dict(base, mode="shadow_readback", decode_wall_ms=20., readback_wait_ms=6.),
            dict(base, mode="live", decode_wall_ms=80., depths=[4, 4]),
        ]
        r = summarize(data)
        self.assertEqual(r["valid_triplets"], 1)
        self.assertAlmostEqual(r["paired_metrics"]["delta_B_minus_A_wall_ms_per_position"]["median"], 1.)
        self.assertAlmostEqual(r["paired_metrics"]["delta_C_minus_B_wall_ms_per_position"]["median"], 4.)
        self.assertNotIn("live", str(r["paired_metrics"]))
        data[2]["predicted_digest"] = "different"
        self.assertEqual(summarize(data)["valid_triplets"], 0)
        self.assertEqual(len(summarize(data)["excluded_triplets"]), 1)
        data[2]["predicted_digest"] = "same"
        data[2]["prediction_match_fraction_to_reference"] = 0.5
        self.assertEqual(summarize(data)["valid_triplets"], 0)
        self.assertIn("source AR", summarize(data)["excluded_triplets"][0]["reason"])

    def test_duplicate_trial_rejected(self):
        row = dict(idx=1, repeat=0, mode="replay")
        with self.assertRaises(ValueError):
            summarize([row, row])


if __name__ == "__main__":
    unittest.main()
