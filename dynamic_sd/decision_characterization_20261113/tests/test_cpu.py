import gzip
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cf_receipts import ReceiptError, assert_disjoint, prompt_split, validate  # noqa: E402
import decision_decompose as dd  # noqa: E402
import conditioned_crossover as cc  # noqa: E402

ACTIONS = ["AR", "DtV:d1:K2", "DtV:d1:K4", "DtV:d2:K2", "DtV:d2:K4"]


def make_rows(n_prompts=6, n_ck=3, seed=0):
    rng = random.Random(seed)
    rows = []
    for p in range(n_prompts):
        for c in range(n_ck):
            k = 4 + 16 * c
            D = rng.choice([1, 2, 3, 4])
            for a in ACTIONS:
                kind, d, K = (("AR", None, None) if a == "AR" else ("DtV", int(a[5]), int(a[8])))
                if kind == "AR":
                    tok, info, base = 1, {"depth": D}, 10.0
                else:
                    acc = rng.randint(0, K)
                    tok = acc + 1
                    info = {"n_acc": acc, "depths": [D] * (K + 1), "verify_ticks": max(0, D - d)}
                    base = 10.0 + 6 * K * d
                for r in range(3):
                    rows.append({"prefix_id": f"{p}:{k}", "prefix_hash": f"h{p}:{k}", "target_hash": f"t{p}",
                                 "exit_policy": "ouro:tau0.4", "kv_policy": "FULL", "batch_regime": "B1",
                                 "backend": "x", "precision": "float32", "action": a,
                                 "wall_ms": base * (1 + 0.05 * rng.random()), "committed_tokens": tok,
                                 "target_exact": True, "measured": True, "repeat": r, "prompt_idx": p, "k": k,
                                 "pos": 100 + k, "ref_next_depths": [D] * 5, "info": info})
    return rows


class ValidationTests(unittest.TestCase):
    def test_ok(self):
        cell = validate(make_rows(), expected_repeats=3)
        self.assertEqual(cell["audit"]["prefixes_kept"], 18)

    def test_missing_grid(self):
        rows = [r for r in make_rows() if not (r["prefix_id"] == "0:4" and r["action"] == "DtV:d2:K4")]
        with self.assertRaisesRegex(ReceiptError, "missing action grid"):
            validate(rows)

    def test_missing_repeat(self):
        rows = [r for r in make_rows() if not (r["prefix_id"] == "0:4" and r["action"] == "AR" and r["repeat"] == 2)]
        with self.assertRaisesRegex(ReceiptError, "repeats"):
            validate(rows)

    def test_nonexact_prefix_excluded_as_whole_grid(self):
        rows = make_rows()
        for r in rows:
            if r["prefix_id"] == "1:20" and r["action"] == "DtV:d1:K4":
                r["target_exact"] = False
        cell = validate(rows)
        self.assertIn("1:20", cell["excluded"])
        self.assertNotIn("1:20", cell["prefixes"])
        self.assertEqual(cell["audit"]["inexact_trials"], 3)

    def test_changed_prefix(self):
        rows = make_rows()
        next(r for r in rows if r["prefix_id"] == "2:4" and r["action"] == "AR")["prefix_hash"] = "other"
        with self.assertRaisesRegex(ReceiptError, "changed prefix"):
            validate(rows)

    def test_zero_wall(self):
        rows = make_rows()
        rows[7]["wall_ms"] = 0.0
        with self.assertRaisesRegex(ReceiptError, "zero/invalid wall"):
            validate(rows)

    def test_unmeasured(self):
        rows = make_rows()
        rows[3]["measured"] = False
        with self.assertRaisesRegex(ReceiptError, "not physically measured"):
            validate(rows)

    def test_repeat_geometry(self):
        rows = make_rows()
        t = next(r for r in rows if r["prefix_id"] == "0:4" and r["action"] == "DtV:d1:K2" and r["repeat"] == 1)
        t["info"] = dict(t["info"], verify_ticks=99)
        with self.assertRaisesRegex(ReceiptError, "repeat geometry"):
            validate(rows)

    def test_prompt_leakage(self):
        folds = prompt_split(range(10), 2)
        assert_disjoint(folds[0], folds[1])
        with self.assertRaisesRegex(ReceiptError, "leakage"):
            assert_disjoint([1, 2, 3], [3, 4])


class DecomposeTests(unittest.TestCase):
    def _cell_file(self, rows):
        d = tempfile.mkdtemp()
        p = Path(d) / "c.jsonl.gz"
        with gzip.open(p, "wt") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return p

    def test_classes_nested_and_calibration_disjoint(self):
        res = dd.analyze_cell(self._cell_file(make_rows(8, 4)), n_boot=50)
        for f in res["folds"]:
            self.assertFalse(set(f["calib_prompts"]) & set(f["eval_prompts"]))
            C = f["classes"]
            self.assertTrue(set(C["S"]) <= set(C["A"]) <= set(C["K"]) <= set(C["Joint"]))
            self.assertTrue(set(C["A"]) <= set(C["d"]) <= set(C["Joint"]))

    def test_aggregate_selection_is_nested_in_sample(self):
        cell = validate(make_rows(8, 4, seed=3), expected_repeats=3)
        prefs = cell["prefixes"]
        def train_rate(cands, r):
            pick = dd.choose(prefs, cands, r, "aggregate")
            T = M = 0
            for pid, a in pick.items():
                for x in prefs[pid]["actions"][a]:
                    if x["repeat"] != r:
                        T += x["committed_tokens"]; M += x["wall_ms"]
            return T / M
        small = ["AR", "DtV:d1:K2", "DtV:d1:K4"]
        for r in (0, 1, 2):
            self.assertGreaterEqual(train_rate(ACTIONS, r) + 1e-12, train_rate(small, r))


class CrossoverTests(unittest.TestCase):
    def test_derived_rows(self):
        t = {"committed_tokens": 2, "info": {"n_acc": 1, "depths": [3, 3, 4, 2, 3], "verify_ticks": 2}}
        d = cc.derived("DtV:d1:K4", t)
        self.assertEqual(d["core_rows"], 15)
        self.assertEqual(d["core_calls"], 5 + 2)
        self.assertEqual(d["rejected_rows"], 4 + 2 + 3)

    def test_classify(self):
        rows = [("a", 50, 1.2, 1.1, 1.3, False), ("b", 50, 0.8, 0.7, 0.9, False)]
        self.assertEqual(cc.classify(rows), "reproducible crossover")
        rows = [("a", 50, 1.2, 0.9, 1.3, False), ("b", 50, 0.9, 0.7, 1.1, False)]
        self.assertTrue(cc.classify(rows).startswith("noisy"))


if __name__ == "__main__":
    unittest.main()
