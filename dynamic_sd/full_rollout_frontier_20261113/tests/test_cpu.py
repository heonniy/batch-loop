import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import validate_frontier_receipts as V  # noqa: E402
import analyze_frontier as A  # noqa: E402

ARMS = ["AR", "DtV:d1:K2", "DtV:d1:K4", "WFD:d1"]


def meta(**kw):
    args = dict(model="ouro-1.4b", dtype="float32", seed=100, T_total=4, d=1, exit="live", tau=0.4, eps=None,
                Ks="2,4", wfd=1, max_new_tokens=64, max_prompt_tokens=2048, stops=1, dataset="gsm8k",
                num_fewshot=3, repeats=1)
    args.update(kw)
    return {"args": args, "arms": ARMS, "backend": "wavefront-eager (SDPA-fallback varlen)" if args["dtype"] == "float32"
            else "wavefront-eager (FA varlen)", "wf_commit": "w", "gpu": "H200", "torch": "2.8", "source_commit": "s",
            "target_exit_rule": {"rule": "x"}}


def rows(n=6, exact=True, speed=None):
    out = []
    for p in range(n):
        for arm in ARMS:
            sp = (speed or {}).get(arm, 1.0)
            out.append({"prompt_id": f"gsm8k:{p}", "prompt_hash": f"h{p}", "arm": arm, "repeat": 0,
                        "tokens_committed": 50, "eos_compare_calls": 50, "measured_wall_ms": 1000.0 / sp,
                        "prefill_ms": 50.0, "decode_ms": 1000.0 / sp - 50, "exact_match_to_AR": exact or arm == "AR",
                        "first_mismatch_vs_AR": None if (exact or arm == "AR") else 3, "correct": True,
                        "depths": [2, 3] if arm == "AR" else None, "accepted_count": 30, "draft_tokens": 40,
                        **{f: None for f in V.STAGE}})
    return out


def write(d, name, m, rs):
    p = Path(d) / "s1"
    p.mkdir(parents=True, exist_ok=True)
    f = p / f"{name}.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in rs))
    if m is not None:
        Path(str(f) + ".meta.json").write_text(json.dumps(m))
    return str(f)


class ValidatorTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def check(self, files):
        return V.validate_cell(V.load_cell_files(files))

    def test_pass(self):
        self.assertEqual(self.check([write(self.d, "c_o0n6", meta(), rows())])["status"], "passed")

    def test_cell_key(self):
        self.assertEqual(V.cell_key("/x/s1/ouro_fixed_gsm8k3_o100n50.jsonl"), "s1/ouro_fixed_gsm8k3")
        self.assertEqual(V.cell_key("/x/parity/h_e_spec_n8.jsonl"), "parity/h_e_spec")
        self.assertEqual(V.cell_key("/x/rep/h_e_gsm8k3_o0n16r5.jsonl"), "rep/h_e_gsm8k3")

    def test_mixed_model(self):
        r = self.check([write(self.d, "c_o0n6", meta(), rows()), write(self.d, "c_o6n6", meta(model="x"), rows())])
        self.assertEqual(r["status"], "failed")
        self.assertTrue(any("mixed model" in x for x in r["reasons"]))

    def test_mixed_seed(self):
        r = self.check([write(self.d, "c_o0n6", meta(), rows()), write(self.d, "c_o6n6", meta(seed=1), rows())])
        self.assertTrue(any("mixed seed" in x for x in r["reasons"]))

    def test_dory_live_rejected(self):
        r = self.check([write(self.d, "c_o0n6", meta(model="dory-4b"), rows())])
        self.assertTrue(any("Dory" in x for x in r["reasons"]))

    def test_incomplete_grid(self):
        rs = [x for x in rows() if not (x["prompt_id"] == "gsm8k:2" and x["arm"] == "DtV:d1:K4")]
        r = self.check([write(self.d, "c_o0n6", meta(), rs)])
        self.assertTrue(any("incomplete arm grid" in x for x in r["reasons"]))

    def test_prompt_hash(self):
        rs = rows(); rs[1]["prompt_hash"] = "other"
        r = self.check([write(self.d, "c_o0n6", meta(), rs)])
        self.assertTrue(any("prompt_hash" in x for x in r["reasons"]))

    def test_fabricated_stage_time(self):
        rs = rows(); rs[0]["verifier_ms"] = 0.0
        r = self.check([write(self.d, "c_o0n6", meta(), rs)])
        self.assertTrue(any("not instrumented" in x for x in r["reasons"]))

    def test_stop_matcher_calls(self):
        rs = rows(); rs[2]["eos_compare_calls"] = 49
        r = self.check([write(self.d, "c_o0n6", meta(), rs)])
        self.assertTrue(any("matcher" in x for x in r["reasons"]))

    def test_backend_mismatch(self):
        m = meta(); m["backend"] = "wavefront-eager (FA varlen)"
        r = self.check([write(self.d, "c_o0n6", m, rows())])
        self.assertTrue(any("backend" in x for x in r["reasons"]))

    def test_parity_failed_kept_as_data(self):
        r = self.check([write(self.d, "c_o0n6", meta(), rows(exact=False))])
        self.assertEqual(r["status"], "parity_failed")
        self.assertEqual(r["parity"]["inexact"], 18)

    def test_missing_meta(self):
        r = self.check([write(self.d, "c_o0n6", None, rows())])
        self.assertEqual(r["status"], "failed")


class AnalyzerTests(unittest.TestCase):
    def test_kopt_and_plateau(self):
        d = tempfile.mkdtemp()
        f = write(d, "c_o0n6", meta(dtype="bfloat16"), rows(speed={"DtV:d1:K2": 1.5, "DtV:d1:K4": 1.505, "WFD:d1": 1.7}))
        st = V.validate_cell(V.load_cell_files([f])) | {"files": [f]}
        o = A.analyze_cell("c", st, 200)
        self.assertAlmostEqual(o["arms"]["DtV:d1:K2"]["speedup_e2e"][0], 1.5, places=6)
        self.assertEqual(o["K_opt"]["best"], "DtV:d1:K4")
        self.assertTrue(o["K_opt"]["plateau"])
        self.assertEqual(o["ar_faster_than_bestK_frac"], 0.0)


if __name__ == "__main__":
    unittest.main()
