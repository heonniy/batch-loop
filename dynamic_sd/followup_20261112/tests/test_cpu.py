import copy
import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from oracle_frontier import analyze
from compare_shadow import compare


def row(prefix, action, wall, tokens, **kw):
    d=dict(prefix_id=prefix,prefix_hash="h"+prefix,target_hash="target",
           exit_policy="ouro:tau0.4",kv_policy="FULL",batch_regime="B1",
           backend="wavefront-eager",precision="fp32",action=action,
           wall_ms=wall,committed_tokens=tokens,target_exact=True,measured=True,repeat=0)
    d.update(kw)
    return d


class OracleTests(unittest.TestCase):
    def test_matched_oracle(self):
        rows=[row("p1","AR",10,1),row("p1","SD",7,1),
              row("p2","AR",10,1),row("p2","SD",15,1)]
        result=analyze(rows)
        self.assertEqual(result["n_prefixes"],2)
        self.assertEqual(result["best_static_action"],"AR")
        self.assertGreater(result["optimistic_ratio_vs_best_static"],1)
        self.assertEqual(result["by_prefix"]["p1"]["oracle_action"],"SD")
        self.assertEqual(result["by_prefix"]["p2"]["oracle_action"],"AR")

    def test_reject_missing_pair(self):
        with self.assertRaisesRegex(ValueError,"incomplete action"):
            analyze([row("p1","AR",1,1),row("p1","SD",1,1),row("p2","AR",1,1)])

    def test_reject_semantic_mismatch(self):
        with self.assertRaisesRegex(ValueError,"nonidentical prefix"):
            analyze([row("p1","AR",1,1),row("p1","SD",1,1,kv_policy="last")])

    def test_reject_mismatched_output(self):
        with self.assertRaisesRegex(ValueError,"target-exact"):
            analyze([row("p1","AR",1,1),row("p1","SD",1,1,target_exact=False)])

    def test_reject_unmeasured(self):
        with self.assertRaisesRegex(ValueError,"physically measured"):
            analyze([row("p1","AR",1,1,measured=False)])


def probe(mode, wall):
    arg=dict(model="ouro-1.4b",dataset="gsm8k",offset=0,limit=1,dtype="float32",
             T_total=4,T_draft=1,K=4,max_new_tokens=8,max_prompt_tokens=100,
             seed=100,sampling="greedy",kv_s=0,exit="replay",replay_from="ref.json",arms="ar,wfd")
    sample=dict(idx=0,tokens_ar=[1,2],rcalls_ar=3,rows_ar=3,width_ar={"1":3},depths_ar=[3],
                tokens_wfd=[1,2],rcalls_wfd=3,rows_wfd=3,width_wfd={"1":3},ev_wfd={"depth":{"1":3}},
                wall_ar=wall,wall_wfd=wall)
    return dict(meta=dict(args=arg,wf_commit="pinned"),per_sample=[sample],
                shadow_probe=dict(mode=mode,calls=0))


class CompareTests(unittest.TestCase):
    def test_match(self):
        r=compare({m:probe(m,t) for m,t in [("off",1),("gpu",1.1),("cpu",1.2)]})
        self.assertTrue(r["matched_geometry"])
        self.assertEqual(r["n_prompts"],1)
        self.assertAlmostEqual(r["arms"]["ar"]["cpu_minus_off_ms"]["mean"],200,places=4)

    def test_reject_geometry(self):
        cases={m:probe(m,1) for m in ("off","gpu","cpu")}
        cases["gpu"]["per_sample"][0]["rcalls_wfd"]=4
        with self.assertRaisesRegex(ValueError,"NOT MATCHED"):
            compare(cases)

    def test_reject_config(self):
        cases={m:probe(m,1) for m in ("off","gpu","cpu")}
        cases["cpu"]["meta"]["args"]["kv_s"]=1
        with self.assertRaisesRegex(ValueError,"configuration mismatch"):
            compare(cases)


if __name__=="__main__":
    unittest.main()
