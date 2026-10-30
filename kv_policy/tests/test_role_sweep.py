"""CPU-only frozen input/manifest and role-label provenance checks.

No pretrained weights, datasets, GPU or network are needed. Only torch and
the project modules are imported. Model parity remains a separate gate.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


class CharTokenizer:
    def encode(self,text,add_special_tokens=False):
        return [ord(c) for c in text]
    def __call__(self,text,add_special_tokens=False,return_offsets_mapping=True):
        return {"input_ids":self.encode(text),
                "offset_mapping":[(i,i+1) for i in range(len(text))]}
    def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=True):
        return ("<system>"+messages[0]["content"]+"</system><user>"+
                messages[1]["content"]+"</user><assistant>")


def load_runner():
    path=Path(__file__).resolve().parents[1]/"scripts/role_sensitivity_sweep.py"
    spec=importlib.util.spec_from_file_location("role_runner_test",path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_prepare_freezes_role_positions_and_question_count(tmp_path,monkeypatch):
    r=load_runner()
    tokenizer=CharTokenizer()
    fake_auto=types.SimpleNamespace(
        from_pretrained=lambda *args,**kwargs:tokenizer)
    monkeypatch.setitem(sys.modules,"transformers",
                        types.SimpleNamespace(AutoTokenizer=fake_auto))
    docs={
      "gsm8k":[{"uid":"gsm-a","question":"How many of 12 apples remain after 5?",
                "answer":"7 #### 7"},
              {"uid":"gsm-b","question":"If 3 added to 5, how many?",
               "answer":"8 #### 8"}],
      "math500":[{"uid":"math-a","problem":"Evaluate 2+2.","answer":"4",
                  "solution":"2+2=4."},
                {"uid":"math-b","problem":"Compute 3+3.","answer":"6",
                  "solution":"3+3=6."}]}
    def fake_subset(ds):
        return docs[ds],{"repo":"fixture","revision":"r1",
                         "sha256":"dataset-hash","selection_sha256":"selected"}
    monkeypatch.setitem(sys.modules,"kvp.data",
                        types.SimpleNamespace(load_subset=fake_subset))
    monkeypatch.setitem(sys.modules,"kvp.prompts",
                        types.SimpleNamespace(gold_target=lambda doc,proto:"7 8 9",
                                              protocol=lambda family,ds:object()))
    monkeypatch.setattr(r,"git_sha",lambda:"fixture-code")
    cfg={"models":["huginn-0125","ouro-1.4b","ouro-2.6b"],
         "datasets":["gsm8k","math500"],"decode_modes":["fixed","dynamic"],
         "n_examples":2,"n_content":2,"n_instruction":1,
         "n_distractor":1,"distractor":True,
         "max_prompt_tokens":1536,"max_target_tokens":64,
         "seed":20261030,"ouro_q":.7,"huginn_cosine":.001,
         "huginn_max_depth":32,"huginn_cosine_dtype":"float32",
         "min_depth":2,"one_loop":True,"live_exit":True}
    run=tmp_path/"run"
    result=r.prepare(run,cfg)
    assert result["jobs"]==24
    manifest,jobs=r.manifest(run)
    assert manifest["dataset_sources"]["gsm8k"]["revision"]=="r1"
    assert len({(j["model"],j["dataset"],j["uid"],j["decode_mode"]) for j in jobs})==24
    for j in jobs:
        assert max(j["content_positions"])<min(j["instruction_positions"])
        assert len(j["prompt_ids"])==len(j["prompt_roles"])
        assert all(j["prompt_roles"][p]=="content" for p in j["content_candidates"])
        assert len(j["role_control_candidates"])==2
        assert all(j["prompt_roles"][p] in ("instruction","distractor")
                   for p in j["role_control_candidates"])
        assert j["target_truncated"] is False


def test_request_level_holdout_is_not_per_token_leakage():
    r=load_runner()
    rows=[]
    for u in range(30):
        for i in range(2):
            rows.append({
                "uid":f"question-{u}",
                "age_at_prefill_end":i+3,
                "attention_mean":i+.01*u,
                "attention_max":i+.02*u,
                "source_mean_recurrent_change":.1+.001*u,
                "prefill_persistent_final_error":.2+.002*u,
                "postprefill_eviction_delta_future_nll":.03*u+.01*i})
    output=r.heldout_importance_test(rows)
    assert output["train_requests"]==21
    assert output["test_requests"]==9
    assert output["train_tokens"]==42
    assert output["test_tokens"]==18
    assert {"attention_age","attention_age_plus_prefill_state",
            "ORACLE_attention_plus_masked_instruction"}==set(output["heldout_r2"])
