"""CPU-only exact role-span and intervention sanity tests."""
import pytest
import torch

from kvp.role_data import RolePrompt, build_role_prompt, choose_role_sources
from kvp.role_sensitivity import RoleProbeKV, _trajectory_stats


class CharTokenizer:
    def encode(self, s, add_special_tokens=False):
        return [ord(c) for c in s]

    def __call__(self, s, add_special_tokens=False, return_offsets_mapping=False):
        assert return_offsets_mapping
        return {"input_ids":self.encode(s),
                "offset_mapping":[(i,i+1) for i in range(len(s))]}

    def apply_chat_template(self, history, tokenize=False,
                            add_generation_prompt=True):
        return ("<system>"+history[0]["content"]+"</system>" +
                "<user>"+history[1]["content"]+"</user><assistant>")


def test_role_attribution_is_causal_and_exact_for_both_families():
    for family in ("ouro","huginn"):
        p=build_role_prompt(CharTokenizer(),family,
                            "Alice has 12 apples.",dataset="gsm8k")
        assert max(p.content_positions)<min(p.instruction_positions)
        assert len(p.token_ids)==len(p.roles)
        assert set(p.roles)>={"content","instruction","distractor","other"}
        assert all(p.roles[i]=="content" for i in p.content_positions)
        assert all(p.roles[i]=="instruction" for i in p.instruction_positions)
        assert any(i>max(p.instruction_positions) for i in range(len(p.roles)))
        assert choose_role_sources(p,2,7)==choose_role_sources(p,2,7)
        assert len(set(choose_role_sources(p,2,7,1)))==3


def test_role_mapping_rejects_wrong_causal_order():
    with pytest.raises(ValueError,match="causally"):
        RolePrompt("bad",(1,2,3),("instruction","other","content"),
                   (2,),(0,),())


def make_kv(mode="none",source=None,selected_loop=1,observe=False):
    class Tiny:
        R=3
        n_rec=1
        n_plain=0
        H=D=1
    p=RolePrompt(
        text="source",
        token_ids=tuple(range(7)),
        roles=("distractor","content","content","other",
               "instruction","instruction","other"),
        content_positions=(1,2),
        instruction_positions=(4,5),
        distractor_positions=(0,))
    return RoleProbeKV(Tiny(),9,"cpu",torch.float32,
                       prompt=p,mode=mode,source=source,
                       perturbed_loop=selected_loop,observe=observe,
                       tracked_positions=(5,))


def prime(kv):
    q=torch.zeros(1,1,1,1)
    for c in range(4):
        for r in range(3):
            v=torch.full_like(q,100 if c==1 else 0)
            kv.attend(("rec",0,r),q,q,v,c,c+1,
                      torch.ones(1,1,1,c+1,dtype=torch.bool))
        kv.finish(c,3)


def test_instruction_masks_apply_only_to_instruction_query_and_scope():
    q=torch.zeros(1,1,1,1)
    mask=torch.ones(1,1,1,6,dtype=torch.bool)
    vals=[]
    for mode in ("none","instruction_all","instruction_one_loop",
                 "postprefill"):
        kv=make_kv(mode,1)
        prime(kv)
        outs=[]
        for r in (0,1):
            outs.append(float(kv.attend(
                ("rec",0,r),q,q,q,5,6,mask).item()))
        vals.append(outs)
    original,persistent,once,post=vals
    assert original[0]>0
    assert persistent==pytest.approx([0,0])
    assert once==pytest.approx([0,original[1]])
    assert post==pytest.approx(original)
    # The single-loop arm MUST NOT inject into earlier Instruction queries.
    kv=make_kv("instruction_one_loop",1)
    prime(kv)
    assert not kv._should_mask(("rec",0,0),4)
    assert kv._should_mask(("rec",0,0),5)
    assert not kv._should_mask(("rec",0,1),5)


def test_attention_mass_is_not_misread_as_causal_effect():
    kv=make_kv(observe=True)
    prime(kv)
    q=torch.zeros(1,1,1,1)
    for r in range(3):
        kv.attend(("rec",0,r),q,q,q,4,5,
                  torch.ones(1,1,1,5,dtype=torch.bool))
    values=kv.observed_attention()
    assert len(values["loop_roles"])==3
    assert all(x["n_query_layer_observations"]==1
               for x in values["loop_roles"])
    assert all(x["content_mass"]==pytest.approx(.4)
               for x in values["loop_roles"])
    assert set(values["content_attention_by_source"])=={"1","2"}


def test_trajectory_is_per_loop_and_can_recover():
    b={1:torch.tensor([1.0,0.0]),2:torch.tensor([1.0,0.0])}
    x={1:torch.tensor([2.0,0.0]),2:torch.tensor([1.1,0.0])}
    d=_trajectory_stats(x,b,2)
    assert d["first_error"]==pytest.approx(1.0)
    assert d["final_error"]==pytest.approx(.1,abs=1e-6)
    assert d["final_over_first"]==pytest.approx(.1,abs=1e-6)


def test_postprefill_mask_source_must_belong_to_a_real_role():
    kv=make_kv("postprefill",4)
    prime(kv)
    assert kv.mode=="postprefill"
    with pytest.raises(ValueError,match="allowed"):
        make_kv("instruction_all",4)
