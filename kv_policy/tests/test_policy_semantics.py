"""Policy read/write semantics on tiny random-weight Ouro/Huginn models (CPU, float64).

kvp.engine (broadcast layout + finalize) is compared against kvp.reference (native HF forward
with a dictionary cache that materialises each policy's history from its definition).
"""

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from kvp.engine import HuginnBackbone, OuroBackbone, Runner
from kvp.reference import reference_logits
from kvp.specs import MODELS

torch.manual_seed(0)


def _tiny(family: str, R: int):
    spec = MODELS["ouro-1.4b" if family == "ouro" else "huginn-0125"]
    cfg = AutoConfig.from_pretrained(spec.repo, revision=spec.revision, trust_remote_code=True)
    if family == "ouro":
        cfg.hidden_size, cfg.intermediate_size, cfg.head_dim = 64, 128, 16
        cfg.num_attention_heads = cfg.num_key_value_heads = 4
        cfg.num_hidden_layers, cfg.total_ut_steps, cfg.max_window_layers = 2, R, 2
        cfg.layer_types = ["full_attention"] * 2
        cfg.vocab_size = 512
    else:
        cfg = type(cfg)(n_embd=64, n_heads=4, n_layers=4, block_size=256, vocab_size=512,
                        intermediate_size=128, n_layers_in_prelude=1, n_layers_in_recurrent_block=2,
                        n_layers_in_coda=1, mean_recurrence=R, qk_bias=True, tie_embeddings=True,
                        pad_token_id=1, bos_token_id=2, eos_token_id=3)
    if family == "ouro":
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, torch_dtype=torch.float64)
    else:
        cls = get_class_from_dynamic_module("raven_modeling_minimal.RavenForCausalLM", spec.repo,
                                            revision=spec.revision)
        model = cls(cfg)
    with torch.no_grad():  # random but non-degenerate weights (incl. norms / qk bias)
        for name, p in model.named_parameters():
            if p.dim() == 1:
                p.copy_(1 + 0.2 * torch.randn_like(p))
            elif "qk_bias" in name:
                p.copy_(0.1 * torch.randn_like(p))
            else:
                p.copy_(torch.randn_like(p) / p.shape[-1] ** 0.5)
    model = model.double().eval()
    if family == "ouro":
        bb = OuroBackbone(model)
        pad = 0
    else:
        bb = HuginnBackbone(model, R, noise_seed=7)
        pad = 1
    return model, Runner(bb, pad)


PROMPT = [5, 17, 99, 3, 250, 42, 7]
CONT = [11, 300, 8, 64, 2]


def _noise(run, n):
    return run.bb.example_noise("u", n) if run.bb.family == "huginn" else None


@pytest.fixture(scope="module", params=["ouro", "huginn"])
def tiny(request):
    return request.param, *_tiny(request.param, 3)


@pytest.mark.parametrize("policy", ["full", "last", "first", "average", "blend25", "blend50", "blend75"])
@pytest.mark.parametrize("mode", ["decode", "convert", "strict"])
def test_engine_matches_reference(tiny, policy, mode):
    family, model, run = tiny
    ref = reference_logits(model, family, run.bb.R, PROMPT, CONT, policy, mode, _noise(run, 64))
    eng = run.teacher_forced_logits(PROMPT, CONT, "u", policy, mode)
    assert torch.allclose(eng, ref, atol=1e-9, rtol=1e-9), (eng - ref).abs().max()


def test_policies_actually_differ(tiny):
    family, model, run = tiny
    full = run.teacher_forced_logits(PROMPT, CONT, "u", "full", "decode")
    last = run.teacher_forced_logits(PROMPT, CONT, "u", "last", "decode")
    avg = run.teacher_forced_logits(PROMPT, CONT, "u", "average", "decode")
    P = len(PROMPT)
    # prompt logits and the first continuation prediction see only full-region history
    assert torch.allclose(full[:P], last[:P]) and torch.allclose(full[:P], avg[:P])
    # position P (first fed continuation token) reads only the prompt -> identical too
    assert torch.allclose(full[P], last[P])
    # from P+1 on, history contains policy-region tokens -> must differ
    assert (full[P + 1:] - last[P + 1:]).abs().max() > 1e-6
    assert (last[P + 1:] - avg[P + 1:]).abs().max() > 1e-6


def test_single_loop_all_policies_equal():
    for family in ("ouro", "huginn"):
        model, run = _tiny(family, 1)
        outs = [run.teacher_forced_logits(PROMPT, CONT, "u", p, "strict") for p in ("full", "last", "first", "average", "blend25", "blend50", "blend75")]
        assert all(torch.allclose(outs[0], out, atol=1e-12) for out in outs[1:])


def test_strict_full_equals_decode_full(tiny):
    family, model, run = tiny
    a = run.teacher_forced_logits(PROMPT, CONT, "u", "full", "decode")
    b = run.teacher_forced_logits(PROMPT, CONT, "u", "full", "strict")
    assert torch.allclose(a, b, atol=1e-9)


@pytest.mark.parametrize("policy", ["full", "last", "first", "average", "blend25", "blend50", "blend75"])
def test_no_future_leakage(tiny, policy):
    family, model, run = tiny
    a = run.teacher_forced_logits(PROMPT, CONT, "u", policy, "strict")
    b = run.teacher_forced_logits(PROMPT, CONT[:3] + [123, 456], "u", policy, "strict")
    n = len(PROMPT) + 3  # logits up to and including the last shared token
    assert torch.allclose(a[:n], b[:n], atol=1e-12)


@pytest.mark.parametrize("policy", ["full", "last", "first", "average", "blend25", "blend50", "blend75"])
@pytest.mark.parametrize("mode", ["decode", "convert", "strict"])
def test_batched_left_padding_matches_b1(tiny, policy, mode):
    """Positions/masks: a short prompt left-padded in a batch must equal its B=1 run."""
    family, model, run = tiny
    short = PROMPT[:4]
    gold = [run.teacher_forced_logits(p, CONT, u, policy, mode) for p, u in ((PROMPT, "u"), (short, "v"))]
    res = run.score([PROMPT, short], [CONT, CONT], ["u", "v"], policy, mode)
    lp = [torch.log_softmax(g, -1) for g in gold]
    for b, (p, g) in enumerate(zip((PROMPT, short), lp)):
        ref = [g[len(p) - 1 + i, t].item() for i, t in enumerate(CONT)]
        assert max(abs(x - y) for x, y in zip(ref, res.target_logprobs[b])) < 1e-9


def test_average_buffer_is_mean_of_loops(tiny):
    """After finalisation the stored history equals the fp32 mean of the per-loop K and V."""
    family, model, run = tiny
    ids, pos, valid, kv, aux, P, T = run._setup([PROMPT], 2, "average", "strict", ["u"])
    captured = {}
    orig = kv.finalize

    def spy(c):
        captured[c] = [(k[:, :, :, c].clone(), v[:, :, :, c].clone()) for k, v in zip(kv.rk, kv.rv)]
        orig(c)

    kv.finalize = spy
    with torch.inference_mode():
        run._prefill(ids, pos, valid, kv, aux, P, "strict")
    for c, per_layer in captured.items():
        for (k, v), kb, vb in zip(per_layer, kv.rk, kv.rv):
            acc = torch.promote_types(k.dtype, torch.float32)
            mk = k.to(acc).mean(0).to(k.dtype)
            mv = v.to(acc).mean(0).to(v.dtype)
            for r in range(run.bb.R):
                assert torch.equal(kb[r, :, :, c], mk) and torch.equal(vb[r, :, :, c], mv)
            # pre-finalisation planes were the distinct per-loop versions (written in loop order)
            assert not torch.equal(k[0], k[-1])


def test_last_history_written_only_after_final_loop(tiny):
    family, model, run = tiny
    ids, pos, valid, kv, aux, P, T = run._setup([PROMPT], 2, "last", "strict", ["u"])
    seen = []
    orig_attend = kv.attend

    def spy(layer, q, k, v, c0, c1, mask):
        if layer[0] == "rec" and c0 >= 1:
            kb = kv.rk[layer[1]]
            # every plane of the previous column already holds the final-loop version
            prev = kb[:, :, :, c0 - 1]
            seen.append(all(torch.equal(prev[r], prev[run.bb.R - 1]) for r in range(run.bb.R)))
        return orig_attend(layer, q, k, v, c0, c1, mask)

    kv.attend = spy
    with torch.inference_mode():
        run._prefill(ids, pos, valid, kv, aux, P, "strict")
    assert seen and all(seen)


def test_convert_reads_policy_versions_of_prompt(tiny):
    """convert: same prefill as decode/full, but later tokens see the prompt's policy version."""
    family, model, run = tiny
    P = len(PROMPT)
    full = run.teacher_forced_logits(PROMPT, CONT, "u", "full", "decode")
    for pol in ("last", "average"):
        conv = run.teacher_forced_logits(PROMPT, CONT, "u", pol, "convert")
        dec = run.teacher_forced_logits(PROMPT, CONT, "u", pol, "decode")
        assert torch.allclose(conv[:P], full[:P])          # prompt logits: per-loop prefill
        assert (conv[P:] - dec[P:]).abs().max() > 1e-6       # first decode step already differs


def test_convert_buffer_holds_single_version(tiny):
    family, model, run = tiny
    ids, pos, valid, kv, aux, P, T = run._setup([PROMPT], 2, "last", "convert", ["u"])
    with torch.inference_mode():
        run._prefill(ids, pos, valid, kv, aux, P, "convert")
    for kb in kv.rk:
        for r in range(run.bb.R):
            assert torch.equal(kb[r, :, :, :P], kb[run.bb.R - 1, :, :, :P])


def test_first_strict_writes_first_loop_kv(tiny):
    """Every completed prompt token retains loop-0 K/V across all recurrence read planes."""
    family, model, run = tiny
    ids, pos, valid, kv, aux, P, T = run._setup([PROMPT], 2, "first", "strict", ["u"])
    with torch.inference_mode():
        run._prefill(ids, pos, valid, kv, aux, P, "strict")
    for kb, vb in zip(kv.rk, kv.rv):
        for r in range(1, run.bb.R):
            assert torch.equal(kb[r, :, :, :P], kb[0, :, :, :P])
            assert torch.equal(vb[r, :, :, :P], vb[0, :, :, :P])



@pytest.mark.parametrize("policy", ["lastwin1024", "lastpromptwin1024"])
def test_unlimited_window_is_exact_last(tiny, policy):
    """A window covering the entire sequence must give LAST/strict logits."""
    family, model, runner = tiny
    ref = runner.teacher_forced_logits(PROMPT, CONT, "u", "last", "strict")
    obs = runner.teacher_forced_logits(PROMPT, CONT, "u", policy, "strict")
    assert torch.equal(ref, obs)


def test_window_mask_is_monotonic_and_left_padding_safe(tiny):
    """Evicted historical columns cannot re-enter as decode proceeds."""
    family, model, runner = tiny
    ids, pos, valid, kv, aux, P, T = runner._setup(
        [PROMPT, PROMPT[:4]], 5, "lastwin2", "strict", ["u", "v"])
    # First four valid positions are pinned attention sinks. For the short
    # second row, left padding means positions [P-4, ..., P-1].
    c = P + 3
    keep = kv.historical_keep_mask(c)
    assert bool(keep[1, P-4]) and bool(keep[1, P-1])
    assert bool(keep[0, 0]) and bool(keep[0, 3])
    assert bool(keep[0, c])  # current token always legal
    assert not bool(keep[0, 4])  # old, nonsink position should be evicted
    assert bool(keep[0, c-1])
    assert bool(keep[0, c])
    assert not bool(kv.historical_keep_mask(c+1)[0, 4])


def test_prompt_protection_changes_survival(tiny):
    family, model, runner = tiny
    w = runner._setup([PROMPT], 10, "lastwin2", "strict", ["u"])[3]
    p = runner._setup([PROMPT], 10, "lastpromptwin2", "strict", ["u"])[3]
    c = len(PROMPT) + 5
    assert not bool(w.historical_keep_mask(c)[0, 4])
    assert bool(p.historical_keep_mask(c)[0, 4])


def test_window_evicts_only_history_not_current(tiny):
    family, model, runner = tiny
    a = runner.teacher_forced_logits(PROMPT, CONT, "u", "last", "strict")
    b = runner.teacher_forced_logits(PROMPT, CONT, "u", "lastwin2", "strict")
    assert (a - b).abs().max() > 1e-6
    # Re-run on the same input: policy selection must be deterministic.
    assert torch.equal(b, runner.teacher_forced_logits(PROMPT, CONT, "u", "lastwin2", "strict"))


def test_protected_window_large_matches_protected_strict(tiny):
    from kvp.banks import REGISTRY
    family, model, runner = tiny
    key = (getattr(runner, "model_key", None), "bankP0")
    # This local fixture has a two-layer / three-recurrence block.
    # No global calibration registry or model checkpoint is required.
    try:
        prior = REGISTRY.get(key)
        REGISTRY[key] = frozenset({(0, 0)})
        full_window = runner.teacher_forced_logits(PROMPT, CONT, "u", "bankP0win1024", "strict")
        from kvp.banks import BankKV
        original = runner.teacher_forced_logits(PROMPT, CONT, "u", "bankP0", "strict")
        assert torch.equal(full_window, original)
    finally:
        if prior is None:
            REGISTRY.pop(key, None)
        else:
            REGISTRY[key] = prior
