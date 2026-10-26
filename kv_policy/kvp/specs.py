"""Pinned model / dataset identities and evaluation protocol constants."""

from __future__ import annotations

from dataclasses import dataclass

SELECTION_SEED = 20261022
SUBSET_N = 200

GSM8K_REPO = "openai/gsm8k"
MATH500_REPO = "HuggingFaceH4/MATH-500"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo: str
    revision: str
    family: str  # "ouro" | "huginn"
    native_depth: int
    depth_note: str


MODELS = {
    "ouro-1.4b": ModelSpec(
        "ouro-1.4b", "ByteDance/Ouro-1.4B", "574fa66cb8bf5abdc979642d01cf2b79b16bfab1",
        "ouro", 4, "config.total_ut_steps=4 (training depth); early exit disabled, step-4 logits",
    ),
    "ouro-2.6b": ModelSpec(
        "ouro-2.6b", "ByteDance/Ouro-2.6B", "1ed04250da1a9936042725d302e81c8fa2ab5abd",
        "ouro", 4, "config.total_ut_steps=4 (training depth); early exit disabled, step-4 logits",
    ),
    "huginn-0125": ModelSpec(
        "huginn-0125", "tomg-group-umd/huginn-0125", "bb6621b65e90b6a4b9b29ef88dc83866d450470c",
        "huginn", 32,
         "num_steps=32 matches the documented mean recurrence; the checkpoint "
        "accepts other depths (config.mean_recurrence=32; the authors' GSM8K lm-eval command uses 32)",
    ),
}

# Separately labelled Huginn depth sweep (HUGINN_OURO_FULL_LAST_CONVERT_PLAN.md, E2): same checkpoint and
# protocol, fixed num_steps R in {4, 8, 16}; not the native R=32 evaluation.
for _r in (4, 8, 16):
    MODELS[f"huginn-0125-r{_r}"] = ModelSpec(
        f"huginn-0125-r{_r}", MODELS["huginn-0125"].repo, MODELS["huginn-0125"].revision, "huginn", _r,
        f"E2 depth sweep: num_steps={_r} (non-native; checkpoint mean_recurrence=32)")

POLICIES = ("full", "last", "average")
# "decode": paper-faithful (Ouro sec. 5.4.2) -- prompt KV keeps every loop, policy applies to
#           generated / teacher-forced target positions only.
# "strict": policy applies to every historical position including prompt tokens; prompt is
#           processed token-by-token (same semantics as vllm-rlt `shared`).
MODES = ("decode", "strict")

# Generation limits. GSM8K: lm-eval HFLM default max_gen_toks=256 (gsm8k_cot sets none).
# MATH-500: frozen project choice (no verifiable official value), applied to every policy.
MAX_NEW_TOKENS = {"gsm8k": 512, "math500": 2048}

HUGINN_SYSTEM = "You are a helpful assistant that can assist users with mathematical reasoning."
