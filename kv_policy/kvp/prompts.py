"""Prompt rendering through lm-eval's own fewshot/chat-template code path.

Per-model protocol (frozen before any policy comparison; identical for FULL/LAST/AVERAGE):

* Ouro base, GSM8K: Ouro paper Table 16 says "strict match, 3-shot CoT, lm-eval-harness".
  Rendered as lm-eval `gsm8k_cot` with num_fewshot=3 (first_n exemplars), no chat template
  (the paper does not state one; base-model default). Verifiable source = paper table.
* Ouro base, MATH-500: paper says "strict match, 5-shot CoT, In-house" -- the in-house prompt is
  not published, so NOT reproducible. Frozen substitute: lm-eval minerva_math 4-shot prompt.
* Huginn, GSM8K: the authors' README lm-eval command (gsm8k_cot, 8-shot, apply_chat_template,
  fewshot_as_multiturn, system instruction) except mean_recurrence=32 -> 16 (scope document).
* Huginn, MATH-500: no published protocol. Frozen substitute: minerva 4-shot prompt with the same
  chat template / multiturn / system instruction as the authors' GSM8K command.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from .specs import HUGINN_SYSTEM, MAX_NEW_TOKENS


@dataclass(frozen=True)
class PromptProtocol:
    model_family: str
    dataset: str
    num_fewshot: int
    apply_chat_template: bool
    fewshot_as_multiturn: bool
    system_instruction: str | None
    until: tuple[str, ...]
    max_new_tokens: int
    provenance: str


def protocol(model_family: str, dataset: str) -> PromptProtocol:
    if dataset == "gsm8k":
        until = ("Q:", "</s>", "<|im_end|>")
        if model_family == "ouro":
            return PromptProtocol("ouro", dataset, 3, False, False, None, until, MAX_NEW_TOKENS[dataset],
                                  "Ouro paper Table 16 (lm-eval gsm8k_cot, 3-shot CoT, strict match); "
                                  "chat template not specified by paper -> none")
        return PromptProtocol("huginn", dataset, 8, True, True, HUGINN_SYSTEM, until, MAX_NEW_TOKENS[dataset],
                              "seal-rg/recurrent-pretraining README lm-eval command "
                              "(gsm8k_cot, chat template, fewshot_as_multiturn, system instruction); "
                              "mean_recurrence 32 replaced by 16")
    if dataset == "math500":
        until = ("Problem:",)
        if model_family == "ouro":
            return PromptProtocol("ouro", dataset, 4, False, False, None, until, MAX_NEW_TOKENS[dataset],
                                  "NOT official: paper uses an unpublished in-house 5-shot prompt; "
                                  "frozen substitute = lm-eval minerva_math 4-shot")
        return PromptProtocol("huginn", dataset, 4, True, True, HUGINN_SYSTEM, until, MAX_NEW_TOKENS[dataset],
                              "NOT official: no published Huginn MATH protocol; frozen substitute = "
                              "minerva 4-shot + authors' GSM8K chat settings")
    raise ValueError(dataset)


@lru_cache(maxsize=None)
def _task(dataset: str):
    from lm_eval.tasks import TaskManager, get_task_dict

    if dataset == "gsm8k":
        return get_task_dict(["gsm8k_cot"], TaskManager())["gsm8k_cot"]
    from lm_eval.api.task import ConfigurableTask
    from lm_eval.tasks.minerva_math import utils as mu

    config = {
        "task": "kvp_math500_minerva",
        "dataset_path": "HuggingFaceH4/MATH-500",
        "dataset_kwargs": {"revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"},
        "test_split": "test",
        "output_type": "generate_until",
        "doc_to_text": mu.doc_to_text,
        "doc_to_target": "{{answer if few_shot is undefined else solution}}",
        "generation_kwargs": {"until": ["Problem:"], "do_sample": False},
        "num_fewshot": 4,
        "fewshot_config": {"sampler": "first_n", "samples": mu.list_fewshot_samples},
        "metric_list": [{"metric": "exact_match"}],
    }
    return ConfigurableTask(config=config)


def make_chat_fn(tokenizer):
    """Replicates lm_eval.models.huggingface.HFLM.apply_chat_template (v0.4.9.1)."""

    def apply(chat_history, add_generation_prompt=True):
        return tokenizer.apply_chat_template(
            chat_history, tokenize=False, add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )

    return apply


def render(doc: dict, proto: PromptProtocol, tokenizer) -> str:
    task = _task(proto.dataset)
    lm_doc = {"question": doc["question"], "answer": doc["answer"]} if proto.dataset == "gsm8k" else {
        "problem": doc["problem"], "solution": doc["solution"], "answer": doc["answer"]}
    return task.fewshot_context(
        lm_doc, proto.num_fewshot, system_instruction=proto.system_instruction,
        apply_chat_template=proto.apply_chat_template, fewshot_as_multiturn=proto.fewshot_as_multiturn,
        chat_template=make_chat_fn(tokenizer) if proto.apply_chat_template else None,
    )


def encode_prompt(text: str, tokenizer) -> list[int]:
    # lm-eval HFLM.tok_encode for causal models: add_special_tokens=False (add_bos_token default False).
    return tokenizer.encode(text, add_special_tokens=False)


_CALC = re.compile(r"<<[^>]*>>")


def gold_target(doc: dict, proto: PromptProtocol) -> str:
    """Teacher-forced NLL target, written in the same answer format the prompt elicits.

    GSM8K: reference rationale with calculator annotations <<...>> removed and the trailing
    "#### N" line replaced by "The answer is N." (gsm8k_cot exemplar format).
    MATH-500: reference solution followed by the minerva closing line used in the exemplars.
    Prepended with lm-eval's target_delimiter " " when no chat template is used (completion
    format "...A: <target>"); with a chat template the target starts the assistant turn.
    """
    if proto.dataset == "gsm8k":
        rationale, final = doc["answer"].split("####")
        lines = [_CALC.sub("", ln).strip() for ln in rationale.strip().splitlines() if ln.strip()]
        text = " ".join(lines) + f" The answer is {final.strip()}."
    else:
        text = doc["solution"].strip() + (
            f"\nFinal Answer: The final answer is ${doc['answer']}$. I hope it is correct.")
    return text if proto.apply_chat_template else " " + text
