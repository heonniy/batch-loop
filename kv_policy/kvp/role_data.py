"""Audited Content -> Instruction prompt construction and exact token-role spans.

Do not use the old few-shot prompts for this mechanism study: their question,
examples and instruction tokens overlap, and the earlier 96-token truncation
can remove the instruction entirely. This study uses a NEW frozen prompt
protocol and must not be compared numerically to the old benchmark tables.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

ROLES = ("content", "instruction", "distractor", "other")


@dataclass(frozen=True)
class RolePrompt:
    text: str
    token_ids: tuple[int, ...]
    roles: tuple[str, ...]
    content_positions: tuple[int, ...]
    instruction_positions: tuple[int, ...]
    distractor_positions: tuple[int, ...]

    def __post_init__(self):
        if len(self.token_ids) != len(self.roles):
            raise ValueError("Token-role length mismatch")
        if not self.content_positions or not self.instruction_positions:
            raise ValueError("Missing Content or Instruction span")
        if max(self.content_positions) >= min(self.instruction_positions):
            raise ValueError("Content must causally precede Instruction")
        for positions, role in ((self.content_positions, "content"),
                                (self.instruction_positions, "instruction"),
                                (self.distractor_positions, "distractor")):
            if any(self.roles[p] != role for p in positions):
                raise ValueError("Role positions inconsistent with token tags")


def _span(text: str, string: str):
    if text.count(string) != 1:
        raise ValueError("Role-delimited text must occur exactly once")
    a = text.index(string)
    return a, a + len(string)


def _label_tokens(tokenizer, text: str, spans: dict[str, tuple[int, int]]):
    """Prefer true tokenizer offsets, fail closed if alignment is ambiguous.

    Token pieces that cross a role boundary are marked 'other' and therefore
    never used as a candidate. No silent char-to-token proportional mapping.
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    try:
        result = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = result["offset_mapping"]
        check_ids = result["input_ids"]
        if len(check_ids) != len(token_ids) or list(check_ids) != list(token_ids):
            raise ValueError("Offset tokenizer ID mismatch")
        if len(offsets) != len(token_ids):
            raise ValueError("Offset mapping length mismatch")
        labels = []
        for a,b in offsets:
            if b <= a:
                labels.append("other")
                continue
            qualifying = [name for name,(x,y) in spans.items() if x <= a and b <= y]
            labels.append(qualifying[0] if len(qualifying)==1 else "other")
        return tuple(token_ids), tuple(labels)
    except (TypeError, KeyError, NotImplementedError, ValueError) as exc:
        # Some slow tokenizers have no offsets. Only accept prefix-complete
        # BPE boundaries; otherwise STOP instead of guessing token ownership.
        roles = ["other"] * len(token_ids)
        for name,(a,b) in spans.items():
            left = tokenizer.encode(text[:a], add_special_tokens=False)
            right = tokenizer.encode(text[:b], add_special_tokens=False)
            if token_ids[:len(left)] != left or token_ids[:len(right)] != right:
                raise RuntimeError(f"No trustworthy token alignment for {name}: {exc}") from exc
            if len(right) <= len(left):
                raise RuntimeError(f"Empty aligned span {name}") from exc
            for p in range(len(left),len(right)):
                if roles[p] != "other":
                    raise RuntimeError("Overlapping role spans")
                roles[p] = name
        return tuple(token_ids), tuple(roles)


def build_role_prompt(tokenizer, family: str, question: str, *,
                      dataset: str, use_distractor: bool = True) -> RolePrompt:
    if family not in {"huginn", "ouro"} or dataset not in {"gsm8k", "math500"}:
        raise ValueError("Unsupported model family or dataset")
    if not question.strip():
        raise ValueError("Empty question")
    distractor = ("A museum displayed paintings of a quiet courtyard and "
                  "a winter landscape. The gallery closed at dusk.")
    instruction = ("Solve the mathematical question given in the Content section. "
                   "Show the reasoning and state the final answer clearly.")
    prefix = "Unrelated Background:\n"
    content_hdr = "\nContent:\n"
    instruction_hdr = "\nInstruction:\n"
    suffix = "\nAnswer:\n"
    if use_distractor:
        body = prefix + distractor + content_hdr + question + instruction_hdr + instruction + suffix
    else:
        body = "Content:\n" + question + instruction_hdr + instruction + suffix
    if family == "huginn":
        text = tokenizer.apply_chat_template(
            [{"role":"system","content":"You are a helpful assistant for mathematical reasoning."},
             {"role":"user","content":body}],
            tokenize=False, add_generation_prompt=True)
    else:
        text = body
    # Match exact nonempty strings. Tokenizer offset spans are used to
    # disambiguate chat delimiters, special tokens and adjacent BPE merges.
    parts = {"content": question, "instruction": instruction}
    if use_distractor:
        parts["distractor"] = distractor
    spans = {name:_span(text,frag) for name,frag in parts.items()}
    ids, roles = _label_tokens(tokenizer, text, spans)
    if any(roles.count(k)==0 for k in ("content","instruction")):
        raise ValueError("Tokenizer did not expose nonempty role token spans")
    indices = {name:tuple(i for i,v in enumerate(roles) if v==name) for name in ROLES}
    return RolePrompt(text,ids,roles,indices["content"],
                      indices["instruction"],indices["distractor"])


def choose_role_sources(prompt: RolePrompt, count: int, seed: int,
                        distractor_count: int = 0):
    """Deterministic spread across Content, optional unrelated controls.

    This is random content coverage, NOT attention/answer-leaked selection.
    Each intervention is single-source; negative controls are kept separate.
    """
    import random
    if count < 1 or distractor_count < 0:
        raise ValueError("Invalid candidate count")
    rng = random.Random(seed)
    sources = list(prompt.content_positions)
    if len(sources) < count:
        raise ValueError("Insufficient aligned Content candidates")
    chosen = sorted(rng.sample(sources,count))
    if distractor_count:
        distractors = list(prompt.distractor_positions)
        if len(distractors)<distractor_count:
            raise ValueError("Insufficient distractor tokens")
        chosen.extend(sorted(rng.sample(distractors,distractor_count)))
    return chosen
