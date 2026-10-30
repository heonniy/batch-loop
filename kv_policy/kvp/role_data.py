"""Audited Content -> Instruction prompt construction and exact token-role spans.

Do not use the old few-shot prompts for this mechanism study: their question,
examples and instruction tokens overlap, and the earlier 96-token truncation
can remove the instruction entirely. This study uses a NEW frozen prompt
protocol and must not be compared numerically to the old benchmark tables.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

ROLES = ("content", "instruction", "distractor", "other", "relevant_content", "irrelevant_content", "question")


@dataclass(frozen=True)
class RolePrompt:
    text: str
    token_ids: tuple[int, ...]
    roles: tuple[str, ...]
    content_positions: tuple[int, ...]
    instruction_positions: tuple[int, ...]
    distractor_positions: tuple[int, ...]
    # v2: evidence/distractor are both part of the causal Content prefix.
    relevant_positions: tuple[int, ...] = ()
    irrelevant_positions: tuple[int, ...] = ()
    question_positions: tuple[int, ...] = ()
    context_spans: tuple[dict, ...] = ()

    def __post_init__(self):
        if len(self.token_ids) != len(self.roles):
            raise ValueError("Token-role length mismatch")
        if not self.content_positions or not self.instruction_positions:
            raise ValueError("Missing Content or Instruction span")
        if max(self.content_positions) >= min(self.instruction_positions):
            raise ValueError("Content must causally precede Instruction")
        allowed = ((self.content_positions, {"content", "relevant_content", "irrelevant_content"}),
                   (self.instruction_positions, {"instruction", "question"}),
                   (self.distractor_positions, {"distractor"}),
                   (self.relevant_positions, {"relevant_content"}),
                   (self.irrelevant_positions, {"irrelevant_content"}),
                   (self.question_positions, {"question"}))
        for positions, labels in allowed:
            if any(p < 0 or p >= len(self.roles) or self.roles[p] not in labels for p in positions):
                raise ValueError("Role positions inconsistent with token tags")
        for entry in self.context_spans:
            positions = entry["positions"]
            if not positions or any(p not in self.content_positions for p in positions):
                raise ValueError("Evidence span lacks fully aligned Context tokens")
            if any(self.roles[p] != entry["role"] for p in positions):
                raise ValueError("Evidence span role misattributed")


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


def build_evidence_prompt(tokenizer, family: str, doc: dict) -> RolePrompt:
    """Build an auditable Context -> Instruction -> Question prompt.

    doc["spans"] has explicit sentence-level evidence labels, not guessed
    from an answer or random selected token. A span's full byte/text offsets
    are resolved through tokenizer offsets. Cross-boundary tokens are Other.
    """
    if family not in ("huginn", "ouro"):
        raise ValueError("Unsupported family")
    if not doc.get("question") or not doc.get("answer") or not doc.get("spans"):
        raise ValueError("Missing question, answer or Context spans")
    text_parts = []
    byte_spans = {}
    source_labels = {}
    span_records = []

    def append(part, key=None):
        start = sum(len(x) for x in text_parts)
        text_parts.append(part)
        if key is not None:
            byte_spans[key] = (start, start + len(part))

    append("Context:\n")
    for i, span in enumerate(doc["spans"]):
        role = span["role"]
        if role not in ("relevant_content", "irrelevant_content"):
            raise ValueError(f"Unexpected Context role: {role}")
        part = span["text"].strip()
        if not part:
            raise ValueError("Empty Context sentence")
        key = f"context_{i}"
        append(part, key)
        source_labels[key] = role
        span_records.append({"id": str(span["id"]), "key": key,
                             "role": role, "source": span.get("source", ""),
                             "part": part})
        append("\n")
    append("\nInstruction:\n")
    append("Use the context to answer the question. Reply with 'Answer: <short answer>'.",
           "instruction")
    append("\nQuestion:\n")
    append(doc["question"].strip(), "question")
    append("\nResponse:\n")
    body = "".join(text_parts)

    if family == "huginn":
        text = tokenizer.apply_chat_template(
            [{"role": "system",
              "content": "Use the provided context to answer questions accurately."},
             {"role": "user", "content": body}],
            tokenize=False, add_generation_prompt=True)
        if text.count(body) != 1:
            raise ValueError("Chat template did not preserve exactly one user body")
        shift = text.index(body)
    else:
        text, shift = body, 0
    offsets = {key: (a + shift, b + shift)
               for key, (a, b) in byte_spans.items()}
    ids, labels = _label_tokens(tokenizer, text, offsets)
    roles = tuple(source_labels.get(tag, tag) for tag in labels)
    content = tuple(i for i, v in enumerate(roles)
                    if v in ("relevant_content", "irrelevant_content"))
    instruction = tuple(i for i, v in enumerate(roles)
                        if v in ("instruction", "question"))
    relevant = tuple(i for i, v in enumerate(roles) if v == "relevant_content")
    irrelevant = tuple(i for i, v in enumerate(roles) if v == "irrelevant_content")
    question = tuple(i for i, v in enumerate(roles) if v == "question")
    if not relevant or not irrelevant or not question or "instruction" not in roles:
        raise ValueError("Missing aligned evidence, distractor, question or instruction")

    # Offsets for an *optional* ground-truth answer anchor. This is used only
    # to construct controlled diagnostic windows, NOT a deployable predictor.
    try:
        tok_result = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        token_offsets = tok_result["offset_mapping"]
        if list(tok_result["input_ids"]) != list(ids):
            raise ValueError("Tokenizer offsets inconsistent")
    except (KeyError, ValueError, TypeError, NotImplementedError):
        token_offsets = None
    aligned_spans = []
    for source in span_records:
        a, b = offsets[source["key"]]
        positions = tuple(i for i, tag in enumerate(labels) if tag == source["key"])
        if len(positions) < 2:
            raise ValueError(f"Not enough aligned tokens for {source['id']}")
        anchor = None
        if source["role"] == "relevant_content":
            hit = source["part"].casefold().find(doc["answer"].strip().casefold())
            if hit >= 0 and token_offsets is not None:
                absolute = a + hit
                for p in positions:
                    x, y = token_offsets[p]
                    if x <= absolute < y:
                        anchor = p
                        break
        aligned_spans.append({"id": source["id"], "role": source["role"],
                              "positions": positions, "source": source["source"],
                              "answer_anchor_token": anchor})
    return RolePrompt(text, ids, roles, content, instruction, (),
                      relevant, irrelevant, question, tuple(aligned_spans))


def choose_evidence_candidates(prompt: RolePrompt, seed: int, *,
                               n_relevant: int = 2, n_irrelevant: int = 2,
                               window_tokens: int = 12) -> list[dict]:
    """Choose ground-truth labeled *matched-width* Context KV interventions.

    Each candidate is a consecutive window WITHIN a labeled sentence.
    The same number of KV tokens is hidden for every candidate in a request.
    For short answers, an answer occurrence, if present, anchors the relevant
    window. This is a characterization oracle and must never be called a
    cheap inference-time candidate selector.
    """
    import random
    if n_relevant < 1 or n_irrelevant < 1 or window_tokens < 1:
        raise ValueError("Require positive evidence, distractor and window counts")
    groups = {role: [s for s in prompt.context_spans if s["role"] == role]
              for role in ("relevant_content", "irrelevant_content")}
    if any(not group for group in groups.values()):
        raise ValueError("Evidence and irrelevant context must both exist")
    rng = random.Random(seed)
    selected = []
    for role, count in (("relevant_content", n_relevant),
                        ("irrelevant_content", n_irrelevant)):
        pool = sorted(groups[role], key=lambda s: s["id"])
        selected.extend(rng.sample(pool, min(count, len(pool))))
    width = min(window_tokens, *(len(s["positions"]) for s in selected))
    if width < 2:
        raise ValueError("Matched intervention must mask >=2 tokens")
    chosen = []
    for span in selected:
        positions = tuple(span["positions"])
        anchor = span.get("answer_anchor_token")
        if anchor is not None and anchor in positions:
            center = positions.index(anchor)
        else:
            center = len(positions) // 2
        start = max(0, min(len(positions) - width, center - width // 2))
        window = positions[start:start + width]
        chosen.append({"id": span["id"], "role": span["role"],
                       "positions": list(window), "source": span["source"],
                       "full_sentence_token_count": len(positions),
                       "window_tokens": width,
                       "answer_anchor_used": anchor is not None})
    return sorted(chosen, key=lambda s: (s["role"], s["id"]))
