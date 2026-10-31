"""LoopZip v6 workloads W1 (natural-language continuation) and W2 (instruction-conditioned generation).

Source text: Wikipedia paragraphs of the pinned HotpotQA-Distractor validation Parquet (the same
checksum-verified file as v5). NO question answers or supporting-fact labels are used; HotpotQA is only
a frozen corpus of natural paragraphs. Items are ordered by a seeded digest, assigned to disjoint
Stage A / Stage B-C / holdout / reserve pools BEFORE any model output exists, and each prompt must fit
the frozen token-length window for BOTH model tokenizers (else it is skipped and the next reserve
item is used; skips are counted in the manifest).
"""
from __future__ import annotations

import hashlib

from .role_data import RolePrompt, _label_tokens

W2_OPS = (
    ("summarize", "Summarize the passage above in a few sentences of your own words.", None),
    ("explain", "Explain the main subject of the passage above to a general reader.", None),
    ("paraphrase", "Paraphrase the passage above sentence by sentence.", None),
    ("synthesize", "Write a short synthesis that connects the two passages above.",
     "What links {a} and {b}?"),
)
W1_INSTRUCTION = "Continue the passage above with the next few sentences."
SYSTEM = "You are a helpful assistant."


def _digest(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def build_trajectory_prompt(tokenizer, family: str, context: str, instruction: str,
                            question: str | None = None) -> RolePrompt:
    """Context -> Instruction -> optional Question -> 'Response:' with exact tokenizer-offset roles."""
    parts, spans = [], {}

    def add(text, key=None):
        start = sum(len(x) for x in parts)
        parts.append(text)
        if key:
            spans[key] = (start, start + len(text))

    add("Context:\n")
    add(context.strip(), "content")
    add("\n\nInstruction:\n")
    add(instruction, "instruction")
    if question:
        add("\n\nQuestion:\n")
        add(question, "question")
    add("\n\nResponse:\n")
    body = "".join(parts)
    if family == "huginn":
        body = body.rstrip()  # Huginn's chat template strips trailing whitespace from content
        text = tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM},
                                              {"role": "user", "content": body}],
                                             tokenize=False, add_generation_prompt=True)
        if text.count(body) != 1:
            raise ValueError("Chat template did not preserve the user body exactly once")
        shift = text.index(body)
    elif family == "ouro":
        text, shift = body, 0
    else:
        raise ValueError("Unsupported family")
    ids, roles = _label_tokens(tokenizer, text, {k: (a + shift, b + shift) for k, (a, b) in spans.items()})
    pos = {r: tuple(i for i, x in enumerate(roles) if x == r) for r in ("content", "instruction", "question")}
    if not pos["content"] or not pos["instruction"] or (question and not pos["question"]):
        raise ValueError("Missing aligned role span")
    return RolePrompt(text, ids, roles, pos["content"], pos["instruction"] + pos["question"], (),
                      question_positions=pos["question"])


def _paragraphs(row) -> list[tuple[str, list[str]]]:
    c = row["context"]
    titles = list(c["title"].tolist() if hasattr(c["title"], "tolist") else c["title"])
    sents = [list(s.tolist() if hasattr(s, "tolist") else s) for s in c["sentences"]]
    return [(str(t), [str(x).strip() for x in ss if str(x).strip()]) for t, ss in zip(titles, sents)]


def candidate_items(frame, workload: str, seed: int):
    """Yield deterministic (item_id, context, instruction, question, attrs) in seeded digest order.
    Context sentences are a growing prefix; the caller picks the first length-feasible prefix."""
    rows = sorted(frame.to_dict("records"), key=lambda r: _digest(f"v6/{workload}/{seed}/{r['id']}"))
    for k, row in enumerate(rows):
        # longest paragraphs first (by characters); ties by original order
        paras = sorted((p for p in _paragraphs(row) if len(p[1]) >= 2),
                       key=lambda p: -sum(len(x) for x in p[1]))
        if workload == "W1":
            if not paras:
                continue
            title, ss = paras[0]
            yield {"item_id": f"hotpot-{row['id']}-W1", "source_id": str(row["id"]), "titles": [title],
                   "sentences": [ss], "op": "continue", "instruction": W1_INSTRUCTION, "question": None}
        else:
            op, instr, q = W2_OPS[int(_digest(f"v6op/{seed}/{row['id']}"), 16) % len(W2_OPS)]
            if op == "synthesize":
                if len(paras) < 2:
                    continue
                (ta, sa), (tb, sb) = paras[0], paras[1]
                yield {"item_id": f"hotpot-{row['id']}-W2", "source_id": str(row["id"]), "titles": [ta, tb],
                       "sentences": [sa, sb], "op": op, "instruction": instr, "question": q.format(a=ta, b=tb)}
            elif paras:
                title, ss = paras[0]
                yield {"item_id": f"hotpot-{row['id']}-W2", "source_id": str(row["id"]), "titles": [title],
                       "sentences": [ss], "op": op, "instruction": instr, "question": None}


def context_text(item, n_sent: int) -> str:
    """Use the first n_sent sentences of each paragraph (paragraph texts separated by a blank line)."""
    return "\n\n".join(" ".join(ss[:n_sent]) for ss in item["sentences"])


def fit_item(item, tokenizers: dict, lo: int, hi: int):
    """Smallest per-paragraph sentence count whose prompt length is in [lo, hi] for EVERY model."""
    longest = max(len(ss) for ss in item["sentences"])
    for n in range(1, longest + 1):
        ctx = context_text(item, n)
        lens = {}
        for fam_key, (fam, tok) in tokenizers.items():
            try:
                p = build_trajectory_prompt(tok, fam, ctx, item["instruction"], item["question"])
            except (ValueError, RuntimeError):
                return None
            lens[fam_key] = len(p.token_ids)
        if all(lo <= v <= hi for v in lens.values()):
            return {"n_sent": n, "context": ctx, "lengths": lens}
        if any(v > hi for v in lens.values()):
            return None
    return None
