"""LoopZip v5 Stage 0: evidence-coverage audit of committed Phase-1/2 results (CPU only).

Evidence E_u = aligned token positions annotated relevant_content (Controlled Retrieval: the
single target Record sentence; HotpotQA: union of annotated supporting-fact sentences).
Controlled Retrieval key (Record ID) and value (access code) subspans are derived from exact
tokenizer offsets of the frozen prompt text; any token straddling a subspan boundary, any
re-tokenization mismatch, or a code that differs from the gold answer fails closed.

Diagnostic metadata only: never an input to the Phase-1 policies being audited.
"""
from __future__ import annotations

import re
from typing import Sequence

CR_SENTENCE = re.compile(r"Record (?P<id>\S+) has access code (?P<code>[^\s.]+)\.")


class AlignmentError(ValueError):
    pass


def token_offsets(tokenizer, text: str, ids: Sequence[int]) -> list[tuple[int, int]]:
    res = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if list(res["input_ids"]) != list(ids):
        raise AlignmentError("Re-tokenized prompt differs from frozen prompt_ids")
    return [tuple(o) for o in res["offset_mapping"]]


def _tokens_in(offsets, a: int, b: int, text: str = "") -> tuple[list[int], list[int]]:
    """Tokens fully inside [a,b) and tokens straddling its boundary.

    A token's leading whitespace (BPE " V") is not content and does not count as straddling."""
    inside, straddle = [], []
    for i, (x, y) in enumerate(offsets):
        while text and x < y and text[x].isspace():
            x += 1
        if y <= x or y <= a or x >= b:
            continue
        (inside if a <= x and y <= b else straddle).append(i)
    return inside, straddle


def evidence_spans(job: dict, offsets) -> dict:
    """Per-job evidence sets: E, per-sentence spans and (CR) key/value subspans."""
    text = job["prompt_text"]
    rel = set(job["relevant_positions"])
    ctx = set(job["content_positions"])
    out = {"E": sorted(rel), "sentences": [], "boundary_tokens": 0, "ambiguous": []}
    for sp in job["context_spans"]:
        pos = list(sp["positions"])
        if not pos:
            continue
        a, b = offsets[pos[0]][0], offsets[pos[-1]][1]
        inside, straddle = _tokens_in(offsets, a, b, text)
        unaligned = [i for i in inside + straddle if i not in ctx]
        if sp["role"] == "relevant_content":
            out["boundary_tokens"] += len(unaligned)
            out["sentences"].append({"id": sp["id"], "source": sp.get("source", ""),
                                     "positions": pos, "unaligned_tokens": len(unaligned)})
    if not out["sentences"] or set().union(*(s["positions"] for s in out["sentences"])) != rel:
        raise AlignmentError("Relevant spans do not reproduce relevant_positions")
    if job["dataset"] == "controlled_retrieval":
        if len(out["sentences"]) != 1:
            raise AlignmentError("Controlled retrieval must have exactly one target sentence")
        pos = out["sentences"][0]["positions"]
        a, b = offsets[pos[0]][0], offsets[pos[-1]][1]
        m = CR_SENTENCE.search(text, a, b + 1)
        if not m or m.start() < a or m.end() > b + 1:
            raise AlignmentError("Target Record sentence not found at evidence offsets")
        if m.group("code") != job["gold_answer"].strip():
            raise AlignmentError("Evidence code differs from gold answer")
        for name in ("id", "code"):
            inside, straddle = _tokens_in(offsets, m.start(name), m.end(name), text)
            if straddle:
                out["ambiguous"].append(name)
            if not inside or not set(inside) <= rel:
                raise AlignmentError(f"Record {name} tokens not inside aligned evidence")
            out[name] = inside
            out[name + "_straddle"] = straddle
    return out


def coverage_row(retained: Sequence[int], ev: dict, context: Sequence[int],
                 scores: dict | None = None) -> dict:
    S = set(retained)
    E = set(ev["E"])
    non = set(context) - E
    row = {"n_evidence": len(E), "evidence_retained": len(S & E),
           "coverage_E": len(S & E) / len(E), "complete_E": E <= S,
           "evidence_missed": len(E - S),
           "nonevidence_coverage": len(S & non) / len(non) if non else None}
    sent = [len(S & set(s["positions"])) / len(s["positions"]) for s in ev["sentences"]]
    row.update(sentence_min_coverage=min(sent), sentence_mean_coverage=sum(sent) / len(sent),
               sentences_fully_kept_fraction=sum(c == 1 for c in sent) / len(sent))
    for name in ("id", "code"):
        if name in ev:
            t = set(ev[name])
            row[f"{name}_coverage"] = len(S & t) / len(t)
            row[f"{name}_complete"] = t <= S
    if scores:
        ranked = sorted(scores, key=lambda p: (-scores[p], int(p)))
        rank = {int(p): i for i, p in enumerate(ranked)}
        pct = [rank[p] / max(1, len(ranked) - 1) for p in E if p in rank]
        row["evidence_rank_percentile_mean"] = sum(pct) / len(pct) if pct else None
        row["evidence_rank_percentile_best"] = min(pct) if pct else None
    return row


def evidence_position(ev: dict, context: Sequence[int]) -> float:
    """Mean position of evidence within the Context, as a percentile in [0,1]."""
    ctx = sorted(context)
    idx = {p: i for i, p in enumerate(ctx)}
    v = [idx[p] / max(1, len(ctx) - 1) for p in ev["E"] if p in idx]
    return sum(v) / len(v)
