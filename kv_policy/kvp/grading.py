"""Answer extraction and scoring with explicit parse-failure accounting.

Each grader returns a dict with the parsed prediction, the canonical gold, the primary
`correct` flag, `parse_ok`, a `failure_reason` and secondary reference metrics (lm-eval's own
filters/equivalence) so that the primary scorer can be audited against the reference one.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction

# ----------------------------------------------------------------------------- generic


def truncate_at_stops(text: str, until: tuple[str, ...]) -> str:
    """lm-eval generate_until post-processing: cut at the first occurrence of each stop string."""
    for term in until:
        if term:
            text = text.split(term)[0]
    return text


def last_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...} / \\fbox{...} with balanced braces (nesting-safe)."""
    best = None
    for m in re.finditer(r"\\(?:boxed|fbox)\s*\{", text):
        depth, i = 1, m.end()
        while i < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        if depth == 0:
            best = text[m.end(): i - 1]
    if best is None:
        m = None
        for m in re.finditer(r"\\boxed (\S+)", text):
            pass
        if m:
            best = m.group(1).rstrip("$.")
    return best


# ----------------------------------------------------------------------------- GSM8K

_NUM = r"-?(?:\\?\$)?\s*-?[0-9][0-9,]*(?:\.[0-9]+)?(?:/[0-9]+)?|-?(?:\\?\$)?\s*\.[0-9]+"


def canon_number(raw: str) -> Fraction | None:
    s = raw.strip()
    s = re.sub(r"\\(?:text|mathrm)\{[^}]*\}", "", s)
    s = s.replace("\\$", "").replace("$", "").replace(",", "").replace("\\!", "").replace(" ", "").rstrip(".")
    s = s.replace("−", "-")
    m = re.fullmatch(r"(-?)\\?(?:d?frac)\{(-?\d+)\}\{(-?\d+)\}", s)
    if m:
        try:
            v = Fraction(int(m.group(2)), int(m.group(3)))
        except ZeroDivisionError:
            return None
        return -v if m.group(1) else v
    m = re.fullmatch(r"(-?\d+)/(\d+)", s)
    if m and int(m.group(2)) != 0:
        return Fraction(int(m.group(1)), int(m.group(2)))
    if s.endswith("%"):
        s = s[:-1]
    try:
        return Fraction(Decimal(s))
    except (InvalidOperation, ValueError):
        return None


def gsm8k_gold(answer_field: str) -> Fraction:
    value = canon_number(answer_field.split("####")[-1])
    if value is None:
        raise ValueError(f"unparseable GSM8K gold: {answer_field!r}")
    return value


_ANSWER_IS = re.compile(r"[Tt]he (?:final )?answer is[:\s]*(?:\\\(|\$)?\s*(" + _NUM + r")")


def gsm8k_extract(text: str) -> tuple[Fraction | None, str, str | None]:
    """Returns (value, method, raw). Priority: 'The answer is N' (first, as the exemplars and
    lm-eval strict-match), '#### N', last \\boxed{N}. No last-number fallback in the primary score."""
    m = _ANSWER_IS.search(text)
    if m:
        return canon_number(m.group(1)), "answer_is", m.group(1)
    m = re.search(r"####\s*(" + _NUM + r")", text)
    if m:
        return canon_number(m.group(1)), "hashes", m.group(1)
    b = last_boxed(text)
    if b is not None:
        return canon_number(b), "boxed", b
    return None, "none", None


# lm-eval gsm8k_cot reference filters + exact_match normalisation (v0.4.9.1 yaml)
_LM_STRICT = re.compile(r"The answer is (\-?[0-9\.\,]+).")
_LM_FLEX = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")
_LM_IGNORE = [",", r"\$", r"(?s).*#### ", r"\.$"]


def _lm_norm(s: str) -> str:
    for pat in _LM_IGNORE:
        s = re.sub(pat, "", s)
    return s.lower().strip()


def lm_eval_gsm8k(text: str, answer_field: str) -> tuple[bool, bool]:
    gold = _lm_norm(answer_field.split("####")[-1].strip())
    strict = _LM_STRICT.findall(text)
    strict_pred = strict[0] if strict else "[invalid]"
    flex = _LM_FLEX.findall(text)
    flex_pred = (flex[-1][0] or flex[-1][1]) if flex else "[invalid]"
    return _lm_norm(strict_pred) == gold, _lm_norm(flex_pred) == gold


def grade_gsm8k(text: str, doc: dict) -> dict:
    gold = gsm8k_gold(doc["answer"])
    value, method, raw = gsm8k_extract(text)
    strict, flex = lm_eval_gsm8k(text, doc["answer"])
    if raw is None:
        reason = "no_answer_pattern"
    elif value is None:
        reason = "unparseable_number"
    else:
        reason = None
    return {
        "gold": str(gold), "pred": None if value is None else str(value), "pred_raw": raw,
        "extract_method": method, "parse_ok": value is not None,
        "correct": bool(value is not None and value == gold), "failure_reason": reason,
        "lm_eval_strict_match": strict, "lm_eval_flexible_extract": flex,
    }


# ----------------------------------------------------------------------------- MATH-500

_MINERVA_FINAL = re.compile(r"Final Answer: The final answer is(.*?)\. I hope it is correct\.", re.S)


def math_extract(text: str) -> tuple[str | None, str]:
    m = _MINERVA_FINAL.search(text)
    if m and m.group(1).strip():
        return m.group(1).strip(), "minerva_final"
    b = last_boxed(text)
    if b is not None and b.strip():
        return b.strip(), "boxed"
    m = re.search(r"[Tt]he (?:final )?answer is[:\s]*(.+?)(?:\.\s|\.$|\n|$)", text)
    if m and m.group(1).strip():
        return m.group(1).strip(), "answer_is"
    return None, "none"


def _strip_dollars(s: str) -> str:
    s = s.strip()
    while len(s) >= 2 and s[0] == "$" and s[-1] == "$":
        s = s[1:-1].strip()
    # percent sign: "10\\%" == "10" (applied to pred and gold alike)
    return re.sub(r"\\?%$", "", s).strip()


def math_equiv(pred: str, gold: str) -> tuple[bool, dict]:
    """Primary: math_verify (gold wrapped in $\\boxed{}$, pred wrapped in $...$).
    Reference: lm-eval minerva normalise+sympy is_equiv. Both reported."""
    from math_verify import LatexExtractionConfig, parse, verify
    from lm_eval.tasks.minerva_math import utils as mu

    pred_c = _strip_dollars(pred)
    gold = _strip_dollars(gold)
    info: dict = {}
    try:
        g = parse(f"$\\boxed{{{gold}}}$", extraction_config=[LatexExtractionConfig()])
        p = parse(f"$\\boxed{{{pred_c}}}$", extraction_config=[LatexExtractionConfig()])
        mv = bool(g) and bool(p) and bool(verify(g, p))
        info["math_verify_parsed"] = [str(x) for x in p[:1]]
    except Exception as exc:  # noqa: BLE001 - verifier failures are recorded, not fatal
        mv = False
        info["math_verify_error"] = repr(exc)[:200]
    try:
        minerva = bool(mu.is_equiv(mu.normalize_final_answer(pred_c), mu.normalize_final_answer(gold)))
    except Exception as exc:  # noqa: BLE001
        minerva = False
        info["minerva_error"] = repr(exc)[:200]
    info["minerva_is_equiv"] = minerva
    info["normalized_string_equal"] = _norm_str(pred_c) == _norm_str(gold)
    return mv, info


def _norm_str(s: str) -> str:
    s = re.sub(r"\\(?:left|right|!|,|;|:)", "", s)
    s = s.replace("dfrac", "frac").replace("tfrac", "frac").replace(" ", "")
    return s.strip("$.")


def grade_math(text: str, doc: dict) -> dict:
    gold = doc["answer"]
    pred, method = math_extract(text)
    out = {"gold": gold, "pred": pred, "pred_raw": pred, "extract_method": method, "parse_ok": pred is not None}
    if pred is None:
        out.update({"correct": False, "failure_reason": "no_answer_found", "minerva_is_equiv": False,
                    "normalized_string_equal": False})
        return out
    mv, info = math_equiv(pred, gold)
    # Primary = math_verify OR exact normalised string match (guards verifier parse failures on
    # plain-text answers such as \text{Evelyn}); disagreements with minerva are audited.
    correct = mv or info["normalized_string_equal"]
    out.update(info)
    out.update({"math_verify": mv, "correct": bool(correct), "failure_reason": None})
    return out


def grade(dataset: str, text: str, doc: dict) -> dict:
    return grade_gsm8k(text, doc) if dataset == "gsm8k" else grade_math(text, doc)
