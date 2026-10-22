"""Answer-parser fixtures: positive, negative and adversarial cases."""

from fractions import Fraction

import pytest

from kvp.grading import (canon_number, grade_gsm8k, grade_math, gsm8k_extract, last_boxed,
                         math_extract, truncate_at_stops)

GSM_DOC = {"answer": "Some work <<2*3=6>>6\n#### 1,234"}


@pytest.mark.parametrize("text,expected", [
    ("So the total is 1234. The answer is 1234.", Fraction(1234)),
    ("The answer is 1,234.", Fraction(1234)),
    ("The answer is $1,234.", Fraction(1234)),
    ("The answer is \\$1,234.", Fraction(1234)),
    ("The answer is 1234.00.", Fraction(1234)),
    ("Final: #### 1234", Fraction(1234)),
    ("thus \\boxed{1234}", Fraction(1234)),
])
def test_gsm8k_positive(text, expected):
    value, _, _ = gsm8k_extract(text)
    assert value == expected
    assert grade_gsm8k(text, GSM_DOC)["correct"]


@pytest.mark.parametrize("text,expected", [
    ("The answer is -5.", Fraction(-5)),
    ("The answer is -0.75.", Fraction(-3, 4)),
    ("The answer is .5.", Fraction(1, 2)),
    ("The answer is 3/4.", Fraction(3, 4)),
    ("The answer is \\frac{3}{4}.", None),  # regex only captures leading digits -> no number; see below
])
def test_gsm8k_signs_decimals(text, expected):
    value, _, raw = gsm8k_extract(text)
    if expected is None:
        assert value is None or raw is not None
    else:
        assert value == expected


def test_gsm8k_first_answer_is_wins():
    # mirrors lm-eval strict-match take_first: the first stated answer counts
    g = grade_gsm8k("The answer is 12. Wait, the answer is 1234.", GSM_DOC)
    assert g["pred"] == "12" and not g["correct"] and g["parse_ok"]


def test_gsm8k_parse_failure_distinct_from_wrong():
    g = grade_gsm8k("I am not sure how to solve this problem", GSM_DOC, finish_reason="eos")
    assert not g["parse_ok"] and g["failure_reason"] == "finished_no_answer" and not g["correct"]
    g = grade_gsm8k("The answer is 999.", GSM_DOC)
    assert g["parse_ok"] and not g["correct"] and g["failure_reason"] is None


def test_gsm8k_no_last_number_fallback_when_truncated():
    g = grade_gsm8k("We get 1234 apples in total.", GSM_DOC, finish_reason="length")
    assert not g["parse_ok"] and g["lm_eval_flexible_extract"]


def test_lm_eval_reference_strict():
    g = grade_gsm8k("The answer is 1,234.", GSM_DOC)
    assert g["lm_eval_strict_match"]


def test_canon_number():
    assert canon_number("$1,000,000") == 1_000_000
    assert canon_number("−7") == -7
    assert canon_number("abc") is None
    assert canon_number("12%") == 12


def test_truncate_at_stops():
    assert truncate_at_stops("ans 5.\n\nQ: next", ("Q:",)) == "ans 5.\n\n"
    assert truncate_at_stops("x</s>y<|im_end|>", ("Q:", "</s>", "<|im_end|>")) == "x"


# ----------------------------------------------------------------------------- MATH

@pytest.mark.parametrize("text,boxed", [
    ("so \\boxed{\\frac{1}{2}}", "\\frac{1}{2}"),
    ("\\boxed{\\frac{\\sqrt{3}}{\\frac{1}{2}}}", "\\frac{\\sqrt{3}}{\\frac{1}{2}}"),
    ("first \\boxed{1} then \\boxed{\\left( 3, \\frac{\\pi}{2} \\right)}", "\\left( 3, \\frac{\\pi}{2} \\right)"),
    ("unbalanced \\boxed{\\frac{1}{2}", None),
    ("\\fbox{7}", "7"),
    ("\\boxed 5", "5"),
])
def test_last_boxed(text, boxed):
    assert last_boxed(text) == boxed


def _doc(answer):
    return {"answer": answer}


@pytest.mark.parametrize("text,gold", [
    ("... \\boxed{\\frac12}.\nFinal Answer: The final answer is $\\frac{1}{2}$. I hope it is correct.", "\\frac{1}{2}"),
    ("so \\boxed{0.5}", "\\frac{1}{2}"),
    ("so \\boxed{\\dfrac{3}{4}}", "\\frac{3}{4}"),
    ("so \\boxed{\\left( 3, \\frac{\\pi}{2} \\right)}", "\\left( 3, \\frac{\\pi}{2} \\right)"),
    ("so \\boxed{x^2+2x+1}", "(x+1)^2"),
    ("so \\boxed{\\text{Evelyn}}", "\\text{Evelyn}"),
    ("Final Answer: The final answer is $[2,5)$. I hope it is correct.", "[2,5)"),
    ("so \\boxed{10\\%}", "10"),
    ("so \\boxed{\\sqrt{2}/2}", "\\frac{\\sqrt{2}}{2}"),
])
def test_math_positive(text, gold):
    g = grade_math(text, _doc(gold))
    assert g["parse_ok"] and g["correct"], g


@pytest.mark.parametrize("text,gold", [
    ("so \\boxed{\\frac{1}{3}}", "\\frac{1}{2}"),
    ("so \\boxed{(2,5]}", "[2,5)"),
    ("so \\boxed{-1}", "1"),
    ("so \\boxed{\\text{Bob}}", "\\text{Evelyn}"),
])
def test_math_negative(text, gold):
    g = grade_math(text, _doc(gold))
    assert g["parse_ok"] and not g["correct"], g


def test_math_parse_failure():
    g = grade_math("I could not finish the computation", _doc("3"), finish_reason="length")
    assert not g["parse_ok"] and g["failure_reason"] == "truncated_no_answer"


def test_math_minerva_priority_over_boxed():
    pred, method = math_extract("\\boxed{1} ... Final Answer: The final answer is $2$. I hope it is correct.")
    assert (pred, method) == ("$2$", "minerva_final")


# ----------------------------------------------------------------------------- fallbacks / taxonomy

def test_gsm8k_fallback_only_for_natural_termination():
    text = "So in total she pays 1234 dollars"
    g = grade_gsm8k(text, GSM_DOC, finish_reason="eos")
    assert g["correct"] and g["answer_source"] == "fallback" and not g["correct_format_only"]
    g = grade_gsm8k(text, GSM_DOC, finish_reason="length")
    assert not g["parse_ok"] and g["failure_reason"] == "truncated_no_answer"
    g = grade_gsm8k("I do not know", GSM_DOC, finish_reason="stop_string")
    assert g["failure_reason"] == "finished_no_answer"


@pytest.mark.parametrize("text,gold,ok", [
    ("The diagonal is $\\sqrt{576}=24$.", "24", True),
    ("Hence the area is $\\frac{3}{4}$.", "\\frac{3}{4}", True),
    ("Therefore the graph is a circle.", "\\text{circle}", True),
    ("The student with the greatest average speed is Angela.", "\\text{Angela}", True),
    ("So $x = 5$, which is the answer.\nThe result is $7$.", "5", False),
])
def test_math_fallback(text, gold, ok):
    g = grade_math(text, _doc(gold), finish_reason="eos")
    assert g["parse_ok"] and g["answer_source"] == "fallback" and g["correct"] == ok, g


def test_math_fallback_not_used_when_truncated():
    g = grade_math("The diagonal is $\\sqrt{576}=24$.", _doc("24"), finish_reason="length")
    assert not g["parse_ok"] and g["failure_reason"] == "truncated_no_answer"


def test_format_answer_takes_priority_over_fallback():
    g = grade_math("\\boxed{3} so the area is $4$.", _doc("3"), finish_reason="eos")
    assert g["extract_method"] == "boxed" and g["correct"] and g["correct_format_only"]
