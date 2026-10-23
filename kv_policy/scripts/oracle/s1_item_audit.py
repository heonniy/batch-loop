"""Stage 1.4: audit every generation of the R32 quality run (complete outputs, not just percentages).

Per (dataset, policy, item): length, finish reason, cap hit, parse status, primary vs reference
scorers, answer format, and a repetition-loop flag computed on the generated token ids. Writes
<run>/analysis/stage1_item_audit.csv and the aggregate <run>/analysis/stage1_item_audit.md.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

CAPS = {"gsm8k": 512, "math500": 2048}
REF_KEYS = {"gsm8k": ("lm_eval_strict_match", "lm_eval_flexible_extract"),
            "math500": ("math_verify", "minerva_is_equiv", "normalized_string_equal")}


def repetition(ids: list[int], tail: int = 256, n: int = 16) -> tuple[bool, int]:
    """Loop flag: some n-gram occurs >= 3 times in the last `tail` tokens; also the max count."""
    t = ids[-tail:]
    if len(t) < 3 * n:
        return False, 0
    c = Counter(tuple(t[i:i + n]) for i in range(len(t) - n + 1))
    m = max(c.values())
    return m >= 3, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-run", required=True)
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    rows = []
    for f in sorted((Path(args.base_run) / "raw").glob("*__gen__*.jsonl")):
        for line in f.read_text().splitlines()[1:]:
            r = json.loads(line)
            ds = r["dataset"]
            rep, rep_max = repetition(r["gen_token_ids"])
            refs = {k: r.get(k) for k in (*REF_KEYS["gsm8k"], *REF_KEYS["math500"])}
            ref_vals = [refs[k] for k in REF_KEYS[ds] if refs[k] is not None]
            rows.append({
                "dataset": ds, "policy": r["policy"], "mode": r["mode"], "uid": r["uid"],
                "prompt_tokens": r["prompt_tokens"], "n_gen_tokens": r["n_gen_tokens"],
                "finish_reason": r["finish_reason"], "hit_cap": r["n_gen_tokens"] >= CAPS[ds],
                "parse_ok": r["parse_ok"], "extract_method": r.get("extract_method"),
                "answer_source": r.get("answer_source"), "failure_reason": r.get("failure_reason"),
                "correct": r["correct"], **refs,
                "scorer_disagreement": bool(ref_vals) and any(bool(v) != bool(r["correct"]) for v in ref_vals),
                "has_final_answer_line": "Final Answer" in r["answer_text"] or "The answer is" in r["answer_text"],
                "repetition_loop": rep, "max_16gram_count_tail256": rep_max,
                "pred": r.get("pred"), "gold": r.get("gold"),
                "tail_text": r["answer_text"][-160:].replace("\n", "\\n"),
            })
    out = Path(args.run_dir) / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "stage1_item_audit.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    md = ["# Stage 1.4 item audit of the R32 quality run", "",
          f"Source: `{args.base_run}` (all generation records, {len(rows)} rows).", "",
          "| dataset | policy | n | acc | hit cap | cap & loop | cap & answer line | ended, no parse | "
          "scorer disagreements | loops (any) |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for ds in ("gsm8k", "math500"):
        for pol in ("full", "last", "average"):
            g = [r for r in rows if r["dataset"] == ds and r["policy"] == pol]
            cap = [r for r in g if r["hit_cap"]]
            md.append(f"| {ds} | {pol} | {len(g)} | {sum(r['correct'] for r in g) / len(g):.3f} | {len(cap)} | "
                      f"{sum(r['repetition_loop'] for r in cap)} | {sum(r['has_final_answer_line'] for r in cap)} | "
                      f"{sum((not r['hit_cap']) and not r['parse_ok'] for r in g)} | "
                      f"{sum(r['scorer_disagreement'] for r in g)} | {sum(r['repetition_loop'] for r in g)} |")
    md += ["", "Cap-hit generations that are NOT flagged as repetition loops (inspect manually):", ""]
    for r in rows:
        if r["hit_cap"] and not r["repetition_loop"]:
            md.append(f"- {r['dataset']} {r['policy']} `{r['uid']}` max16gram={r['max_16gram_count_tail256']}: "
                      f"…{r['tail_text'][-100:]}")
    md += ["", "Scorer disagreements (primary `correct` vs any reference metric):", ""]
    for r in rows:
        if r["scorer_disagreement"]:
            refs = ", ".join(f"{k}={r[k]}" for k in REF_KEYS[r["dataset"]])
            md.append(f"- {r['dataset']} {r['policy']} `{r['uid']}` correct={r['correct']} ({refs}) "
                      f"pred={r['pred']!r} gold={r['gold']!r}")
    (out / "stage1_item_audit.md").write_text("\n".join(md) + "\n")
    print("wrote", out / "stage1_item_audit.csv")


if __name__ == "__main__":
    main()
