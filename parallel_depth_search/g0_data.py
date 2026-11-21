"""G0_DATA: freeze disjoint pilot/dev/test prompt-ID splits before any adaptive decision (CPU).

GSM8K test (1319): pilot = 0..127 (teacher-forced diagnostics + PoC history, incl. old E5
100..149); dev = 200..399 (n=200, all selection / calibration / champion choice);
test = 700..1199 (n=500, final only, frozen config). MATH-500 (500): dev = 0..199, test =
200..499 (n=300 < 500: final MATH claims are labelled sample-limited).
Checks: duplicate normalised question text across splits, and few-shot exemplar leakage
(few-shot exemplar questions must not appear in any split).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

SPLITS = {
    "gsm8k": {"pilot": (0, 128), "dev": (200, 400), "test": (700, 1200)},
    "math500": {"dev": (0, 200), "test": (200, 500)},
}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    from experiments.exp3_accuracy import tasks as ds
    man = {"rule": "Never use test IDs for calibration, threshold, predictor or champion selection.",
           "tasks": {}}
    dup_total, leak_total = 0, 0
    for task, fs in (("gsm8k", 3), ("math500", 4)):
        rows = ds.load_dataset_prompts(task, fs, 0)
        q = [norm(r["meta"]["question"] if task == "gsm8k" else r["prompt"].split("Problem:\n")[-1])
             for r in rows]
        prefix = rows[0]["prompt"][: rows[0]["prompt"].rfind("Q:" if task == "gsm8k" else "Problem:\n")]
        seen, info = {}, {}
        for name, (lo, hi) in SPLITS[task].items():
            ids = list(range(lo, min(hi, len(rows))))
            hs = [hashlib.sha256(q[i].encode()).hexdigest() for i in ids]
            for h in hs:
                seen.setdefault(h, set()).add(name)
            leak = sum(1 for i in ids if q[i][:80] and q[i][:80] in norm(prefix))
            leak_total += leak
            info[name] = dict(range=[lo, min(hi, len(rows))], n=len(ids),
                              ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                              questions_sha256=hashlib.sha256("".join(hs).encode()).hexdigest(),
                              fewshot_leaks=leak)
        dups = sum(1 for v in seen.values() if len(v) > 1)
        dup_total += dups
        man["tasks"][task] = dict(n_total=len(rows), fewshot=fs,
                                  fewshot_prefix_sha256=hashlib.sha256(prefix.encode()).hexdigest(),
                                  splits=info, cross_split_duplicates=dups)
    man["cross_split_duplicates"] = dup_total
    man["fewshot_leaks"] = leak_total
    man["split_manifest_sha256"] = hashlib.sha256(json.dumps(man["tasks"], sort_keys=True).encode()).hexdigest()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(man, indent=1) + "\n")
    print(json.dumps({k: man[k] for k in ("cross_split_duplicates", "fewshot_leaks", "split_manifest_sha256")}))


if __name__ == "__main__":
    main()
