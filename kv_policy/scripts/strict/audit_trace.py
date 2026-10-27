"""Audit for FULL_STRICT_BENICE_VALIDATION.md: toy read/write trace and engine-vs-reference checks.

For tiny random Ouro (2 layers, R=4) and Huginn (2 core layers, R=4) models, prompt A,B -> generated C,D:
records, with the independent dictionary reference (kvp.reference), every K/V write and which
(layer, recurrence, token) version each attention query reads under full/decode, full/strict,
last/strict, first/strict (and last/convert for contrast), plus the call schedule (parallel prompt vs one token per
call). Checks the engine reproduces the reference logits for each policy (CPU float64).
Writes <out>/trace_reads_writes.csv, trace_table.md, trace_checks.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "diag"))

from kvp import reference as refmod  # noqa: E402
from s0_semantics import TracingStore  # noqa: E402

POLICIES = (("full/decode", "full", "decode"), ("full/strict", "full", "strict"),
            ("last/strict", "last", "strict"), ("first/strict", "first", "strict"),
            ("last/convert (contrast)", "last", "convert"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from tests.test_policy_semantics import _tiny

    rows, checks = [], []
    names = ["A", "B", "C", "D"]
    prompt, cont = [5, 17], [99, 3]
    for family in ("ouro", "huginn"):
        model, run = _tiny(family, 4)
        noise = run.bb.example_noise("u", 64) if family == "huginn" else None
        for label, pol, mode in POLICIES:
            log, calls = [], []
            orig_store, orig_sr = refmod._DictPolicyStore, TracingStore.store_and_read

            def factory(policy, R, region_start, keep=frozenset(), banks=frozenset(), _log=log):
                return TracingStore(policy, R, region_start, keep, banks, log=_log, names=names)

            def sr(self, kind, layer, r, k, v, _calls=calls):
                if kind == "rec" and layer == 0 and r == 0:
                    _calls.append("".join(names[self.p0 + i] for i in range(k.shape[2])))
                return orig_sr(self, kind, layer, r, k, v)

            refmod._DictPolicyStore = factory
            TracingStore.store_and_read = sr
            try:
                ref = refmod.reference_logits(model, family, run.bb.R, prompt, cont, pol, mode, noise)
            finally:
                refmod._DictPolicyStore = orig_store
                TracingStore.store_and_read = orig_sr
            eng = run.teacher_forced_logits(prompt, cont, "u", pol, mode)
            checks.append({"family": family, "policy": label, "call_schedule": calls,
                           "engine_vs_reference_max_abs_logit_diff": float((eng - ref).abs().max())})
            for r in log:
                rows.append({"family": family, "policy": label, **r})
    seen, uniq = set(), []
    for r in rows:
        k = tuple(r.values())
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    with open(out / "trace_reads_writes.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(uniq[0]))
        w.writeheader()
        w.writerows(uniq)
    (out / "trace_checks.json").write_text(json.dumps(checks, indent=1))
    md = ["| family | policy | query | layer | recurrence | reads A | reads B | reads C | prompt-call: B reads A |",
          "|---|---|---|---:|---:|---|---|---|---|"]
    for fam in ("ouro", "huginn"):
        for label, _, _ in POLICIES:
            sel = [r for r in uniq if r["family"] == fam and r["policy"] == label]
            for rr in (1, 2, 4):
                pb = next((r["version_read"] for r in sel if r["query_tokens"] == "B" and r["hist_token"] == "A"
                           and r["layer"] == 0 and r["recurrence"] == rr and r["event"] in ("read", "read_within_call")), "")
                for q in ("D",):
                    got = {r["hist_token"]: r["version_read"] for r in sel if r["event"] == "read" and r["query_tokens"] == q
                           and r["layer"] == 0 and r["recurrence"] == rr}
                    md.append(f"| {fam} | {label} | {q} | 0 | {rr} | {got.get('A', '')} | {got.get('B', '')} | {got.get('C', '')} | {pb} |")
    (out / "trace_table.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(json.dumps(checks, indent=1))


if __name__ == "__main__":
    main()
