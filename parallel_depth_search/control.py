"""Stdlib-only, append-only research supervisor. Never launches GPU or claims completion.

Decision tree is preregistered in tree.json. Receipts are GPU-run attestations
from an external executor. Code does not synthesize measurements or paper reviews.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TREE = ROOT / "tree.json"
STATES = {"executed", "not_run", "failed_infrastructure", "failed_correctness", "inconclusive"}
FINAL_STAGES = {"final"}
FOUNDATION_KEYS = {"G0_NATIVE": "native_parity_pass", "G0_KV": "exact_kv_commit_pass", "G0_DATA": "frozen_split_pass"}


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(obj):
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


def program():
    p = json.loads(TREE.read_text(encoding="utf-8"))
    validate_tree(p)
    return p


def validate_tree(p):
    nodes = p["nodes"]
    by_id = {n["id"]: n for n in nodes}
    if len(nodes) != len(by_id):
        raise ValueError("duplicate hypothesis node ID")
    if p.get("max_gpu_hours", 0) <= 0 or sum(n["planned_gpu_hours"] for n in nodes) > p["max_gpu_hours"]:
        raise ValueError("planned GPU hours exceed preregistered budget")
    for n in nodes:
        if not n["refs"] or not n["question"] or not n["method"]:
            raise ValueError(f"missing question/method/literature for {n['id']}")
        if n["planned_gpu_hours"] < 0 or n["id"] in n["requires_all"] + n.get("requires_any", []):
            raise ValueError(f"invalid node dependency: {n['id']}")
        for dep in n["requires_all"] + n.get("requires_any", []):
            if dep not in by_id:
                raise ValueError(f"unknown dependency {dep} for {n['id']}")
    stack, done = set(), set()

    def visit(id_):
        if id_ in stack:
            raise ValueError(f"cycle through {id_}")
        if id_ in done:
            return
        stack.add(id_)
        for dep in by_id[id_]["requires_all"] + by_id[id_].get("requires_any", []):
            visit(dep)
        stack.remove(id_)
        done.add(id_)
    for id_ in by_id:
        visit(id_)
    return True


def init_run(directory):
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    cfg = program()
    path = d / "manifest.json"
    value = {
        "schema_version": 1, "created": dt.datetime.now(dt.timezone.utc).isoformat(),
        "tree_sha256": digest(cfg), "source_commit": cfg["source_commit"],
        "max_gpu_hours": cfg["max_gpu_hours"],
        "status": "PLAN_ONLY_UNTIL_REAL_RECEIPTS"
    }
    if path.exists():
        if json.loads(path.read_text())["tree_sha256"] != digest(cfg):
            raise ValueError("tree changed after run init; create a new run dir")
        return value
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return value


def read_ledger(directory):
    d = Path(directory)
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    if manifest["tree_sha256"] != digest(program()):
        raise ValueError("run manifest mismatch; never silently rewrite prior design")
    path = d / "evidence.jsonl"
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                x = json.loads(line)
                if x["sha256"] != digest(x["receipt"]):
                    raise ValueError("evidence ledger hash mismatch")
                rows.append(x)
    return rows


def validate_receipt(r, p):
    nodes = {n["id"]: n for n in p["nodes"]}
    id_ = r.get("node_id")
    if id_ not in nodes:
        raise ValueError("unknown node_id")
    n = nodes[id_]
    if r.get("state") not in STATES:
        raise ValueError("receipt.state must be explicit, not a guessed GO/NO-GO")
    for k in ("trial_id", "source_git_sha", "command", "data_split", "model", "seed",
              "method", "literature_checks", "metrics", "hardware", "artifacts"):
        if k not in r:
            raise ValueError(f"receipt lacks {k}")
    if r["data_split"] not in ("hardware", "train", "pilot", "dev", "test"):
        raise ValueError("invalid data split")
    if r["data_split"] == "test" and n["phase"] != "final":
        raise ValueError("held-out test is reserved for final stage; no adaptive tuning on test")
    if n["phase"] == "final" and r["state"] == "executed" and r["data_split"] != "test":
        raise ValueError("final executed receipt requires held-out test")
    if not isinstance(r["metrics"], dict) or not isinstance(r["artifacts"], dict):
        raise ValueError("metrics/artifacts must be dictionaries")
    if r["state"] == "executed":
        checks = r["literature_checks"]
        if not isinstance(checks, list) or not checks:
            raise ValueError("executed trials require inspected references")
        for item in checks:
            if not isinstance(item, dict):
                raise ValueError("literature_checks entries must be records")
            if item.get("paper_id") not in n["refs"]:
                raise ValueError("literature must be relevant to the tested node")
            if not all(item.get(k) for k in ("url", "version", "claim", "transfer_caveat")):
                raise ValueError("need reviewed version, claim and transfer limitation")
        if r.get("gpu_hours", 0) <= 0:
            raise ValueError("executed trial must report positive GPU-hours")
        h = r["hardware"]
        if not isinstance(h, dict) or not h.get("gpu_name"):
            raise ValueError("executed receipt requires observed hardware, not H200 assumption")
        if not r["artifacts"].get("sha256"):
            raise ValueError("executed trial needs artifact checksum")
        if int(r["metrics"].get("sample_count", 0)) < 1:
            raise ValueError("executed trial must have real sample count")
    else:
        if r.get("gpu_hours", 0) not in (0, None):
            raise ValueError("no-run/failure receipt cannot claim measured GPU-hours")
        if not str(r.get("reason", "")).strip():
            raise ValueError("no-run/failure must explain why")
    return n


@contextlib.contextmanager
def file_lock(path):
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def ingest(directory, receipt_path):
    cfg = program()
    init_run(directory)
    r = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    validate_receipt(r, cfg)
    with file_lock(Path(directory) / ".ledger.lock"):
        rows = read_ledger(directory)
        if any(x["receipt"]["trial_id"] == r["trial_id"] for x in rows):
            raise ValueError("trial_id already ingested (immutable ledger)")
        spent = sum(x["receipt"].get("gpu_hours", 0) or 0 for x in rows)
        if spent + (r.get("gpu_hours", 0) or 0) > cfg["max_gpu_hours"] + 1e-6:
            raise ValueError("GPU-hours budget exhausted; do not hide cost")
        e = {"sha256": digest(r), "ingested": dt.datetime.now(dt.timezone.utc).isoformat(), "receipt": r}
        with open(Path(directory) / "evidence.jsonl", "a", encoding="utf-8") as f:
            f.write(canonical(e) + "\n")
            f.flush()
            os.fsync(f.fileno())
    return {"trial_id": r["trial_id"], "evidence_sha256": e["sha256"], "classification": classify(r, cfg)}


def classify(r, cfg):
    if r["state"] == "not_run":
        return "not_run"
    if r["state"] in ("failed_infrastructure", "failed_correctness"):
        return "investigate_failure"
    if r["state"] == "inconclusive":
        return "investigate_inconclusive"
    n = next(x for x in cfg["nodes"] if x["id"] == r["node_id"])
    m = r["metrics"]
    if n["id"] in FOUNDATION_KEYS:
        return "completed" if m.get(FOUNDATION_KEYS[n["id"]]) is True else "investigate_correctness"
    count = int(m.get("sample_count", 0))
    if n["phase"] == "final":
        if count < cfg["final_min_samples"]:
            return "inconclusive_sample_size"
    elif count < cfg["screen_min_samples"]:
        return "inconclusive_sample_size"
    if n["evidence_kind"] == "correctness":
        return "completed" if m.get("correctness_pass") is True else "investigate_correctness"
    if n["evidence_kind"] != "performance":
        return "completed_diagnostic"
    if r["data_split"] not in ("dev", "test"):
        return "screen_only"
    if m.get("full_e2e_measured") is not True:
        return "not_full_e2e"
    if not isinstance(m.get("speedup_ci_low"), (int, float)) or not isinstance(m.get("accuracy_delta_ci_low"), (int, float)):
        return "inconclusive_missing_ci"
    if m["accuracy_delta_ci_low"] < -cfg["noninferiority_margin_accuracy"]:
        return "contradicted_quality"
    if m.get("unverified_kv_fidelity") is True:
        return "investigate_kv_fidelity"
    if m["speedup_ci_low"] > cfg["minimum_speedup_lower_ci"]:
        return "promising_candidate" if n["phase"] != "final" else "supported_heldout"
    return "inconclusive_or_no_speedup"


def state_for(directory):
    cfg = program()
    rows = read_ledger(directory)
    out = {}
    for rec in rows:
        r = rec["receipt"]
        out[r["node_id"]] = {"trial_id": r["trial_id"], "state": r["state"],
                             "classification": classify(r, cfg), "data_split": r["data_split"],
                             "sample_count": r["metrics"].get("sample_count", 0)}
    return cfg, rows, out


def propose(directory):
    cfg, rows, state = state_for(directory)
    finished = {"completed", "completed_diagnostic", "promising_candidate",
                "supported_heldout", "contradicted_quality", "inconclusive_or_no_speedup"}
    ready, investigate, blocked = [], [], []
    for n in cfg["nodes"]:
        id_, current = n["id"], state.get(n["id"])
        if current:
            if current["classification"] not in finished:
                investigate.append({"id": id_, "action": "diagnose_or_expand_sample", "why": current["classification"]})
            continue
        missing = [x for x in n["requires_all"] if state.get(x, {}).get("classification") not in finished]
        any_ok = not n.get("requires_any") or any(state.get(x, {}).get("classification") in finished
                                                  for x in n["requires_any"])
        if missing or not any_ok:
            blocked.append({"id": id_, "requires": missing, "any_option_done": any_ok})
        else:
            ready.append({"id": id_, "phase": n["phase"], "gpu_hours": n["planned_gpu_hours"],
                          "question": n["question"], "paper_ids": n["refs"], "method": n["method"]})
    spent = sum(x["receipt"].get("gpu_hours", 0) or 0 for x in rows)
    remaining = cfg["max_gpu_hours"] - spent
    if remaining <= 0:
        ready = []
        investigate.append({"id": "BUDGET", "why": "inconclusive_budget"})
    return {"run_gpu_hours": spent, "remaining_gpu_hours": remaining,
            "completed_or_attempted": state, "eligible_next": ready, "needs_investigation": investigate, "blocked": blocked}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    sub.add_parser("validate")
    for name in ("init", "status", "propose"):
        z = sub.add_parser(name)
        z.add_argument("--run-dir", required=True)
    z = sub.add_parser("ingest")
    z.add_argument("--run-dir", required=True)
    z.add_argument("--receipt", required=True)
    a = p.parse_args(argv)
    if a.action == "validate":
        cfg = program()
        output = {"valid": True, "nodes": len(cfg["nodes"]), "gpu_hours": cfg["max_gpu_hours"]}
    elif a.action == "init":
        output = init_run(a.run_dir)
    elif a.action == "status":
        cfg, rows, state = state_for(a.run_dir)
        output = {"nodes": len(cfg["nodes"]), "receipt_count": len(rows), "status": state}
    elif a.action == "propose":
        output = propose(a.run_dir)
    elif a.action == "ingest":
        output = ingest(a.run_dir, a.receipt)
    else:
        raise ValueError(a.action)
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
