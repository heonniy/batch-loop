"""Pinned dataset loading and deterministic seed-fixed subset selection."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

from .specs import GSM8K_REPO, MATH500_REPO, SELECTION_SEED, SUBSET_N

GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
MATH500_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_gsm8k_test() -> tuple[list[dict], dict]:
    root = Path(snapshot_download(GSM8K_REPO, repo_type="dataset", revision=GSM8K_REVISION))
    path = root / "main" / "test-00000-of-00001.parquet"
    frame = pd.read_parquet(path)
    docs = [
        {"uid": f"gsm8k-test-{i:04d}", "index": i, "question": r.question, "answer": r.answer}
        for i, r in enumerate(frame.itertuples(index=False))
    ]
    return docs, {"repo": GSM8K_REPO, "revision": GSM8K_REVISION, "file": str(path.relative_to(root)),
                  "sha256": _sha256_file(path), "rows": len(docs)}


def load_math500_test() -> tuple[list[dict], dict]:
    root = Path(snapshot_download(MATH500_REPO, repo_type="dataset", revision=MATH500_REVISION))
    path = root / "test.jsonl"
    docs = []
    for i, line in enumerate(path.read_text().splitlines()):
        row = json.loads(line)
        row["index"] = i
        row["uid"] = row["unique_id"]
        docs.append(row)
    return docs, {"repo": MATH500_REPO, "revision": MATH500_REVISION, "file": "test.jsonl",
                  "sha256": _sha256_file(path), "rows": len(docs)}


def select_gsm8k(docs: list[dict], n: int = SUBSET_N, seed: int = SELECTION_SEED) -> list[dict]:
    idx = sorted(random.Random(seed).sample(range(len(docs)), n))
    return [docs[i] for i in idx]


def select_math500(docs: list[dict], n: int = SUBSET_N, seed: int = SELECTION_SEED) -> list[dict]:
    """Proportional (largest-remainder) allocation over subject x level cells."""
    cells: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for d in docs:
        cells[(d["subject"], int(d["level"]))].append(d)
    keys = sorted(cells)
    exact = {k: len(cells[k]) * n / len(docs) for k in keys}
    alloc = {k: int(exact[k]) for k in keys}
    remainder = n - sum(alloc.values())
    for k in sorted(keys, key=lambda k: (-(exact[k] - alloc[k]), k))[:remainder]:
        alloc[k] += 1
    rng = random.Random(seed)
    chosen = []
    for k in keys:
        members = sorted(cells[k], key=lambda d: d["index"])
        chosen.extend(rng.sample(members, alloc[k]))
    return sorted(chosen, key=lambda d: d["index"])


def selection_hash(uids: list[str]) -> str:
    return hashlib.sha256("\n".join(uids).encode()).hexdigest()


def load_subset(dataset: str) -> tuple[list[dict], dict]:
    if dataset == "gsm8k":
        docs, meta = load_gsm8k_test()
        chosen = select_gsm8k(docs)
        meta["selection"] = f"random.Random({SELECTION_SEED}).sample(range({len(docs)}), {SUBSET_N}), sorted"
    elif dataset == "math500":
        docs, meta = load_math500_test()
        chosen = select_math500(docs)
        meta["selection"] = (f"subject x level proportional largest-remainder allocation, "
                             f"random.Random({SELECTION_SEED}) within cells, sorted by index")
    else:
        raise ValueError(dataset)
    uids = [d["uid"] for d in chosen]
    meta.update({"seed": SELECTION_SEED, "n": len(chosen), "uids": uids,
                 "selection_sha256": selection_hash(uids)})
    return chosen, meta
