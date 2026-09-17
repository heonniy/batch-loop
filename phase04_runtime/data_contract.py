"""Effective-input split and immutable-artifact checks for Phase-4 cohorts."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def effective_token_hashes(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> list[str]:
    if input_ids.shape != attention_mask.shape or input_ids.ndim != 2:
        raise ValueError("token IDs and attention mask must have matching [N,S] shapes")
    output = []
    for tokens, valid in zip(input_ids.cpu(), attention_mask.cpu().bool()):
        selected = tokens[valid].long().tolist()
        payload = len(selected).to_bytes(8, "little") + b"".join(
            int(token).to_bytes(8, "little", signed=True) for token in selected)
        output.append(hashlib.sha256(payload).hexdigest())
    return output


def audit_splits(hashes: list[str], families: list[str], splits: dict[str, list[int]]) -> dict:
    if len(hashes) != len(families):
        raise ValueError("family and tokenized-input counts differ")
    all_ids = [index for group in splits.values() for index in group]
    if sorted(all_ids) != list(range(len(hashes))):
        raise ValueError("request splits must partition every canonical ID exactly once")
    family_owner: dict[str, str] = {}
    hash_owner: dict[str, str] = {}
    within_split_duplicates = 0
    for name, ids in splits.items():
        seen_here: set[str] = set()
        for index in ids:
            family = families[index]
            old_family = family_owner.setdefault(family, name)
            if old_family != name:
                raise ValueError(f"conversation/document family crosses splits: {family}")
            old_hash = hash_owner.setdefault(hashes[index], name)
            if old_hash != name:
                raise ValueError(f"effective model input crosses splits: {hashes[index]}")
            if hashes[index] in seen_here:
                within_split_duplicates += 1
            seen_here.add(hashes[index])
    return {"split_sizes": {name: len(ids) for name, ids in splits.items()},
            "unique_effective_inputs": len(set(hashes)),
            "within_split_duplicate_rows": within_split_duplicates,
            "cross_split_duplicate_rows": 0,
            "effective_input_hashes": hashes}


def verify_artifacts(root: Path, expected: dict[str, str], required: tuple[str, ...]) -> None:
    for name in required:
        if name not in expected:
            raise ValueError(f"missing expected artifact digest: {name}")
        path = root / name
        if not path.is_file() or sha256_file(path) != expected[name]:
            raise ValueError(f"missing or changed artifact: {path}")
