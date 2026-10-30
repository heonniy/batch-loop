"""Frozen, auditable evidence-vs-distractor workloads for recurrent KV study.

Controlled retrieval is NIAH/RULER-inspired, NOT an official RULER score.
HotpotQA default is a documented supporting-fact + distractor-sentence excerpt,
NOT the official full 10-paragraph benchmark. Select full_context explicitly
to use the complete supplied distractor setting; no implicit prompt truncation.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

HOTpot_REPO = "hotpotqa/hotpot_qa"
HOTpot_REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"
HOTpot_FILE = "distractor/validation-00000-of-00001.parquet"
HOTpot_SHA256 = "c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_text(title: str, sentence: str) -> str:
    return f"[{title}] {sentence.strip()}"


def controlled_retrieval(n: int, seed: int, n_facts: int = 16) -> tuple[list[dict], dict]:
    """One uniquely identifiable necessary fact and n_facts-1 unrelated facts.

    Positions cycle through front, middle and back; all code values are
    synthetic and generated once per uid. Every model receives the same data.
    """
    if n <= 0 or n_facts < 4 or n_facts > 512:
        raise ValueError("Invalid controlled retrieval sample/fact count")
    docs = []
    for i in range(n):
        rng = random.Random(f"evidence-v2/{seed}/{i}")
        codes = rng.sample(range(100000), n_facts)
        facts = []
        target_j = rng.randrange(n_facts)
        for j, code in enumerate(codes):
            record = f"R{i:04d}-{j:03d}"
            facts.append({"id": f"fact-{j:03d}",
                          "role": ("relevant_content" if j == target_j
                                   else "irrelevant_content"),
                          "text": f"Record {record} has access code VX-{code:05d}.",
                          "source": f"synthetic/fact-{j:03d}"})
        rng.shuffle(facts)
        # Deterministic placement independent of text and of model.
        target = next(s for s in facts if s["role"] == "relevant_content")
        facts.remove(target)
        target_index = (0, n_facts // 2, n_facts - 1)[i % 3]
        facts.insert(target_index, target)
        record_id = target["text"].split(" has access code")[0].removeprefix("Record ")
        answer = target["text"].split("access code ")[1].rstrip(".")
        docs.append({"uid": f"controlled-{seed}-{i:05d}",
                     "dataset": "controlled_retrieval",
                     "spans": facts,
                     "question": f"What is the access code for record {record_id}?",
                     "answer": answer,
                     "attributes": {"target_position": target_index,
                                    "n_facts": n_facts,
                                    "target_source": target["id"]}})
    source_hash = digest(json.dumps(docs, sort_keys=True).encode())
    return docs, {"repo": "synthetic-evidence-v2", "revision": "generator-v2",
                  "sha256": source_hash, "n": len(docs), "seed": seed,
                  "n_facts": n_facts, "protocol": "controlled-retrieval-v2"}


def _as_list(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _hotpot_sentences(row: dict) -> list[dict]:
    """Trust original (title, sent_id) annotation, not answer-string matches."""
    context, sf = row["context"], row["supporting_facts"]
    titles = _as_list(context["title"])
    paragraphs = _as_list(context["sentences"])
    supported = {(str(t), int(k)) for t, k in
                 zip(_as_list(sf["title"]), _as_list(sf["sent_id"]))}
    if len(titles) != len(paragraphs) or not supported:
        raise ValueError("Invalid HotpotQA context/support schema")
    seen = set()
    sentences = []
    for d, (title, paragraph) in enumerate(zip(titles, paragraphs)):
        for k, sentence in enumerate(_as_list(paragraph)):
            role = ("relevant_content" if (str(title), k) in supported
                    else "irrelevant_content")
            if role == "relevant_content":
                seen.add((str(title), k))
            sentences.append({"id": f"doc{d:02d}-sent{k:02d}",
                              "source": f"{title}/sent-{k}",
                              "title": str(title), "doc_idx": d, "sent_idx": k,
                              "role": role,
                              "text": _source_text(str(title), str(sentence))})
    if seen != supported:
        raise ValueError("HotpotQA supporting facts missing from context")
    if len({s["title"] for s in sentences
            if s["role"] == "relevant_content"}) < 2:
        raise ValueError("Require at least two distinct supporting documents")
    if not any(s["role"] == "irrelevant_content" for s in sentences):
        raise ValueError("No irrelevant Context for negative control")
    return sentences


def _excerpt(sentences: list[dict], uid: str, seed: int, limit: int) -> list[dict]:
    """Keep ALL supporting facts and sample unrelated-document distractors.

    Excerpts are analysis inputs; never claim them as an official full-context
    HotpotQA evaluation. First prefer distractors from distinct non-support
    documents to reduce duplicated near-identical background.
    """
    if limit < 1:
        raise ValueError("Need at least one distractor sentence")
    support = [s for s in sentences if s["role"] == "relevant_content"]
    support_titles = {s["title"] for s in support}
    irrelevant = [s for s in sentences if s["role"] == "irrelevant_content"]
    external = [s for s in irrelevant if s["title"] not in support_titles]
    rng = random.Random(f"hotpot-excerpt-v2/{seed}/{uid}")
    rng.shuffle(external)
    chosen = []
    seen_titles = set()
    for s in external:
        if s["title"] not in seen_titles and len(chosen) < limit:
            chosen.append(s)
            seen_titles.add(s["title"])
    remainder = [s for s in irrelevant if s["id"] not in {x["id"] for x in chosen}]
    rng.shuffle(remainder)
    chosen += remainder[:max(0, limit - len(chosen))]
    keep = {s["id"] for s in support + chosen}
    return [s for s in sentences if s["id"] in keep]


def hotpotqa_distractor(n: int, seed: int, *,
                        context_mode: str = "evidence_excerpt",
                        distractor_sentences: int = 8) -> tuple[list[dict], dict]:
    """Pinned HotpotQA-Distractor validation Parquet with sentence provenance."""
    if context_mode not in ("evidence_excerpt", "full_context"):
        raise ValueError("Unknown HotpotQA context mode")
    if n <= 0:
        raise ValueError("Invalid sample count")
    # Heavy dependencies are not imported during CPU fixture tests.
    import pandas as pd
    from huggingface_hub import hf_hub_download
    path = Path(hf_hub_download(HOTpot_REPO, filename=HOTpot_FILE,
                                repo_type="dataset", revision=HOTpot_REVISION))
    actual = digest(path.read_bytes())
    if actual != HOTpot_SHA256:
        raise RuntimeError(f"HotpotQA parquet checksum mismatch: {actual}")
    frame = pd.read_parquet(path)
    candidates = []
    invalid = 0
    for row in frame.to_dict("records"):
        try:
            sentences = _hotpot_sentences(row)
        except (ValueError, KeyError, TypeError):
            invalid += 1
            continue
        uid = str(row["id"])
        if context_mode == "evidence_excerpt":
            sentences = _excerpt(sentences, uid, seed, distractor_sentences)
        candidates.append({
            "uid": f"hotpotqa-distractor-{uid}", "dataset": "hotpotqa_distractor",
            "spans": [{k: s[k] for k in ("id", "role", "text", "source")}
                      for s in sentences],
            "question": str(row["question"]),
            "answer": str(row["answer"]),
            "attributes": {"question_type": str(row["type"]),
                           "difficulty": str(row["level"]),
                           "context_mode": context_mode,
                           "n_support_sentences":
                               sum(s["role"] == "relevant_content" for s in sentences),
                           "n_context_sentences": len(sentences)}})
    ordered = sorted(candidates, key=lambda d: digest(
        f"evidence-hotpot-v2/{seed}/{d['uid']}".encode()))
    if len(ordered) < n:
        raise RuntimeError(f"Only {len(ordered)} valid HotpotQA items (need {n})")
    chosen = ordered[:n]
    return chosen, {"repo": HOTpot_REPO, "revision": HOTpot_REVISION,
                    "file": HOTpot_FILE, "sha256": actual,
                    "split": "distractor/validation", "n_available": len(candidates),
                    "n_invalid_support": invalid, "n": n, "seed": seed,
                    "context_mode": context_mode,
                    "distractor_sentences": distractor_sentences,
                    "selection_sha256": digest("\n".join(
                        d["uid"] for d in chosen).encode())}


def load_evidence_workload(name: str, n: int, seed: int, *,
                           controlled_facts: int = 16,
                           hotpot_context_mode: str = "evidence_excerpt",
                           hotpot_distractor_sentences: int = 8) -> tuple[list[dict], dict]:
    if name == "controlled_retrieval":
        return controlled_retrieval(n, seed, controlled_facts)
    if name == "hotpotqa_distractor":
        return hotpotqa_distractor(n, seed, context_mode=hotpot_context_mode,
                                  distractor_sentences=hotpot_distractor_sentences)
    raise ValueError(f"Unknown evidence workload {name!r}")
