#!/usr/bin/env python3
"""Audit local 2026-09-16 run outputs and build compact publication tables.

The large original traces remain under the ignored results/ directory. This
script fails closed on missing/duplicated trace records or missing matrix cells.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


LANES = {
    "lane_a_ouro14": ("ouro14", 4),
    "lane_e_ouro14_q04_q07": ("ouro14", 6),
    "lane_b_ouro26": ("ouro26", 5),
    "lane_f_ouro26_q04_q07": ("ouro26", 7),
}
REVISIONS = {
    "ouro14": "37bc7e8ff85a90d813dc096383405f521b6a67b0",
    "ouro26": "f8e38372d72a8aa7f4ea9f14b98739a311589088",
}
SOURCE_SHA = "dbce2e8d1171faf3607f4e8043511c1d1e44ec46"
CALIBRATION_SHA = "b15a3765c4517cf0d8ab1a9f33cfe083c098f41a"
DIAG_SHA = "af049175aa05034779796d6e2fea0ee26e52685c"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    raw = root / "results/four_gpu_model_validation_20260916_114421"
    diag = root / "results/huginn_gpu0_diag_af049175_20260916_132117"
    output = root / "phase01/results/20260916_four_gpu_validation"
    output.mkdir(parents=True, exist_ok=True)
    receipt_dir = output / "run_receipts"
    receipt_dir.mkdir(exist_ok=True)
    sources: dict[str, dict] = {}

    def record(path: Path) -> None:
        assert path.is_file(), path
        rel = str(path.relative_to(root))
        sources[rel] = {"path": rel, "size_bytes": path.stat().st_size, "sha256": sha256(path)}

    trace_rows = []
    fair_rows = []
    validation_rows = []
    e2e_rows = []
    calibration_rows = []
    fit_rows = []
    for lane, (model, physical_gpu) in LANES.items():
        lane_root = raw / lane
        manifest_path = lane_root / "run_manifest.json"
        record(manifest_path)
        shutil.copyfile(manifest_path, receipt_dir / f"{lane}.json")
        manifest = json.loads(manifest_path.read_text())
        assert manifest["source_sha"] == SOURCE_SHA
        assert manifest["model_revision"] == REVISIONS[model]
        assert int(manifest["cuda_visible_devices"]) == physical_gpu

        for dataset, decode_tokens in (("alpaca", 64), ("sharegpt", 128)):
            trace_work_by_threshold = {}
            trace_paths = sorted(path for path in (lane_root / "traces").glob(f"{model}_{dataset}_q*.jsonl")
                                 if ".requests." not in path.name)
            assert len(trace_paths) == 2, (lane, dataset, trace_paths)
            for trace_path in trace_paths:
                record(trace_path)
                receipt_path = trace_path.with_name(trace_path.stem + ".requests.jsonl")
                record(receipt_path)
                receipts = [json.loads(line) for line in receipt_path.open()]
                assert {x["request_id"] for x in receipts} == set(range(256))
                assert all(x["status"] == "OK" and x["generated_tokens"] == decode_tokens
                           and x["trace_rows"] == decode_tokens and x["hit_decode_cap"] for x in receipts)
                seen = defaultdict(set)
                depths = Counter()
                token_count = 0
                metadata = None
                with trace_path.open() as stream:
                    for line in stream:
                        row = json.loads(line)
                        if metadata is None:
                            metadata = row
                        request_id, token_idx, depth = row["request_id"], row["token_idx"], row["depth"]
                        assert 0 <= request_id < 256 and 0 <= token_idx < decode_tokens
                        assert token_idx not in seen[request_id], (trace_path, request_id, token_idx)
                        assert 1 <= depth <= 4 and row["fixed_decode_tokens"]
                        assert row["kv_policy"] == "native-reference" and row["rmax"] == 4
                        seen[request_id].add(token_idx)
                        depths[depth] += 1
                        token_count += 1
                assert metadata is not None
                assert len(seen) == 256 and all(len(v) == decode_tokens for v in seen.values())
                assert token_count == 256 * decode_tokens
                trace_work_by_threshold[float(metadata["exit_threshold"])] = sum(
                    depth * count for depth, count in depths.items()
                )
                trace_rows.append({
                    "model": model, "model_revision": REVISIONS[model], "dataset": dataset,
                    "threshold": metadata["exit_threshold"], "kv_policy": "native-reference",
                    "rmax": 4, "gpu_physical": physical_gpu, "source_sha": SOURCE_SHA,
                    "requests": 256, "fixed_decode_tokens": decode_tokens,
                    "trace_rows": token_count, "eos_observed_requests": sum(x["eos_observed"] for x in receipts),
                    "depth_1_tokens": depths[1], "depth_2_tokens": depths[2],
                    "depth_3_tokens": depths[3], "depth_4_tokens": depths[4],
                    "mean_depth": sum(d * n for d, n in depths.items()) / token_count,
                    "trace_path_local": str(trace_path.relative_to(root)),
                    "trace_sha256": sources[str(trace_path.relative_to(root))]["sha256"],
                    "receipt_sha256": sources[str(receipt_path.relative_to(root))]["sha256"],
                })

            for name, target in (("summary_fair.csv", fair_rows),
                                 ("fairness_validation.csv", validation_rows)):
                path = lane_root / "analysis" / dataset / name
                record(path)
                frame = pd.read_csv(path)
                assert len(frame) == (60 if name == "summary_fair.csv" else 10)
                frame.insert(0, "model_key", model)
                frame.insert(1, "model_revision", REVISIONS[model])
                frame.insert(2, "gpu_physical", physical_gpu)
                frame.insert(3, "source_sha", SOURCE_SHA)
                if name == "fairness_validation.csv":
                    assert frame["fair"].all()
                else:
                    assert (frame["num_selected_tokens"] == 256 * decode_tokens).all()
                    for threshold, work in trace_work_by_threshold.items():
                        subset = frame[frame.exit_threshold == threshold]
                        assert len(subset) == 30
                        assert (subset["selected_work_checksum"] == work).all()
                        assert (subset["total_token_loop_work"] == work).all()
                target.append(frame)

            if model == "ouro26" and dataset == "sharegpt":
                suffix = "q02_q05" if lane == "lane_b_ouro26" else "q04_q07"
                e2e_path = raw / f"lane_{'g' if lane == 'lane_b_ouro26' else 'h'}_ouro26_sharegpt_e2e_{suffix}" / "ouro26_sharegpt_native.csv"
                e2e_gpu = 4 if lane == "lane_b_ouro26" else 6
            else:
                e2e_path = lane_root / "e2e" / f"{model}_{dataset}_native.csv"
                e2e_gpu = physical_gpu
            record(e2e_path)
            frame = pd.read_csv(e2e_path)
            assert len(frame) == 10 and set(frame.batch_size) == {4, 8, 16, 32, 64}
            assert set(frame.threshold) == ({0.2, 0.5} if lane in ("lane_a_ouro14", "lane_b_ouro26") else {0.4, 0.7})
            assert set(frame.model_revision) == {REVISIONS[model]}
            assert frame.fixed_output_tokens.all() and not frame.adaptive_compute_physically_realized.any()
            assert (frame.num_requests_target == 256).all()
            assert (frame.completed_requests + frame.failed_batches * frame.batch_size == 256).all()
            assert all(all("OutOfMemoryError" in item["error"] for item in json.loads(failures))
                       for failures in frame.failures_json)
            frame.insert(0, "model_key", model)
            frame.insert(1, "gpu_physical", e2e_gpu)
            frame.insert(2, "source_sha", SOURCE_SHA)
            frame.insert(3, "source_binding", "LAUNCH_CONTEXT_NO_RUN_MANIFEST" if
                         model == "ouro26" and dataset == "sharegpt" else "LANE_RUN_MANIFEST")
            frame["cohort_status"] = frame.completed_requests.map(lambda n: "COMPLETE" if n == 256 else "OOM_PARTIAL")
            frame["contract_generated_tokens_if_complete"] = frame.apply(
                lambda row: 256 * decode_tokens if row.completed_requests == 256 else float("nan"), axis=1
            )
            frame["contract_tokens_per_wall_s_if_complete"] = (
                frame.contract_generated_tokens_if_complete / frame.offline_bct_e2e_s
            )
            frame["full_cohort_requests_per_wall_s"] = frame.apply(
                lambda row: 256 / row.offline_bct_e2e_s if row.completed_requests == 256 else float("nan"), axis=1
            )
            e2e_rows.append(frame)

        if lane in ("lane_a_ouro14", "lane_b_ouro26"):
            for filename, target in ((f"{model}.csv", calibration_rows),
                                     (f"{model}_fit.csv", fit_rows)):
                path = lane_root / "calibration" / filename
                record(path)
                frame = pd.read_csv(path)
                frame.insert(0, "model_key", model)
                frame.insert(1, "gpu_physical", physical_gpu)
                # These files predate the dbce shard restart; the lane manifest
                # was overwritten when the trace/E2E shard relaunched.
                frame.insert(2, "source_sha", CALIBRATION_SHA)
                frame.insert(3, "source_binding", "INFERRED_FROM_FILE_TIME_AND_COMMIT_CHRONOLOGY")
                target.append(frame)

    assert len(trace_rows) == 16
    assert len(pd.concat(e2e_rows)) == 80
    pd.DataFrame(trace_rows).sort_values(["model", "dataset", "threshold"]).to_csv(output / "ouro_trace_audit.csv", index=False)
    pd.concat(fair_rows).to_csv(output / "ouro_headroom_fair.csv", index=False)
    pd.concat(validation_rows).to_csv(output / "ouro_fairness_validation.csv", index=False)
    pd.concat(e2e_rows).to_csv(output / "ouro_native_e2e.csv", index=False)
    pd.concat(calibration_rows).to_csv(output / "ouro_calibration.csv", index=False)
    pd.concat(fit_rows).to_csv(output / "ouro_calibration_fit.csv", index=False)

    gate_rows = []
    gate_parity = []
    for lane, gpu in (("lane_c2_huginn_latest_gate", 6),
                      ("lane_d2_huginn_compress_gate", 7)):
        lane_root = raw / lane
        manifest_path = lane_root / "run_manifest.json"
        record(manifest_path)
        shutil.copyfile(manifest_path, receipt_dir / f"{lane}.json")
        manifest = json.loads(manifest_path.read_text())
        assert int(manifest["cuda_visible_devices"]) == gpu
        for kind, filename in (("surface", "capacity/surface.csv"),
                               ("target", "capacity/target_b64_l2048_d16.csv")):
            path = lane_root / filename
            record(path)
            frame = pd.read_csv(path)
            frame.insert(0, "lane", lane)
            frame.insert(1, "gpu_physical", gpu)
            frame.insert(2, "source_sha", manifest["source_sha"])
            frame.insert(3, "probe_kind", kind)
            gate_rows.append(frame)
        path = lane_root / "parity/b4_vs_b1.csv"
        record(path)
        frame = pd.read_csv(path)
        frame.insert(0, "lane", lane)
        frame.insert(1, "gpu_physical", gpu)
        frame.insert(2, "source_sha", manifest["source_sha"])
        gate_parity.append(frame)
        for path in (lane_root / "parity").glob("*.json"):
            record(path)
            shutil.copyfile(path, output / f"{lane}_{path.name}")
    pd.concat(gate_rows).to_csv(output / "huginn_gate_capacity.csv", index=False)
    pd.concat(gate_parity).to_csv(output / "huginn_gate_batch_parity.csv", index=False)

    for name in ("run_manifest.json", "summary.json", "memory_cells.json", "adaptive_b1_policy_compare.json"):
        record(diag / name)
        shutil.copyfile(diag / name, output / f"huginn_gpu0_{name}")
    diag_manifest = json.loads((diag / "run_manifest.json").read_text())
    assert diag_manifest["source_sha"] == DIAG_SHA and diag_manifest["cuda_visible_devices"] == "0"
    memory = json.loads((diag / "memory_cells.json").read_text())
    diag_rows = []
    for row in memory["rows"]:
        diag_rows.append({key: value for key, value in row.items() if not isinstance(value, (dict, list))})
    assert len(diag_rows) == 15
    pd.DataFrame(diag_rows).to_csv(output / "huginn_gpu0_memory.csv", index=False)

    replay_paths = sorted((raw / "posthoc_replay").glob("*_b4_b8.csv"))
    if len(replay_paths) == 4:
        replay_rows = []
        for path in replay_paths:
            record(path)
            frame = pd.read_csv(path)
            assert set(frame.batch_size) == {4, 8}
            expected_tokens = 256 * (128 if "sharegpt" in path.name else 64)
            assert (frame.selected_generated_tokens == expected_tokens).all()
            assert (frame.hardware_replay_bct_ms > 0).all()
            assert frame.groupby(["run", "batch_size"]).size().eq(5).all()
            assert frame.groupby(["run", "batch_size"]).selected_generated_tokens.nunique().eq(1).all()
            frame.insert(0, "replay_source_sha", DIAG_SHA)
            replay_rows.append(frame)
        assert len(pd.concat(replay_rows)) == 16 * 2 * 5
        pd.concat(replay_rows).to_csv(output / "ouro_hardware_replay_b4_b8.csv", index=False)

    (output / "raw_source_inventory.json").write_text(
        json.dumps({"raw_root": str(raw.relative_to(root)),
                    "files": list(sources.values())}, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"wrote compact packet at {output} ({len(sources)} audited source files)")


if __name__ == "__main__":
    main()
