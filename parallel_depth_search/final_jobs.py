"""Emit F1/F2 held-out e2e job lists exactly as frozen in FREEZE_20261121.md (CPU, launches nothing)."""
import argparse
import json

ARMS = {"huginn-3.5b": (32, "AR,trunc16,trunc24,skip1_15,jac_g0.1", "prev_shift"),
        "ouro-1.4b": (4, "AR,trunc3,skip1_2,jac_m1,jac_g0.1", "constant"),
        "ouro-2.6b": (4, "AR,trunc3,skip1_2,jac_m1,jac_g0.1", "constant")}
BATCHES = {"huginn-3.5b": (1, 16, 32, 64, 128), "ouro-1.4b": (1, 16, 64, 128), "ouro-2.6b": (1, 16, 64, 128)}
CHUNK = {("huginn-3.5b", 1): 25, ("huginn-3.5b", 16): 128, ("huginn-3.5b", 32): 256,
         ("ouro-1.4b", 1): 50, ("ouro-2.6b", 1): 50, ("ouro-1.4b", 16): 256, ("ouro-2.6b", 16): 256}


def job(node, model, B, idx, n, task="gsm8k", split="test", seed=20261121, dtype="bfloat16", arms=None):
    D, a, init = ARMS[model]
    argv = ["-m", "parallel_depth_search.e2e_decoder", "--model", model, "--depth", str(D), "--arms", arms or a,
            "--init", init, "--task", task, "--split", split, "--idx-start", str(idx), "--n", str(n),
            "--batch", str(B), "--max-new", "256", "--seed", str(seed), "--dtype", dtype, "--allow-gpu"]
    return dict(node_id=node, argv=argv, timeout_s=43200,
                fields=dict(model=model, B=B, idx=idx, n=n, task=task, split=split, seed=seed, dtype=dtype))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["F1", "F2"], required=True)
    p.add_argument("--skip", default="", help="model:B cells to omit (e.g. OOM on dev)")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    skip = set(x for x in a.skip.split(",") if x)
    E = []
    if a.stage == "F1":
        for model, bs in BATCHES.items():
            for B in bs:
                if f"{model}:{B}" in skip:
                    continue
                c = CHUNK.get((model, B), 500)
                for lo in range(700, 1200, c):
                    E.append(job("F1_FULL_ROLLOUT", model, B, lo, min(c, 1200 - lo)))
    else:
        for model in ARMS:                                   # second task, held-out MATH-500
            for lo in range(200, 500, 100 if model == "huginn-3.5b" else 300):
                E.append(job("F2_STRESS", model, 16, lo, 100 if model == "huginn-3.5b" else 300, task="math500"))
        # reproducibility: independent second run of one F1 cell with a different arm-order seed
        E.append(job("F2_STRESS", "ouro-1.4b", 16, 700, 500, seed=7))
        # precision: FP32 rerun of the same frozen arms on 200 test prompts
        E.append(job("F2_STRESS", "ouro-1.4b", 16, 700, 200, dtype="float32"))
    json.dump(E, open(a.out, "w"), indent=1)
    print(len(E))


if __name__ == "__main__":
    main()
