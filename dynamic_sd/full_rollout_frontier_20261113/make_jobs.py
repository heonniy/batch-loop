#!/usr/bin/env python3
"""Pre-registered job list for the 2026-11-13 run (printed in priority order, one shell command per line).

Lanes (all B=1, FULL KV, greedy, same target per arm, AR + DtV(d,K) + WFD(d), d fixed: Ouro 1, Huginn 4):
  parity : FP32 strict-exactness smoke, K in {2,4,8} + WFD, every benchmark class, max_new 256
  s1     : BF16 performance frontier, K in {1,2,4,8,16} + WFD
           gsm8k3   LoopSpec gsm8k-cot 3-shot, max_new 512, pilot offset 0 n100 + confirm offset 100 n200
           math500  LoopSpec/WaveFront minerva 4-shot, max_new 2048, offset 0 n50
           spec     WaveFront Spec-Bench, 8 prompts per task category (48), max_prompt 512, max_new 512
           gsm8k8   WaveFront exp3 8-shot, max_new 512, offset 0 n50 (reproduction lane)
  rep    : 5 timing repeats on gsm8k3 offset 0 n16
Exit regimes: Ouro fixed / tau0.4 / tau0.1 ; Huginn fixed / eps0.1 / eps0.2 (spec, gsm8k8: fixed + primary live).
"""
OUT = "full_rollout_frontier_20261113/results_20261113/raw"
R = "$PY full_rollout_frontier_20261113/run_frontier.py"
SPEC = [81, 91, 101, 111, 121, 131, 141, 151, 161, 171, 181, 191, 201, 211, 221, 231, 241, 251, 261, 271, 281,
        291, 301, 311, 321, 331, 341, 351, 361, 371, 381, 391, 401, 411, 421, 431, 441, 451, 461, 471, 481, 491,
        501, 511, 521, 531, 541, 551]
MODELS = {"ouro": "--model ouro-1.4b --T-total 4 --d 1", "huginn": "--model huginn-3.5b --T-total 32 --d 4"}
EXITS = {"ouro": [("fixed", "--exit fixed"), ("tau0.4", "--exit live --tau 0.4"), ("tau0.1", "--exit live --tau 0.1")],
         "huginn": [("fixed", "--exit fixed"), ("eps0.1", "--exit live --eps 0.1"), ("eps0.2", "--exit live --eps 0.2")]}
DATA = {"gsm8k3": "--dataset gsm8k --num-fewshot 3", "gsm8k8": "--dataset gsm8k --num-fewshot 8",
        "math500": "--dataset math500 --num-fewshot 4", "spec": "--dataset spec_bench --max-prompt-tokens 512"}

jobs = []


def job(lane, m, ex, ds, extra, tag, prio):
    name = f"{lane}/{m}_{ex[0]}_{ds}_{tag}"
    jobs.append((prio, f"{R} {MODELS[m]} {EXITS_MAP[m][ex[0]]} {DATA[ds]} {extra} --out {OUT}/{name}.jsonl"))


EXITS_MAP = {m: dict(v) for m, v in EXITS.items()}
for m in MODELS:
    for ex in EXITS[m]:
        # parity (priority 0)
        P = "--dtype float32 --Ks 2,4,8 --max-new-tokens 256 --protocol parity"
        job("parity", m, ex, "gsm8k3", f"{P} --offset 0 --limit 16", "o0n16", 0)
        job("parity", m, ex, "math500", f"{P} --offset 0 --limit 8", "o0n8", 0)
        job("parity", m, ex, "spec", f"{P} --spec-ids {','.join(map(str, SPEC[::6]))}", "n8", 0)
        S = "--dtype bfloat16 --Ks 1,2,4,8,16"
        # gsm8k3 pilot n100 (priority 1)
        step = 50 if m == "ouro" else 25
        for o in range(0, 100, step):
            job("s1", m, ex, "gsm8k3", f"{S} --max-new-tokens 512 --offset {o} --limit {step}", f"o{o}n{step}", 1)
        # math500 n50 (priority 2)
        step = 10 if m == "ouro" else 5
        for o in range(0, 50, step):
            job("s1", m, ex, "math500", f"{S} --max-new-tokens 2048 --offset {o} --limit {step}", f"o{o}n{step}", 2)
        primary = ex[0] in ("fixed", "tau0.4", "eps0.1")
        if primary:
            step = 8 if m == "ouro" else 4
            for i in range(0, 48, step):
                ids = ",".join(map(str, SPEC[i:i + step]))
                job("s1", m, ex, "spec", f"{S} --max-new-tokens 512 --spec-ids {ids}", f"i{i}n{step}", 3)
        # gsm8k3 confirmation n200 on separate prompts (priority 4)
        step = 50 if m == "ouro" else 25
        for o in range(100, 300, step):
            job("s1", m, ex, "gsm8k3", f"{S} --max-new-tokens 512 --offset {o} --limit {step}", f"o{o}n{step}", 4)
        if primary:
            step = 50 if m == "ouro" else 25
            for o in range(0, 50, step):
                job("s1", m, ex, "gsm8k8", f"{S} --max-new-tokens 512 --offset {o} --limit {step}", f"o{o}n{step}", 5)
        if ex[0] in ("tau0.4", "eps0.1"):
            job("rep", m, ex, "gsm8k3", f"{S} --max-new-tokens 512 --offset 0 --limit 16 --repeats 5", "o0n16r5", 6)

# within a priority, Huginn (slower) first for load balance
for prio, cmd in sorted(jobs, key=lambda x: (x[0], "ouro" in x[1])):
    print(cmd)
