"""Write <run>/banks.json: protected (layer, step) bank sets for OURO_SHARING_COLLAPSE_DIAGNOSIS_PLAN.md Step 1.

Cells are 0-based (layer, step); step 0 = recurrence 1. Fixed before any pilot outcome:
  bankP0     {(0,0)}                  primary: recurrence 1, physical layer 0
  bankNeg1   {(1,0)}                  negative control: recurrence 1, layer 1
  bankR1all  {(l,0) for every layer}  positive diagnostic: all layers at recurrence 1
  bankL0R12  {(0,0),(0,1)}            layer 0 at recurrences 1 and 2
  bankrand0-4  one random bank (l, r < R-1), != (0,0), seeded per model -> same byte budget as bankP0
  bankcal    calibration-selected on DISJOINT prompts (scripts/hof/e3_calibrate.py): the smallest
             cumulative top-k set whose calibration NLL recovers >= 95% of the LAST->FULL gap
             (largest evaluated k if none does). Added only when the model's calibration file exists.
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ARCH = {"ouro-1.4b": (24, 4), "ouro-2.6b": (48, 4), "huginn-0125": (4, 32)}
CALIB = "results/20261026T080106Z_huginn_ouro_full_last_convert/e3/calibration_{}.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--models", default="ouro-1.4b,ouro-2.6b,huginn-0125")
    args = ap.parse_args()
    out = {}
    notes = {}
    for m in args.models.split(","):
        L, R = ARCH[m]
        pol = {"bankP0": [(0, 0)], "bankNeg1": [(1, 0)], "bankR1all": [(l, 0) for l in range(L)], "bankL0R12": [(0, 0), (0, 1)]}
        cells = [(l, r) for l in range(L) for r in range(R - 1) if (l, r) != (0, 0)]
        for s in range(5):
            pol[f"bankrand{s}"] = [random.Random(f"bankrand:{m}:{s}").choice(cells)]
        cal = Path(CALIB.format(m))
        if cal.exists():
            c = json.loads(cal.read_text())
            gap = c["nll_last"] - c["nll_full"]
            chosen = None
            for e in c["cumulative_topk"]:
                if (c["nll_last"] - e["nll"]) / gap >= 0.95:
                    chosen = e
                    break
            chosen = chosen or c["cumulative_topk"][-1]
            pol["bankcal"] = [tuple(x) for x in chosen["cells"]]
            notes[m] = {"calibration_file": str(cal), "k": chosen["k"], "recovered_fraction": (c["nll_last"] - chosen["nll"]) / gap}
        out[m] = {k: [list(x) for x in v] for k, v in pol.items()}
    p = Path(args.run_dir) / "banks.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    (Path(args.run_dir) / "banks_notes.json").write_text(json.dumps(notes, indent=1))
    print(json.dumps(notes), {m: {k: v for k, v in d.items() if k.startswith("bankrand") or k == "bankcal"} for m, d in out.items()})


if __name__ == "__main__":
    main()
