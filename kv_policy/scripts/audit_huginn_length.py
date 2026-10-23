"""Audit generation cutoffs and policy comparisons; report measured, not inferred, length effects."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run-dir",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();run=Path(a.run_dir)
    buckets=defaultdict(list)
    for fp in sorted((run/"raw").glob("*.jsonl")):
        with fp.open() as f:
            meta=json.loads(next(f))["job_meta"]
            if meta["task"]!="gen" or meta["model"]!="huginn-0125":continue
            key=(meta["dataset"],meta["policy"],meta["mode"])
            buckets[key].extend(json.loads(x) for x in f if x.strip())
    lines=["# Huginn R32 generation-length audit","","R32, fixed exit. Compare cutoffs 256/512 (GSM8K) and 1024/2048 (MATH).",
           "A generation finishing before the expanded cap is verified; a sequence exceeding an old cap is only potentially rescued, not proof that the old run would have followed the same trajectory.",
           "","| Dataset | Policy / mode | n | Acc | Mean tokens | hit extended cap | exceeded old cap | ended without answer |",
           "|---|---|---:|---:|---:|---:|---:|---:|"]
    for k,rows in sorted(buckets.items()):
        cap=512 if k[0]=="gsm8k" else 2048;old=cap//2
        n=len(rows); hit=sum(r.get("finish_reason")=="length" for r in rows)
        exceeded=sum(r.get("n_gen_tokens",0)>old for r in rows)
        noans=sum(not r.get("parse_ok",False) for r in rows)
        acc=sum(bool(r.get("correct")) for r in rows)/n
        avg=sum(r.get("n_gen_tokens",0) for r in rows)/n
        lines.append(f"| {k[0]} | {k[1]}/{k[2]} | {n} | {acc:.3f} | {avg:.1f} | {hit} | {exceeded} | {noans} |")
    if not buckets:raise SystemExit("No generation results found")
    for k,rows in buckets.items():
        if len(rows)!=200:raise SystemExit(f"Incomplete cell {k}: {len(rows)} of 200; refusing summary")
    Path(a.output).write_text("\n".join(lines)+"\n")
if __name__=="__main__":main()
