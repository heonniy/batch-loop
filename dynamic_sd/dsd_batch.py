"""Batched (fixed cohort, no refill) AR and DtV for Ouro with LIVE native exits.

NEW code on top of upstream OuroBlocksBatch (unmodified): adds the Ouro early_exit_gate
(cumulative halting prob >= tau, never at the last step -- same rule as upstream
OuroBlocks._advance_one_ut) and the same KV freeze rule (replicate last-computed depth's
K/V into skipped deeper depth slots), for a batch of B sequences.

Halted rows are MASKED (not advanced; union shrinks).  The number of sequential R calls
per lockstep decode step is still max over the batch (AR) or over all B*(K+1) sources (DtV
verify) -- this is what we want to measure.

Real greedy accept/reject (no emulation).  Equal-length prompts: each prompt is cropped to
its LAST L tokens (same cropped prompts for every arm).  No EOS: every row generates exactly
--max-new tokens (fixed cohort)."""
import argparse
import json
import os
import sys
import time

WF_ROOT = os.path.expanduser("~/dsd_work/wavefront-decoding")
sys.path.insert(0, WF_ROOT)
import torch  # noqa: E402

from experiments.exp2_spec_bench.run import MODEL_REGISTRY  # noqa: E402
from experiments.exp3_accuracy import tasks as ds  # noqa: E402
from src.decoding.ouro_blocks_batch import OuroBlocksBatch  # noqa: E402
from src.decoding.state import ActiveToken  # noqa: E402


class ExitBatch:
    def __init__(self, blocks: OuroBlocksBatch, tau):
        self.b = blocks
        self.tau = tau
        self.calls = 0
        self.rows = 0

    def advance(self, seq_ids, toks, cache):
        if not toks:
            return
        self.calls += 1
        self.rows += len(toks)
        pre_steps = [t.step for t in toks]
        self.b.advance_union(seq_ids, toks, cache)
        if self.tau is None:
            return
        x = torch.stack([t.hidden for t in toks], 0).unsqueeze(0)
        lam = torch.sigmoid(self.b.inner.early_exit_gate(x)[0, :, 0].float()).tolist()
        halted = []
        T = self.b.T_total
        for i, t in enumerate(toks):
            if pre_steps[i] < T - 1:
                t.exit_remaining *= (1.0 - lam[i])
                if (1.0 - t.exit_remaining) >= self.tau:
                    t.should_exit = True
                    halted.append((seq_ids[i], t))
        if halted:
            self._freeze(halted, cache)

    def _freeze(self, halted, cache):
        b = self.b
        T, nl, B, S = b.T_total, b.n_layers, cache.B, cache.S
        src, dst = [], []
        for sid, t in halted:
            last = t.step - 1
            for e in range(t.step, T):
                for l in range(nl):
                    src.append((b.slot_of(last, l) * B + sid) * S + t.position)
                    dst.append((b.slot_of(e, l) * B + sid) * S + t.position)
                    row = cache.seqlen[b.slot_of(e, l)]
                    if t.position + 1 > row[sid]:
                        row[sid] = t.position + 1
        if src:
            si = torch.tensor(src, dtype=torch.long, device=b.device)
            di = torch.tensor(dst, dtype=torch.long, device=b.device)
            cache._flatK.index_copy_(0, di, cache._flatK.index_select(0, si))
            cache._flatV.index_copy_(0, di, cache._flatV.index_select(0, si))


def logits_union(blocks, toks):
    x = torch.stack([t.hidden for t in toks], 0)
    return blocks.model.lm_head(x).float()


@torch.no_grad()
def ar_batch(blocks, eb, ids, max_new):
    B, L = ids.shape
    T = blocks.T_total
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    cache, last = blocks.prefill(ids)
    committed = [[x] for x in last.argmax(-1).tolist()]
    torch.cuda.synchronize()
    t_dec = time.perf_counter()
    eb.calls = eb.rows = 0
    depths, maxd = [], []
    while len(committed[0]) < max_new:
        toks = []
        for bi in range(B):
            t = ActiveToken(token_id=committed[bi][-1], position=L + len(committed[bi]) - 1, role="verify")
            toks.append(t)
        sids = list(range(B))
        blocks.prelude_union(sids, toks, cache)
        for _ in range(T):
            live = [(s, t) for s, t in zip(sids, toks) if not t.should_exit]
            if not live:
                break
            eb.advance([s for s, _ in live], [t for _, t in live], cache)
        ds_ = [t.step for t in toks]
        depths += ds_
        maxd.append(max(ds_))
        nxt = logits_union(blocks, toks).argmax(-1).tolist()
        for bi in range(B):
            committed[bi].append(nxt[bi])
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return committed, {"wall": t1 - t0, "decode_wall": t1 - t_dec, "rcalls": eb.calls, "rows": eb.rows,
                       "mean_depth": sum(depths) / len(depths), "mean_max_depth": sum(maxd) / len(maxd)}


@torch.no_grad()
def dtv_batch(blocks, eb, ids, max_new, d, K):
    B, L = ids.shape
    T = blocks.T_total
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    cache, last = blocks.prefill(ids)
    committed = [[x] for x in last.argmax(-1).tolist()]
    torch.cuda.synchronize()
    t_dec = time.perf_counter()
    eb.calls = eb.rows = 0
    rounds = verify_ticks = n_acc = n_prop = 0
    vt_list = []
    while True:
        act = [bi for bi in range(B) if len(committed[bi]) < max_new]
        if not act:
            break
        rounds += 1
        Kc = {bi: min(K, max_new - len(committed[bi])) for bi in act}
        srcs = {bi: [] for bi in act}
        drafts = {bi: [] for bi in act}
        exact = {bi: [] for bi in act}
        cur = {bi: (committed[bi][-1], L + len(committed[bi]) - 1) for bi in act}
        Kmax = max(Kc.values())
        for j in range(Kmax + 1):
            rows = [bi for bi in act if j <= Kc[bi]]
            toks = [ActiveToken(token_id=cur[bi][0], position=cur[bi][1], role="verify") for bi in rows]
            blocks.prelude_union(rows, toks, cache)
            for _ in range(d):
                live = [(s, t) for s, t in zip(rows, toks) if not t.should_exit]
                if not live:
                    break
                eb.advance([s for s, _ in live], [t for _, t in live], cache)
            lg = logits_union(blocks, toks)
            am = lg.argmax(-1).tolist()
            for k, bi in enumerate(rows):
                srcs[bi].append(toks[k])
                exact[bi].append(lg[k] if toks[k].should_exit else None)
                if j < Kc[bi]:
                    drafts[bi].append(am[k])
                cur[bi] = (am[k], cur[bi][1] + 1)
        vt = 0
        while True:
            live = [(bi, t) for bi in act for t in srcs[bi] if not t.should_exit and t.step < T]
            if not live:
                break
            eb.advance([s for s, _ in live], [t for _, t in live], cache)
            vt += 1
        verify_ticks += vt
        vt_list.append(vt)
        allt = [(bi, k, t) for bi in act for k, t in enumerate(srcs[bi])]
        lg = logits_union(blocks, [t for _, _, t in allt]).argmax(-1).tolist()
        gold = {bi: [None] * len(srcs[bi]) for bi in act}
        for (bi, k, t), g in zip(allt, lg):
            gold[bi][k] = int(exact[bi][k].argmax()) if exact[bi][k] is not None else g
        for bi in act:
            p = L + len(committed[bi]) - 1
            rej = False
            for i in range(Kc[bi]):
                n_prop += 1
                if drafts[bi][i] == gold[bi][i]:
                    n_acc += 1
                    committed[bi].append(drafts[bi][i])
                else:
                    committed[bi].append(gold[bi][i])
                    cache.drop_positions_at_and_after(bi, p + i + 1)
                    rej = True
                    break
            if not rej and len(committed[bi]) < max_new:
                committed[bi].append(gold[bi][Kc[bi]])
            committed[bi] = committed[bi][:max_new]
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return committed, {"wall": t1 - t0, "decode_wall": t1 - t_dec, "rcalls": eb.calls, "rows": eb.rows,
                       "rounds": rounds, "verify_ticks_per_round": verify_ticks / rounds,
                       "accept": n_acc / max(n_prop, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ouro-1.4b")
    ap.add_argument("--dataset", default="gsm8k")
    ap.add_argument("--B", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--L", type=int, default=512)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--tau", type=float, nargs="+", default=[None])
    ap.add_argument("--d", type=int, default=1)
    ap.add_argument("--K", type=int, nargs="+", default=[4])
    ap.add_argument("--n-batches", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    wcls, mid, _ = MODEL_REGISTRY[a.model]
    wrapper = wcls.from_pretrained(mid, dtype=torch.bfloat16, device="cuda")
    rows = ds.load_dataset_prompts(a.dataset, None, max(a.B) * a.n_batches)
    enc = [wrapper.encode(r["prompt"])[0] for r in rows]
    enc = [e[-a.L:] for e in enc if e.shape[0] >= a.L]
    print(f"[batch] {len(enc)} prompts with >= {a.L} tokens", flush=True)
    res = []
    blocks_cache = {}
    for B in a.B:
        if B not in blocks_cache:
            blocks_cache.clear()
            torch.cuda.empty_cache()
            blocks_cache[B] = OuroBlocksBatch(wrapper.model, T_total=4, batch_size=B,
                                              max_seq_len=a.L + a.max_new + 16)
        blocks = blocks_cache[B]
        for tau in a.tau:
            tau = None if tau is None or tau < 0 else tau
            eb = ExitBatch(blocks, tau)
            nb = min(a.n_batches, len(enc) // B)
            for bi in range(nb):
                ids = torch.stack(enc[bi * B:(bi + 1) * B]).cuda()
                if bi == 0:   # warmup
                    ar_batch(blocks, eb, ids[:, :], 8)
                    dtv_batch(blocks, eb, ids, 8, a.d, a.K[0])
                out_ar, st_ar = ar_batch(blocks, eb, ids, a.max_new)
                rec = {"B": B, "tau": tau, "batch": bi, "ar": st_ar}
                for K in a.K:
                    out_dtv, st_dtv = dtv_batch(blocks, eb, ids, a.max_new, a.d, K)
                    match = sum(int(x == y) for o1, o2 in zip(out_ar, out_dtv) for x, y in zip(o1, o2)) / (B * a.max_new)
                    st_dtv["token_match_vs_ar"] = match
                    st_dtv["speedup_decode"] = st_ar["decode_wall"] / st_dtv["decode_wall"]
                    rec[f"dtv_K{K}"] = st_dtv
                print(json.dumps(rec), flush=True)
                res.append(rec)
    json.dump({"args": vars(a), "res": res}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
