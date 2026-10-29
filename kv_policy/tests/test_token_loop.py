"""CPU-only tests of per-token R1 preservation (no checkpoint downloads)."""
import pytest
import torch

from kvp.token_loop import TokenFirstBankKV, parse_token_policy, stable_hash_fraction


def make(policy, batch=1, pads=None, P=3):
    return TokenFirstBankKV(
        policy, R=4, n_rec=2, n_plain=0, B=batch, H=1, T=8,
        D=1, dtype=torch.float64, device="cpu",
        uids=[f"uid{i}" for i in range(batch)],
        pads=([0] * batch if pads is None else pads),
        prompt_end=P)


def fill_token(kv, col=0, first_vals=(1, 2)):
    for l, (k, v) in enumerate(zip(kv.rk, kv.rv)):
        for arr in (k, v):
            for r in range(4):
                arr[r, :, :, col, :] = (first_vals[0] if r == 0 else
                                       first_vals[1] if r == 3 else r + 10)


@pytest.mark.parametrize("policy,expected_first", [
    ("tokL1hash0", 2.0), ("tokL1hash100", 1.0),
    ("tokL1prompt", 1.0), ("tokL1decode", 2.0)])
def test_first_bank_read_policy(policy, expected_first):
    kv = make(policy)
    fill_token(kv)
    kv.finalize(0)
    for k, v in zip(kv.rk, kv.rv):
        assert k[0, 0, 0, 0, 0].item() == expected_first
        assert v[0, 0, 0, 0, 0].item() == expected_first
        for r in range(1, 4):
            assert k[r, 0, 0, 0, 0].item() == 2.0
    expected_fraction = .5 if expected_first == 1.0 else .25
    assert kv.logical_summary()["logical_recurrent_kv_vs_full"] == expected_fraction


def test_current_token_is_unmodified_until_finalize():
    kv = make("tokL1hash0")
    fill_token(kv)
    assert kv.rk[0][0, 0, 0, 0, 0].item() == 1
    assert kv.rk[0][1, 0, 0, 0, 0].item() == 11
    kv.finalize(0)
    assert kv.rk[0][0, 0, 0, 0, 0].item() == 2


def test_hash_is_causal_repeatable_and_seeded():
    for seed in (0, 1, 2):
        x = stable_hash_fraction("uid0", 12, seed)
        assert x == stable_hash_fraction("uid0", 12, seed)
        assert 0 <= x < 1
    assert parse_token_policy("tokL1hash40s2") == {
        "kind": "hash", "rate": .4, "seed": 2}
    with pytest.raises(ValueError):
        parse_token_policy("tokL1hash150")


def test_padding_and_prompt_decoding_split():
    kv = make("tokL1prompt", batch=2, pads=[0, 2], P=3)
    fill_token(kv, col=0)
    kv.finalize(0)
    assert kv.n_real_processed == 1
    assert kv.n_selected == 1
    fill_token(kv, col=2)
    kv.finalize(2)
    assert kv.n_real_processed == 3
    assert kv.n_selected == 3
    # The prompt for row 1 has only one REAL token at column 2.
    assert kv.logical_summary()["n_processed_real_token_positions"] == 3


def test_delta_selector_uses_only_completed_token_kv():
    kv = make("tokL1delta50", batch=2)
    fill_token(kv, col=0)
    for k, v in zip(kv.rk, kv.rv):
        # Row 1: identical first vs terminal -> no protection.
        k[0, 1, :, 0, :] = k[3, 1, :, 0, :]
        v[0, 1, :, 0, :] = v[3, 1, :, 0, :]
    kv.finalize(0)
    assert kv.logical_summary()["n_protected_token_positions"] == 1


def test_accounting_excludes_finished_generation_rows():
    kv = make("tokL1hash100", batch=2, pads=[0, 1], P=3)
    for c in range(5):
        fill_token(kv, col=c)
        kv.finalize(c)
    all_stats = kv.logical_summary()
    filtered = kv.logical_summary(output_lengths=[1, 3])
    assert filtered["n_processed_real_token_positions"] < all_stats["n_processed_real_token_positions"]
    assert filtered["logical_recurrent_kv_vs_full"] == .5
    assert filtered["loop1all_saving_fraction"] == 0.0
