"""CPU-only deterministic accounting tests (no GPU or model downloads)."""
import pytest
from kvp.knob_budget import KVLayout, layout_for_model


def test_huginn_r32_last_is_six_percent_not_one_over_32():
    m = layout_for_model("huginn-0125")
    assert (m.full_planes, m.last_planes) == (132, 8)
    assert m.last_over_full() == pytest.approx(8 / 132)
    assert m.protected_over_full(1) == pytest.approx(9 / 132)


def test_huginn_depth_sweep_includes_plain_layers():
    vals = [layout_for_model(f"huginn-0125-r{r}").last_over_full()
            for r in (4, 8, 16, 24)]
    assert vals == pytest.approx([8 / 20, 8 / 36, 8 / 68, 8 / 100])
    assert vals == sorted(vals, reverse=True)


def test_full_last_and_bank_cost_ouro():
    m = layout_for_model("ouro-1.4b")
    assert m.last_over_full() == pytest.approx(0.25)
    assert m.protected_over_full(1) == pytest.approx(25 / 96)
    assert m.protected_over_full(2) == pytest.approx(26 / 96)
    assert m.protected_over_full(24) == pytest.approx(0.5)


def test_sink_window_prompt_intersection_and_monotonicity():
    m = layout_for_model("huginn-0125")
    # At the final token: positions 0..3 and 896..1023 => 132 live
    r = m.postlast_window_over_full(1024, 128, sinks=4)
    assert r == pytest.approx((8 / 132) * (132 / 1024))
    # Pinning a 512-token prompt retains 512+128 tokens, no sink double-count.
    p = m.postlast_window_over_full(1024, 128, 4, 512, True)
    assert p == pytest.approx((8 / 132) * (640 / 1024))
    assert p > r


def test_invalid_knobs_blocked():
    m = KVLayout(4, 4, 32)
    with pytest.raises(ValueError):
        m.postlast_window_over_full(128, 0)
    with pytest.raises(ValueError):
        m.protected_over_full(999)
