"""Unit tests for plot_utils sizing and styling."""

import pytest
from plot_utils import set_size, apply_style


def test_set_size_dimensions():
    golden = (5**0.5 + 1) / 2
    # Single column square panel (1x1): width == height
    w1, h1 = set_size(columns=1, subplots=(1, 1), aspect="square")
    assert w1 == pytest.approx(240.0 / 72.27)
    assert h1 == pytest.approx(w1)

    # Double column 1x2 square panels: panel_width == panel_height
    w2, h2 = set_size(columns=2, subplots=(1, 2), aspect="square")
    assert w2 == pytest.approx(504.0 / 72.27)
    assert h2 == pytest.approx(w2 / 2)

    # Golden ratio mode
    _, h_g = set_size(columns=2, aspect="golden")
    assert h_g == pytest.approx(w2 / golden)




def test_apply_style_execution():
    apply_style(columns=1)
    apply_style(columns=2)
