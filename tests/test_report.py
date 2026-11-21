"""Tests for the generated README / resume bullets (fdda.report.readme)."""

from __future__ import annotations

import pandas as pd

from fdda.report import readme as rd


def _empty_facts() -> dict:
    return {
        k: None
        for k in [
            "hm",
            "myntra",
            "discount",
            "forecast",
            "customers",
            "segments",
            "playbook",
            "retention",
            "tagging",
            "effort",
            "hero",
        ]
    } | {"has_pdf": False}


def test_missing_outputs_render_as_tbd():
    text = rd.render_readme(_empty_facts())
    assert rd.TBD in text and "Key results" in text
    assert "None" not in text  # no raw None leaks into the page
    assert rd.TBD in rd.render_resume(_empty_facts())


def test_numbers_come_from_inputs():
    f = _empty_facts()
    f["hm"] = {
        "raw_rows": 31_788_324,
        "n_customers": 1_362_281,
        "date_min": "2018-09-20",
        "date_max": "2020-09-22",
    }
    f["discount"] = {
        "among_discounts": {"overall": {"ceiling_band": "15-25%", "ceiling_discount": 0.2}}
    }
    f["playbook"] = pd.DataFrame(
        {"customers": [10, 20], "customer_share": [0.1, 0.2], "revenue_at_stake": [1e6, 2e6]}
    )
    f["effort"] = {"effort_reduction": 0.62}
    text = rd.render_readme(f)
    assert "31.8M" in text and "31,788,324" in text
    assert "Markdown ceiling: 15-25% (about 20% off)" in text
    assert "3.0M indexed revenue at stake" in text and "30 churned or at-risk" in text
    assert "saved **62%**" in text


def test_formatters_handle_missing_and_nan():
    assert rd.pct(None) == rd.TBD and rd.pct(float("nan")) == rd.TBD
    assert rd.pct(0.1234, 1) == "12.3%"
    assert rd.millions(1_060_213, 2) == "1.06M"
    assert rd.dig({"a": {"b": 1}}, "a", "b") == 1 and rd.dig({"a": None}, "a", "b") is None
    assert rd.join_and(["x", "y", "z"]) == "x, y and z" and rd.join_and([]) == "none"
