"""Tests for the ML product tagger (fdda.tagging)."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from fdda.config import load_settings
from fdda.tagging import labels as lb
from fdda.tagging import model as md
from fdda.tagging.effort_study import match_option, summarize

SETTINGS = load_settings()
CFG = SETTINGS["tagging"]


def _xgboost_available() -> bool:
    try:
        import xgboost  # noqa: F401
    except Exception:  # XGBoostError if the OpenMP runtime is missing
        return False
    return True


def test_map_colour_rules():
    rules = CFG["hm_eval"]["colour_rules"]
    assert lb.map_colour("Dark Blue", rules) == "Blue"
    assert lb.map_colour("Light Turquoise", rules) == "Blue"
    assert lb.map_colour("Greyish Beige", rules) == "Beige"
    assert lb.map_colour("Yellowish Brown", rules) == "Brown"
    assert lb.map_colour("Gold", rules) == "Metallic"
    assert lb.map_colour("Transparent", rules) == lb.UNKNOWN
    assert lb.map_colour(None, rules) == lb.UNKNOWN


def test_truth_labels_mapping():
    df = pd.DataFrame(
        {
            "product_group_name": ["Garment Lower body", "Garment Upper body", "Furniture"],
            "product_type_name": ["Trousers", "T-shirt", "Cushion"],
            "colour_group_name": ["Black", "Unknown", "Light Pink"],
            "graphical_appearance_name": ["Solid", "All over pattern", "Neps"],
        }
    )
    t = lb.truth_labels(df, CFG["schema"], CFG["hm_eval"])
    assert t["category"].tolist() == ["Bottomwear", "Topwear", "Other"]
    assert t["subcategory"].tolist() == ["Trousers", "T-shirt", "Other"]
    assert t["primary_colour"].tolist() == ["Black", "Unknown", "Pink"]
    assert t["pattern"].tolist() == ["Solid", "Printed", "Textured"]


def test_labels_stay_inside_schema():
    df = pd.DataFrame(
        {
            "product_group_name": ["Shoes", "Bags", "Nightwear"],
            "product_type_name": ["Sneakers", "Bag", "Pyjama set"],
            "colour_group_name": ["Other Orange", "Silver", "Dark Green"],
            "graphical_appearance_name": ["Dot", "Sequin", "Treatment"],
        }
    )
    t = lb.truth_labels(df, CFG["schema"], CFG["hm_eval"])
    for f in lb.FIELDS:
        assert set(t[f]) <= set(CFG["schema"][f]), f


def test_article_text_joins_and_skips_missing():
    df = pd.DataFrame(
        {"prod_name": ["Jelly tee", "Honey"], "detail_desc": ["Soft jersey T-shirt.", None]}
    )
    assert lb.article_text(df, ["prod_name", "detail_desc"]).tolist() == [
        "Jelly tee. Soft jersey T-shirt.",
        "Honey",
    ]


def test_group_split_keeps_styles_together_and_is_deterministic():
    groups = pd.Series([f"s{i // 3}" for i in range(3000)])  # 1000 styles x 3 variants
    a = lb.group_split(groups, 0.2, 0.1, seed=42)
    b = lb.group_split(groups, 0.2, 0.1, seed=42)
    assert a.equals(b)
    assert (pd.DataFrame({"g": groups, "s": a}).groupby("g")["s"].nunique() == 1).all()
    shares = a.value_counts(normalize=True)
    assert 0.15 < shares["test"] < 0.25 and 0.05 < shares["val"] < 0.11
    assert not a.equals(lb.group_split(groups, 0.2, 0.1, seed=7))


def _toy_frame(n_styles: int = 240, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    kinds = [
        ("trousers", "Bottomwear", "Trousers"),
        ("tee", "Topwear", "T-shirt"),
        ("dress", "Full-body garment", "Dress"),
    ]
    rows = []
    for s in range(n_styles):
        word, cat, sub = kinds[s % 3]
        pattern = "Striped" if rng.random() < 0.3 else "Solid"
        desc = f"Soft {word} in cotton" + (" with stripes" if pattern == "Striped" else "")
        for v in range(2):
            rows.append(
                {
                    "id": f"{s}-{v}",
                    "product_code": f"s{s}",
                    "title": f"Item {word}",
                    "text": f"Item {word}. {desc}.",
                    "category": cat,
                    "subcategory": sub,
                    "primary_colour": "Unknown",
                    "pattern": pattern,
                }
            )
    df = pd.DataFrame(rows)
    df["split"] = lb.group_split(df["product_code"], 0.25, 0.15, seed=1)
    return df


def _cfg(models: list[str]) -> dict:
    cfg = copy.deepcopy(CFG)
    cfg["tfidf"].update({"min_df": 1, "max_features": 500})
    cfg["xgboost"].update({"n_estimators": 30})
    cfg["models"] = models
    return cfg


def test_train_all_logreg_learns_and_beats_baseline():
    df = _toy_frame()
    bundle, metrics, details = md.train_all(df, _cfg(["most_frequent", "logreg"]), seed=42)
    test = metrics[metrics.split == "test"].set_index(["field", "model"])["accuracy"]
    assert test[("subcategory", "logreg")] == pytest.approx(1.0)
    assert test[("subcategory", "logreg")] > test[("subcategory", "most_frequent")]
    assert details["pattern"]["chosen_model"] == "logreg"
    pred = bundle.predict(["Item dress. Soft dress in cotton with stripes."])
    assert pred.loc[0, "subcategory"] == "Dress" and pred.loc[0, "pattern"] == "Striped"
    assert 0 < pred.loc[0, "subcategory_confidence"] <= 1
    # Unknown-only field: baseline and model both predict the single class.
    assert details["primary_colour"]["test"]["accuracy"] == 1.0


@pytest.mark.skipif(not _xgboost_available(), reason="XGBoost / OpenMP runtime not installed")
def test_train_all_with_xgboost():
    df = _toy_frame()
    _, metrics, _ = md.train_all(df, _cfg(["most_frequent", "xgboost"]), seed=42)
    acc = metrics.query("field == 'subcategory' and model == 'xgboost' and split == 'test'")
    assert acc["accuracy"].item() > 0.9


def test_evaluate_threshold_table():
    y = pd.Series(["a", "a", "b", "b"])
    pred = np.array(["a", "b", "b", "b"])
    conf = np.array([0.95, 0.55, 0.9, 0.6])
    ev = md.evaluate(y, pred, conf, [0.5, 0.8])
    assert ev["accuracy"] == pytest.approx(0.75)
    t80 = ev["thresholds"][1]
    assert t80["coverage"] == pytest.approx(0.5) and t80["accuracy_above"] == pytest.approx(1.0)
    assert t80["accuracy_below"] == pytest.approx(0.5)


def test_auto_accept_summary():
    def preds(true, pred, conf):
        return {
            "predictions": pd.DataFrame(
                {"id": ["1", "2", "3"], "true": true, "pred": pred, "confidence": conf}
            )
        }

    details = {
        "a": preds(["x", "x", "y"], ["x", "x", "x"], [0.95, 0.95, 0.4]),
        "b": preds(["p", "q", "q"], ["p", "p", "q"], [0.99, 0.92, 0.99]),
    }
    s = md.auto_accept_summary(details, ["a", "b"], 0.9)
    assert s["auto_accept_share"] == pytest.approx(2 / 3)
    assert s["all_fields_correct_when_auto_accepted"] == pytest.approx(0.5)
    assert s["all_fields_correct_overall"] == pytest.approx(1 / 3)


def test_match_option():
    opts = ["Topwear", "Bottomwear", "Footwear", "T-shirt", "Top"]
    assert match_option("2", opts) == "Bottomwear"
    assert match_option("foot", opts) == "Footwear"
    assert match_option("top", opts) == "Top"
    assert match_option("t", opts) is None
    assert match_option("99", opts) is None


def test_effort_summary():
    fields = ["category", "pattern"]
    items = pd.DataFrame(
        {
            "id": ["m1", "r1", "r2"],
            "arm": ["manual", "review", "review"],
            "category": ["Topwear"] * 3,
            "pattern": ["Solid"] * 3,
            "sug_category": [None, "Topwear", "Topwear"],
            "sug_pattern": [None, "Solid", "Striped"],
        }
    )
    manual = pd.DataFrame(
        {"id": ["m1"], "seconds": [40.0], "category": ["Topwear"], "pattern": ["Solid"]}
    )
    review = pd.DataFrame(
        {
            "id": ["r1", "r2"],
            "seconds": [8.0, 12.0],
            "category": ["Topwear", "Topwear"],
            "pattern": ["Solid", "Solid"],
        }
    )
    s = summarize(items, manual, review, fields)
    assert s["effort_reduction"] == pytest.approx(1 - 10 / 40)
    assert s["review"]["accuracy_vs_labels"]["pattern"] == 1.0
    assert s["suggestions_before_review"]["accuracy_vs_labels"]["pattern"] == 0.5
    assert s["review"]["fields_changed_share"] == pytest.approx(1 / 4)
