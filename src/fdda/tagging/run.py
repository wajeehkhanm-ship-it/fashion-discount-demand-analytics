"""Phase 6 driver: train/evaluate the tagger, tag Myntra, prepare the effort study.

    python -m fdda.tagging.run train        # train on H&M, write metrics to reports/tagging/
    python -m fdda.tagging.run myntra       # tag a Myntra sample with the saved models
    python -m fdda.tagging.run study-prep   # pick effort-study products, add model suggestions
    python -m fdda.tagging.run all

Models and product-level tags are data (git-ignored, under data/processed/tagging/); only
metrics, distributions and counts go to reports/tagging/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import duckdb
import joblib
import pandas as pd

from fdda.analysis import eda
from fdda.config import load_settings, resolve_path
from fdda.log import get_logger
from fdda.tagging import labels as lb
from fdda.tagging import model as md

log = get_logger(__name__)
MODE = "full"  # the article catalogue is the same in dev and full mode


class Paths:
    """Where Phase 6 reads and writes."""

    def __init__(self, settings: dict[str, Any]):
        self.interim = resolve_path(settings, "interim") / MODE
        self.data = resolve_path(settings, "processed") / "tagging"
        self.reports = resolve_path(settings, "reports") / "tagging"
        self.bundle = self.data / "tagger.joblib"
        self.data.mkdir(parents=True, exist_ok=True)
        self.reports.mkdir(parents=True, exist_ok=True)


def dataset(settings: dict[str, Any], con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """H&M articles with text, labels and a style-level train/val/test split."""
    cfg = settings["tagging"]
    df = lb.build_dataset(con, eda.hm_articles(Paths(settings).interim), cfg)
    df["split"] = lb.group_split(
        df[cfg["group_column"]], cfg["test_size"], cfg["val_size"], settings["project"]["seed"]
    )
    return df


def cmd_train(settings: dict[str, Any], con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """Train every model kind per field, keep the best, write metrics and the model bundle."""
    t0 = time.perf_counter()
    p, cfg = Paths(settings), settings["tagging"]
    df = dataset(settings, con)
    log.info("articles: %s", df["split"].value_counts().to_dict())
    bundle, metrics, details = md.train_all(df, cfg, settings["project"]["seed"])
    joblib.dump(bundle, p.bundle)
    df[["id", "split"]].to_parquet(p.data / "splits.parquet")

    metrics.to_csv(p.reports / "metrics.csv", index=False)
    for f, d in details.items():
        pd.DataFrame(d["test"]["thresholds"]).to_csv(p.reports / f"confidence_{f}.csv", index=False)
        d["per_class"].to_csv(p.reports / f"per_class_{f}.csv", index=False)
        preds = d["predictions"]
        top = preds["true"].value_counts().head(15).index
        cm = pd.crosstab(preds["true"], preds["pred"]).reindex(index=top).fillna(0).astype(int)
        cols = [c for c in top if c in cm.columns] + [c for c in cm.columns if c not in top]
        cm[cols].to_csv(p.reports / f"confusion_{f}.csv")
    thr = cfg["auto_accept_threshold"]
    auto = {
        "combined": md.auto_accept_summary(details, cfg["auto_accept_fields"], thr),
        "per_field": {
            f: next(r for r in d["test"]["thresholds"] if r["threshold"] == thr)
            for f, d in details.items()
        },
    }
    summary = {
        "articles": int(len(df)),
        "styles": int(df[cfg["group_column"]].nunique()),
        "split_articles": df["split"].value_counts().to_dict(),
        "features": int(len(bundle.vectorizer.vocabulary_)),
        "fields": {
            f: {
                "chosen_model": d["chosen_model"],
                "test_accuracy": d["test"]["accuracy"],
                "test_macro_f1": d["test"]["macro_f1"],
                "test_accuracy_title_only": d["test_title_only"]["accuracy"],
                "baseline_test_accuracy": float(
                    metrics.query("field == @f and model == 'most_frequent' and split == 'test'")[
                        "accuracy"
                    ].iloc[0]
                ),
                "classes": int(len(bundle.taggers[f].encoder.classes_)),
            }
            for f, d in details.items()
        },
        "auto_accept": auto,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    (p.reports / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "split_articles"}, indent=2))
    return summary


def _bundle(settings: dict[str, Any]) -> md.TaggerBundle:
    path = Paths(settings).bundle
    if not path.exists():
        raise FileNotFoundError("No trained tagger yet: run `python -m fdda.tagging.run train`.")
    return joblib.load(path)


def cmd_myntra(settings: dict[str, Any], con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """Tag a deterministic Myntra sample (titles only) and summarise the suggestions."""
    p, cfg = Paths(settings), settings["tagging"]
    rel = eda.myntra_with_category(
        eda.myntra_products(p.interim), settings["eda"]["myntra_category_rules"]
    )
    items = con.execute(f"""
        SELECT CAST(id AS VARCHAR) AS id, name AS title, seller AS brand, price, mrp,
               discount_pct, rating_total, category AS keyword_category
        FROM {rel}
        ORDER BY md5(CAST(id AS VARCHAR) || ':{int(settings["project"]["seed"])}')
        LIMIT {int(cfg["myntra_sample_size"])}
    """).df()
    tags = _bundle(settings).predict(items["title"].fillna(""))
    out = pd.concat([items.reset_index(drop=True), tags], axis=1)
    out.to_parquet(p.data / "myntra_tags.parquet")
    dist = pd.concat(
        [
            out[f]
            .value_counts(normalize=True)
            .rename("share")
            .rename_axis("value")
            .reset_index()
            .assign(field=f)
            for f in cfg["fields"]
        ],
        ignore_index=True,
    )[["field", "value", "share"]]
    dist.to_csv(p.reports / "myntra_tag_distribution.csv", index=False)
    pd.crosstab(out["keyword_category"], out["category"]).to_csv(
        p.reports / "myntra_keyword_vs_model_category.csv"
    )
    thr = cfg["auto_accept_threshold"]
    summary = {
        "items": int(len(out)),
        "median_confidence": {f: float(out[f"{f}_confidence"].median()) for f in cfg["fields"]},
        f"share_confidence_at_least_{thr}": {
            f: float((out[f"{f}_confidence"] >= thr).mean()) for f in cfg["fields"]
        },
    }
    (p.reports / "myntra_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def cmd_study_prep(settings: dict[str, Any], con: duckdb.DuckDBPyConnection) -> Path:
    """Pick 2 x n test-set products (unseen in training) and add model suggestions."""
    p, cfg = Paths(settings), settings["tagging"]
    n = cfg["effort_study"]["n_items"]
    df = dataset(settings, con)
    test = df[df["split"] == "test"].drop_duplicates(cfg["group_column"])  # one variant per style
    seed = settings["project"]["seed"]
    key = test["id"].map(lambda x: hashlib.md5(f"{x}:{seed}:study".encode()).hexdigest())
    items = test.assign(_k=key).sort_values("_k").head(2 * n).drop(columns="_k")
    items = items.reset_index(drop=True)
    items["arm"] = ["manual"] * n + ["review"] * n
    sugg = _bundle(settings).predict(items["text"])
    for f in cfg["fields"]:
        items[f"sug_{f}"] = sugg[f].where(items["arm"] == "review")
    items = items.rename(columns={"prod_name": "title_raw", "detail_desc": "description"})
    path = p.data / "effort_study_items.parquet"
    items.to_parquet(path)
    print(f"Wrote {len(items)} study items to {path}")
    return path


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Product tagger (TF-IDF + XGBoost)")
    parser.add_argument("command", choices=["train", "myntra", "study-prep", "all"])
    args = parser.parse_args(argv)
    settings = load_settings()
    con = duckdb.connect()
    if args.command in ("train", "all"):
        cmd_train(settings, con)
    if args.command in ("myntra", "all"):
        cmd_myntra(settings, con)
    if args.command in ("study-prep", "all"):
        cmd_study_prep(settings, con)


if __name__ == "__main__":
    main()
