"""Training labels for the product tagger, from H&M's own article metadata.

Targets per field:
* category       <- product_group_name (mapping in config)
* subcategory    <- product_type_name (types outside the schema become "Other")
* primary_colour <- colour_group_name, collapsed to base colours by keyword rules
* pattern        <- graphical_appearance_name (mapping in config; unmapped -> "Other")

Splits are made by style (`product_code`): colour variants of one style share almost the
same text, so splitting by article would leak near-duplicates into the test set.
"""

from __future__ import annotations

from typing import Any

import duckdb
import pandas as pd

UNKNOWN = "Unknown"
FIELDS = ["category", "subcategory", "primary_colour", "pattern"]


def map_colour(colour_group: str | None, rules: list[list[str]]) -> str:
    """H&M colour group -> base colour via ordered keyword rules; Unknown if none match."""
    if not isinstance(colour_group, str) or not colour_group:
        return UNKNOWN
    low = colour_group.lower()
    for keyword, base in rules:
        if keyword in low:
            return base
    return UNKNOWN


def truth_labels(
    df: pd.DataFrame, schema: dict[str, list[str]], cfg: dict[str, Any]
) -> pd.DataFrame:
    """Target labels for each field from H&M label columns (same row order as `df`)."""
    types = set(schema["subcategory"])
    groups = cfg["category_from_product_group"]
    patterns = cfg["pattern_from_appearance"]
    return pd.DataFrame(
        {
            "category": df["product_group_name"].map(groups).fillna("Other").to_numpy(),
            "subcategory": df["product_type_name"]
            .where(df["product_type_name"].isin(types), "Other")
            .to_numpy(),
            "primary_colour": [map_colour(c, cfg["colour_rules"]) for c in df["colour_group_name"]],
            "pattern": df["graphical_appearance_name"].map(patterns).fillna("Other").to_numpy(),
        },
        index=df.index,
    )


def article_text(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Model input text: the given columns joined with '. ' (missing parts skipped)."""
    parts = df[columns].fillna("").astype(str)
    return parts.apply(lambda r: ". ".join(p.strip() for p in r if p.strip()), axis=1)


def build_dataset(
    con: duckdb.DuckDBPyConnection, articles: str, cfg: dict[str, Any]
) -> pd.DataFrame:
    """One row per article: id, style, title, description, text, and target labels."""
    df = con.execute(f"""
        SELECT article_id AS id, product_code, prod_name, detail_desc,
               product_group_name, product_type_name, colour_group_name,
               graphical_appearance_name
        FROM {articles}
        WHERE prod_name IS NOT NULL
        ORDER BY article_id
    """).df()
    df["text"] = article_text(df, cfg["text_columns"])
    df["title"] = df["prod_name"]
    labels = truth_labels(df, cfg["schema"], cfg["hm_eval"])
    return pd.concat([df, labels], axis=1)


def group_split(groups: pd.Series, test_size: float, val_size: float, seed: int) -> pd.Series:
    """Deterministic train/val/test split by group (md5 of group + seed -> uniform [0, 1)).

    `val_size` is the share of the *non-test* groups held out for model selection.
    """
    import hashlib

    def u(g: str) -> float:
        h = hashlib.md5(f"{g}:{seed}".encode()).hexdigest()
        return int(h[:12], 16) / 16**12

    draws = groups.astype(str).map(u)
    split = pd.Series("train", index=groups.index)
    split[draws < test_size] = "test"
    val_cut = test_size + (1 - test_size) * val_size
    split[(draws >= test_size) & (draws < val_cut)] = "val"
    return split
