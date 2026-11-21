"""Pricing features: discount bands (Phase 2); reference price and discount % follow in Phase 3."""

from __future__ import annotations


def band_case_sql(column: str, edges: list[float], labels: list[str]) -> str:
    """SQL CASE expression mapping a discount fraction in `column` to its band label.

    `edges` are ascending band boundaries (e.g. [0, 0.05, ..., 0.9]); band i covers
    [edges[i], edges[i+1]). Values at or above the last inner edge fall in the last band,
    and values below the first edge fall in the first band.
    """
    if len(labels) != len(edges) - 1:
        raise ValueError("need exactly one label per band (len(edges) - 1)")
    if list(edges) != sorted(edges):
        raise ValueError("band edges must be ascending")
    parts = [
        f"WHEN {column} < {float(upper)} THEN '{label}'"
        for upper, label in zip(edges[1:-1], labels[:-1], strict=True)
    ]
    return f"CASE {' '.join(parts)} ELSE '{labels[-1]}' END"
