"""Timed effort study: manual tagging vs reviewing model suggestions.

Run in a terminal, after `python -m fdda.tagging.run study-prep`:

    python -m fdda.tagging.effort_study manual   # tag 100 products by hand (timed)
    python -m fdda.tagging.effort_study review   # check / correct 100 suggested tag sets (timed)
    python -m fdda.tagging.effort_study report   # effort reduction = 1 - review time / manual time

Answer each field with its number or the start of its name; `?` lists the options again,
Enter accepts the suggested value (review only), `q` saves and quits (resume any time).
Both arms use different products from the tagger's held-out test styles.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from fdda.config import load_settings, resolve_path
from fdda.tagging.labels import UNKNOWN


class Quit(Exception):
    """User asked to stop."""


def match_option(text: str, options: list[str]) -> str | None:
    """Resolve user input to one option: number (1-based), exact name, or unique prefix."""
    text = text.strip()
    if text.isdigit():
        i = int(text)
        return options[i - 1] if 1 <= i <= len(options) else None
    low = text.lower()
    exact = [o for o in options if o.lower() == low]
    if exact:
        return exact[0]
    pref = [o for o in options if o.lower().startswith(low)]
    return pref[0] if len(pref) == 1 else None


def _print_options(options: list[str], cols: int = 3) -> None:
    width = max(len(o) for o in options) + 1
    for i in range(0, len(options), cols):
        row = range(i, min(i + cols, len(options)))
        print("".join(f"{j + 1:>3}. {options[j]:<{width}}" for j in row))


def ask(field: str, options: list[str], default: str | None = None) -> str:
    """Prompt until the user picks a valid option (or Enter for the default)."""
    _print_options(options)
    hint = f" [Enter = {default}]" if default else ""
    while True:
        raw = input(f"  {field}{hint}: ").strip()
        if raw.lower() == "q":
            raise Quit
        if raw == "?":
            _print_options(options)
            continue
        if raw == "" and default:
            return default
        choice = match_option(raw, options) if raw else None
        if choice:
            return choice
        print("    not recognised: type a number, the start of a name, ? or q")


def state_path(data_dir: Path, arm: str) -> Path:
    """File holding one JSON line per finished product for an arm."""
    return data_dir / f"effort_study_{arm}.jsonl"


def _done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["id"] for line in path.read_text().splitlines() if line.strip()}


def run_arm(
    arm: str, items: pd.DataFrame, schema: dict[str, list[str]], fields: list[str], data_dir: Path
) -> None:
    """Interactive loop for one arm; appends one JSON line per finished product."""
    path = state_path(data_dir, arm)
    done = _done_ids(path)
    todo = items[(items["arm"] == arm) & (~items["id"].isin(done))]
    total = int((items["arm"] == arm).sum())
    print(f"\n{arm.upper()} arm: {len(done)} of {total} done, {len(todo)} to go.")
    print("q = save and quit.\n")
    for _, row in todo.iterrows():
        print("=" * 78)
        print(f"[{len(done) + 1}/{total}]  {row['title']}")
        if isinstance(row.get("description"), str):
            print(f"  {row['description']}")
        start = time.perf_counter()
        answers: dict[str, str] = {}
        try:
            for f in fields:
                sug = row.get(f"sug_{f}") if arm == "review" else None
                answers[f] = ask(f, schema[f], sug if isinstance(sug, str) else None)
        except (Quit, EOFError, KeyboardInterrupt):
            print(f"\nSaved. {len(done)} of {total} done. Run the same command to continue.")
            return
        seconds = time.perf_counter() - start
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"id": row["id"], "seconds": seconds, **answers}) + "\n")
        done.add(row["id"])
    print(f"\n{arm} arm complete ({total} products).")


def summarize(
    items: pd.DataFrame, manual: pd.DataFrame, review: pd.DataFrame, fields: list[str]
) -> dict[str, Any]:
    """Timing, effort reduction and label accuracy of each arm and of the raw suggestions.

    `items` holds the true labels (one column per field) and `sug_<field>` suggestions.
    """
    truth = items[["id", *fields]]
    out: dict[str, Any] = {"fields": fields}
    for name, df in (("manual", manual), ("review", review)):
        out[name] = {
            "items": int(len(df)),
            "total_seconds": float(df["seconds"].sum()) if len(df) else 0.0,
            "seconds_per_item": float(df["seconds"].mean()) if len(df) else float("nan"),
        }
    m, r = out["manual"]["seconds_per_item"], out["review"]["seconds_per_item"]
    out["effort_reduction"] = float(1 - r / m) if (m == m and r == r and m > 0) else None

    def acc(df: pd.DataFrame) -> dict[str, float]:
        merged = truth.merge(df, on="id", suffixes=("_true", ""))
        res = {}
        for f in fields:
            valid = merged[f"{f}_true"] != UNKNOWN
            if valid.any():
                res[f] = float((merged.loc[valid, f] == merged.loc[valid, f"{f}_true"]).mean())
        return res

    out["manual"]["accuracy_vs_labels"] = acc(manual) if len(manual) else {}
    out["review"]["accuracy_vs_labels"] = acc(review) if len(review) else {}
    sugg = items[items["arm"] == "review"][["id", *[f"sug_{f}" for f in fields]]]
    sugg.columns = ["id", *fields]
    seen = sugg[sugg["id"].isin(review["id"])] if len(review) else sugg.iloc[0:0]
    out["suggestions_before_review"] = {"accuracy_vs_labels": acc(seen) if len(seen) else {}}
    if len(review):
        merged = seen.merge(review, on="id", suffixes=("_sug", ""))
        changed = sum((merged[f"{f}_sug"] != merged[f]).sum() for f in fields)
        out["review"]["fields_changed_share"] = float(changed / (len(merged) * len(fields)))
    return out


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Timed manual-vs-review tagging study")
    parser.add_argument("arm", choices=["manual", "review", "report"])
    args = parser.parse_args(argv)
    settings = load_settings()
    cfg = settings["tagging"]
    data_dir = resolve_path(settings, "processed") / "tagging"
    items_path = data_dir / "effort_study_items.parquet"
    if not items_path.exists():
        sys.exit("No study items yet: run `python -m fdda.tagging.run study-prep` first.")
    items = pd.read_parquet(items_path)
    fields = cfg["fields"]
    if args.arm in ("manual", "review"):
        run_arm(args.arm, items, cfg["schema"], fields, data_dir)
        return

    def load(arm: str) -> pd.DataFrame:
        p = state_path(data_dir, arm)
        if not p.exists():
            return pd.DataFrame(columns=["id", "seconds", *fields])
        return pd.read_json(p, lines=True, dtype={"id": str})

    result = summarize(items, load("manual"), load("review"), fields)
    out = resolve_path(settings, "reports") / "tagging" / "effort_study.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
