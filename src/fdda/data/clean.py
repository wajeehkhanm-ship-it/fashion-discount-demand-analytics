"""Clean raw H&M and Myntra data with DuckDB and write Parquet to `data/interim/<mode>/`.

H&M cleaning rules (applied in this order, each with a logged row count):
  1. drop rows with a missing or unparseable date, customer, article or price
  2. drop rows with price <= 0
  3. drop rows whose article_id is not in articles.csv (after left-padding ids to 10 digits)
  4. dev mode only: keep a deterministic sample of customers (all rows of a sampled customer)
  Exact duplicate rows are KEPT: in H&M each row is one unit, so two identical rows
  mean two units of the same article in the same purchase. Prices more than
  `price_outlier_multiple` x the article median are flagged (`is_price_outlier`), not dropped.

Myntra cleaning rules:
  1. drop rows with a missing id or name
  2. drop duplicate ids (keep the first)
  3. drop rows with a missing or non-positive price
  4. drop rows with a missing/non-positive MRP or MRP < price
  5. dev mode only: keep a deterministic sample of listings
  discount_pct = 1 - price / mrp, clipped to [0, max_discount].

The full H&M CSV is never loaded into pandas; everything runs in DuckDB.

Usage: `python -m fdda.data.clean [--dataset hm|myntra|all] [--dev | --full]`
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import duckdb

from fdda.config import data_mode, load_settings, resolve_path
from fdda.log import get_logger

log = get_logger(__name__)

ARTICLE_COLUMNS = [
    "prod_name",
    "product_type_name",
    "product_group_name",
    "graphical_appearance_name",
    "colour_group_name",
    "section_name",
    "garment_group_name",
    "detail_desc",
]
HIERARCHY_COLUMNS = [
    "product_group_name",
    "product_type_name",
    "section_name",
    "garment_group_name",
]


def _q(path: Path | str) -> str:
    """Quote a filesystem path as a SQL string literal."""
    return "'" + str(path).replace("'", "''") + "'"


def sample_predicate(column: str, fraction: float, seed: int) -> str:
    """SQL predicate keeping a deterministic ``fraction`` of distinct values of ``column``.

    Uses md5 (stable across DuckDB versions and machines) of value + seed.
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"sample fraction must be in (0, 1], got {fraction}")
    threshold = round(fraction * 10_000)
    return f"(md5_number(CAST({column} AS VARCHAR) || ':{int(seed)}') % 10000) < {threshold}"


def connect(settings: dict[str, Any] | None = None) -> duckdb.DuckDBPyConnection:
    """Open an in-memory DuckDB connection with the configured memory limit and spill dir."""
    con = duckdb.connect()
    if settings:
        cfg = settings.get("duckdb", {})
        if cfg.get("memory_limit"):
            con.execute(f"SET memory_limit = {_q(cfg['memory_limit'])}")
        if cfg.get("temp_directory"):
            tmp = Path(settings["_root"]) / cfg["temp_directory"]
            tmp.mkdir(parents=True, exist_ok=True)
            con.execute(f"SET temp_directory = {_q(tmp)}")
    return con


def _numeric(expr: str) -> str:
    """SQL turning text like 'Rs. 1,299' into 1299.0 (NULL when no number is present)."""
    return f"TRY_CAST(regexp_extract(replace({expr}, ',', ''), '[0-9]+(\\.[0-9]+)?', 0) AS DOUBLE)"


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


# --------------------------------------------------------------------------- H&M


def clean_hm(
    raw_dir: Path,
    out_dir: Path,
    *,
    dev_mode: bool,
    sample_fraction: float,
    seed: int,
    price_rescale_factor: float,
    price_outlier_multiple: float,
    con: duckdb.DuckDBPyConnection | None = None,
) -> dict[str, Any]:
    """Clean H&M transactions + articles (+ customers) and write Parquet.

    Writes `out_dir/transactions/year_month=YYYY-MM/*.parquet`, `out_dir/articles.parquet`
    and `out_dir/customers.parquet`. Returns a summary dict of row counts and date ranges.
    """
    t0 = time.perf_counter()
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    tx_csv, art_csv = raw_dir / "transactions_train.csv", raw_dir / "articles.csv"
    cust_csv = raw_dir / "customers.csv"
    for path in (tx_csv, art_csv):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run `make download` first.")
    con = con or connect()
    _reset_dir(out_dir)
    staging = out_dir / "_staging_transactions.parquet"

    # Articles: one row per 10-digit article_id.
    art_cols = ", ".join(ARTICLE_COLUMNS)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE articles AS
        SELECT lpad(trim(article_id), 10, '0') AS article_id,
               lpad(trim(product_code), 7, '0') AS product_code,
               {art_cols}
        FROM read_csv({_q(art_csv)}, header = true, all_varchar = true)
        WHERE nullif(trim(article_id), '') IS NOT NULL
        QUALIFY row_number() OVER (PARTITION BY lpad(trim(article_id), 10, '0')) = 1
    """)

    # One pass over the big CSV -> typed staging Parquet; later steps read the Parquet.
    log.info("Staging %s -> Parquet ...", tx_csv.name)
    con.execute(f"""
        COPY (
            SELECT TRY_CAST(t_dat AS DATE)                      AS t_dat,
                   nullif(trim(customer_id), '')                AS customer_id,
                   lpad(nullif(trim(article_id), ''), 10, '0')  AS article_id,
                   TRY_CAST(price AS DOUBLE)                    AS price,
                   TRY_CAST(sales_channel_id AS INTEGER)        AS sales_channel_id
            FROM read_csv({_q(tx_csv)}, header = true, all_varchar = true)
        ) TO {_q(staging)} (FORMAT parquet)
    """)

    in_sample = sample_predicate("s.customer_id", sample_fraction, seed) if dev_mode else "TRUE"
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW tx_flagged AS
        SELECT s.*,
               (s.t_dat IS NOT NULL AND s.customer_id IS NOT NULL
                AND s.article_id IS NOT NULL AND s.price IS NOT NULL) AS is_complete,
               a.article_id IS NOT NULL                                AS is_known_article,
               coalesce({in_sample}, FALSE)                            AS is_in_sample
        FROM read_parquet({_q(staging)}) s
        LEFT JOIN articles a ON s.article_id = a.article_id
    """)
    raw_rows, d_missing, d_price, d_article, d_sample = con.execute("""
        SELECT count(*),
               count(*) FILTER (WHERE NOT is_complete),
               count(*) FILTER (WHERE is_complete AND price <= 0),
               count(*) FILTER (WHERE is_complete AND price > 0 AND NOT is_known_article),
               count(*) FILTER (WHERE is_complete AND price > 0 AND is_known_article
                                AND NOT is_in_sample)
        FROM tx_flagged
    """).fetchone()

    hierarchy = ", ".join(f"a.{c}" for c in HIERARCHY_COLUMNS)
    tx_dir = out_dir / "transactions"
    con.execute(f"""
        COPY (
            WITH kept AS (
                SELECT * FROM tx_flagged
                WHERE is_complete AND price > 0 AND is_known_article AND is_in_sample
            ),
            med AS (SELECT article_id, median(price) AS median_price FROM kept GROUP BY article_id)
            SELECT k.t_dat,
                   strftime(k.t_dat, '%Y-%m')                     AS year_month,
                   k.customer_id,
                   k.article_id,
                   a.product_code,
                   {hierarchy},
                   k.sales_channel_id,
                   k.price,
                   k.price * {float(price_rescale_factor)}        AS indexed_price,
                   k.price > {float(price_outlier_multiple)} * m.median_price AS is_price_outlier
            FROM kept k
            JOIN articles a ON k.article_id = a.article_id
            JOIN med m ON k.article_id = m.article_id
        ) TO {_q(tx_dir)} (FORMAT parquet, PARTITION_BY (year_month))
    """)
    staging.unlink()

    con.execute(f"COPY articles TO {_q(out_dir / 'articles.parquet')} (FORMAT parquet)")

    tx = f"read_parquet({_q(tx_dir / '*' / '*.parquet')}, hive_partitioning = true)"
    if cust_csv.exists():
        con.execute(f"""
            COPY (
                SELECT c.* FROM read_csv({_q(cust_csv)}, header = true, all_varchar = true) c
                WHERE c.customer_id IN (SELECT DISTINCT customer_id FROM {tx})
            ) TO {_q(out_dir / "customers.parquet")} (FORMAT parquet)
        """)
    else:
        log.warning("customers.csv not found; skipping customers.parquet")

    final_rows, n_cust, n_art, d_min, d_max, n_outliers, dup_rows = con.execute(f"""
        SELECT count(*), count(DISTINCT customer_id), count(DISTINCT article_id),
               min(t_dat), max(t_dat), count(*) FILTER (WHERE is_price_outlier),
               count(*) - count(DISTINCT (t_dat, customer_id, article_id, price, sales_channel_id))
        FROM {tx}
    """).fetchone()
    by_month = dict(
        con.execute(f"SELECT year_month, count(*) FROM {tx} GROUP BY 1 ORDER BY 1").fetchall()
    )

    summary: dict[str, Any] = {
        "dataset": "hm",
        "mode": "dev" if dev_mode else "full",
        "dev_sample_fraction": sample_fraction if dev_mode else None,
        "seed": seed,
        "price_rescale_factor": price_rescale_factor,
        "raw_rows": raw_rows,
        "dropped_missing_or_unparseable": d_missing,
        "dropped_nonpositive_price": d_price,
        "dropped_unknown_article": d_article,
        "dropped_dev_sample": d_sample,
        "final_rows": final_rows,
        "duplicate_rows_kept_as_units": dup_rows,
        "price_outlier_rows_flagged": n_outliers,
        "n_customers": n_cust,
        "n_articles_sold": n_art,
        "n_articles_catalog": con.execute("SELECT count(*) FROM articles").fetchone()[0],
        "date_min": str(d_min),
        "date_max": str(d_max),
        "rows_by_month": by_month,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    _log_summary(summary)
    return summary


# ------------------------------------------------------------------------ Myntra


def clean_myntra(
    raw_dir: Path,
    out_dir: Path,
    *,
    dev_mode: bool,
    sample_fraction: float,
    seed: int,
    max_discount: float,
    columns: dict[str, str],
    con: duckdb.DuckDBPyConnection | None = None,
) -> dict[str, Any]:
    """Clean the Myntra listings CSV and write `out_dir/products.parquet`.

    `columns` maps our names (id, name, price, mrp, rating, rating_total, seller) to the
    source column names; matching is case-insensitive. Returns a summary dict.
    """
    t0 = time.perf_counter()
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    csvs = sorted(raw_dir.glob("*.csv"), key=lambda p: p.stat().st_size, reverse=True)
    if not csvs:
        raise FileNotFoundError(f"No CSV in {raw_dir}. Run `make download` first.")
    src = csvs[0]
    con = con or connect()
    _reset_dir(out_dir)

    reader = f"read_csv({_q(src)}, header = true, all_varchar = true)"
    source_cols = {
        r[0].lower(): r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {reader}").fetchall()
    }

    def col(key: str, required: bool = True) -> str:
        actual = source_cols.get(columns[key].lower())
        if actual is None:
            if required:
                raise KeyError(f"Myntra column '{columns[key]}' not found in {src.name}")
            return "NULL"
        return '"' + actual.replace('"', '""') + '"'

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE myntra_raw AS
        SELECT row_number() OVER ()                         AS src_row,
               nullif(trim({col("id")}), '')                AS id,
               nullif(trim({col("name")}), '')              AS name,
               nullif(trim({col("seller", False)}), '')     AS seller,
               {_numeric(col("price"))}                     AS price,
               {_numeric(col("mrp"))}                       AS mrp,
               {_numeric(col("rating", False))}             AS rating,
               CAST({_numeric(col("rating_total", False))} AS BIGINT) AS rating_total
        FROM {reader}
    """)
    raw_rows = con.execute("SELECT count(*) FROM myntra_raw").fetchone()[0]

    steps = [
        ("dropped_missing_id_or_name", "id IS NOT NULL AND name IS NOT NULL"),
        ("dropped_duplicate_id", None),  # handled with QUALIFY below
        ("dropped_invalid_price", "price IS NOT NULL AND price > 0"),
        ("dropped_invalid_mrp", "mrp IS NOT NULL AND mrp > 0 AND mrp >= price"),
    ]
    if dev_mode:
        steps.append(("dropped_dev_sample", sample_predicate("id", sample_fraction, seed)))

    summary: dict[str, Any] = {
        "dataset": "myntra",
        "mode": "dev" if dev_mode else "full",
        "dev_sample_fraction": sample_fraction if dev_mode else None,
        "seed": seed,
        "source_file": src.name,
        "raw_rows": raw_rows,
    }
    current, n_before = "myntra_raw", raw_rows
    for i, (name, predicate) in enumerate(steps):
        nxt = f"myntra_step{i}"
        if predicate is None:
            sql = (
                f"SELECT * FROM {current} "
                "QUALIFY row_number() OVER (PARTITION BY id ORDER BY src_row) = 1"
            )
        else:
            sql = f"SELECT * FROM {current} WHERE {predicate}"
        con.execute(f"CREATE OR REPLACE TEMP TABLE {nxt} AS {sql}")
        n_after = con.execute(f"SELECT count(*) FROM {nxt}").fetchone()[0]
        summary[name] = n_before - n_after
        current, n_before = nxt, n_after

    max_d = float(max_discount)
    con.execute(f"""
        COPY (
            SELECT id, name, seller, price, mrp,
                   least(greatest(1 - price / mrp, 0), {max_d}) AS discount_pct,
                   rating, rating_total
            FROM {current} ORDER BY src_row
        ) TO {_q(out_dir / "products.parquet")} (FORMAT parquet)
    """)
    summary["clipped_discount_rows"] = con.execute(
        f"SELECT count(*) FROM {current} WHERE 1 - price / mrp > {max_d}"
    ).fetchone()[0]
    summary["final_rows"] = n_before
    summary["n_sellers"] = con.execute(f"SELECT count(DISTINCT seller) FROM {current}").fetchone()[
        0
    ]
    summary["seconds"] = round(time.perf_counter() - t0, 1)
    _log_summary(summary)
    return summary


# --------------------------------------------------------------------------- CLI


def _log_summary(summary: dict[str, Any]) -> None:
    for key, value in summary.items():
        if key != "rows_by_month":
            log.info("[%s/%s] %s = %s", summary["dataset"], summary["mode"], key, value)


def write_summary(summary: dict[str, Any], path: Path) -> None:
    """Write a cleaning summary as pretty JSON (a pipeline output used by later reports)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    log.info("Wrote %s", path)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Clean raw data -> Parquet")
    parser.add_argument("--dataset", choices=["hm", "myntra", "all"], default="all")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dev", dest="dev_mode", action="store_true", default=None)
    mode.add_argument("--full", dest="dev_mode", action="store_false")
    args = parser.parse_args(argv)

    settings = load_settings()
    dev_mode = (data_mode(settings) == "dev") if args.dev_mode is None else args.dev_mode
    mode_name = "dev" if dev_mode else "full"
    raw = resolve_path(settings, "raw")
    interim = resolve_path(settings, "interim") / mode_name
    reports = resolve_path(settings, "reports")
    seed = settings["project"]["seed"]
    fraction = settings["dev_sample_fraction"]
    cleaning = settings["cleaning"]
    con = connect(settings)

    if args.dataset in ("hm", "all"):
        summary = clean_hm(
            raw / "hm",
            interim / "hm",
            dev_mode=dev_mode,
            sample_fraction=fraction,
            seed=seed,
            price_rescale_factor=cleaning["price_rescale_factor"],
            price_outlier_multiple=cleaning["price_outlier_multiple"],
            con=con,
        )
        write_summary(summary, reports / f"data_summary_hm_{mode_name}.json")
    if args.dataset in ("myntra", "all"):
        summary = clean_myntra(
            raw / "myntra",
            interim / "myntra",
            dev_mode=dev_mode,
            sample_fraction=fraction,
            seed=seed,
            max_discount=cleaning["max_discount"],
            columns=settings["datasets"]["myntra"]["columns"],
            con=con,
        )
        write_summary(summary, reports / f"data_summary_myntra_{mode_name}.json")


if __name__ == "__main__":
    main()
