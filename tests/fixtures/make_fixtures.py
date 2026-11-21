"""Generate the small synthetic fixtures used by the tests (deterministic, no real data).

Run `python tests/fixtures/make_fixtures.py` to regenerate. The planted edge cases and
their counts are listed in EXPECTED so tests can assert exact numbers.
"""

from __future__ import annotations

import csv
import hashlib
import random
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).parent
SEED = 42

PRODUCT_TYPES = [
    ("T-shirt", "Garment Upper body", "Jersey Basic", "Womens Everyday Basics"),
    ("Sweater", "Garment Upper body", "Knitwear", "Womens Tailoring"),
    ("Trousers", "Garment Lower body", "Trousers", "Mens Trousers"),
    ("Dress", "Garment Full body", "Dresses Ladies", "Womens Casual"),
    ("Bag", "Accessories", "Accessories", "Womens Big accessories"),
    ("Sneakers", "Shoes", "Shoes", "Womens Shoes"),
]
COLOURS = ["Black", "White", "Dark Blue", "Light Pink", "Grey"]
APPEARANCES = ["Solid", "Stripe", "All over pattern", "Melange"]
BRANDS = ["Roadster", "HRX", "Mango", "Libas", "H&M", "Puma"]

N_ARTICLES = 40
N_CUSTOMERS = 200
START, END = date(2018, 9, 20), date(2019, 3, 31)

# Planted edge cases (H&M transactions) -> count.
HM_EDGE = {
    "zero_price": 5,
    "negative_price": 3,
    "missing_customer": 4,
    "bad_date": 2,
    "unknown_article": 4,
    "exact_duplicate": 6,
    "unpadded_article_id": 10,
    "outlier_price": 2,
}
HM_TOTAL_ROWS = 1000
# Planted edge cases (Myntra) -> count.
MY_EDGE = {
    "missing_name": 2,
    "duplicate_id": 5,
    "zero_price": 3,
    "missing_mrp": 3,
    "mrp_below_price": 2,
    "extreme_discount": 2,
    "formatted_price": 3,
}
MY_TOTAL_ROWS = 200

EXPECTED = {
    "hm_raw_rows": HM_TOTAL_ROWS,
    "hm_dropped_missing": HM_EDGE["missing_customer"] + HM_EDGE["bad_date"],
    "hm_dropped_price": HM_EDGE["zero_price"] + HM_EDGE["negative_price"],
    "hm_dropped_article": HM_EDGE["unknown_article"],
    "hm_duplicates": HM_EDGE["exact_duplicate"],
    "hm_outliers": HM_EDGE["outlier_price"],
    "my_raw_rows": MY_TOTAL_ROWS,
    "my_dropped_missing": MY_EDGE["missing_name"],
    "my_dropped_duplicate": MY_EDGE["duplicate_id"],
    "my_dropped_price": MY_EDGE["zero_price"],
    "my_dropped_mrp": MY_EDGE["missing_mrp"] + MY_EDGE["mrp_below_price"],
    "my_clipped": MY_EDGE["extreme_discount"],
}
EXPECTED["hm_final_rows"] = (
    HM_TOTAL_ROWS
    - EXPECTED["hm_dropped_missing"]
    - EXPECTED["hm_dropped_price"]
    - EXPECTED["hm_dropped_article"]
)
EXPECTED["my_final_rows"] = (
    MY_TOTAL_ROWS
    - EXPECTED["my_dropped_missing"]
    - EXPECTED["my_dropped_duplicate"]
    - EXPECTED["my_dropped_price"]
    - EXPECTED["my_dropped_mrp"]
)


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_articles(rng: random.Random) -> list[dict]:
    """40 articles: 20 styles (product_code) x 2 colour variants."""
    rows = []
    for i in range(N_ARTICLES):
        ptype, group, garment, section = PRODUCT_TYPES[i % len(PRODUCT_TYPES)]
        product_code = 500000 + i // 2
        colour, look = rng.choice(COLOURS), rng.choice(APPEARANCES)
        rows.append(
            {
                "article_id": f"0{product_code}{1 + i % 2:03d}",
                "product_code": f"0{product_code}",
                "prod_name": f"{look} {ptype} {i}",
                "product_type_name": ptype,
                "product_group_name": group,
                "graphical_appearance_name": look,
                "colour_group_name": colour,
                "section_name": section,
                "garment_group_name": garment,
                "detail_desc": f"{look} {ptype.lower()} in {colour.lower()} cotton blend.",
            }
        )
    return rows


def make_customers(rng: random.Random) -> list[dict]:
    """200 customers with 64-hex ids like the real data."""
    return [
        {
            "customer_id": hashlib.sha256(f"customer-{i}".encode()).hexdigest(),
            "age": str(rng.randint(18, 70)),
            "club_member_status": rng.choice(["ACTIVE", "ACTIVE", "PRE-CREATE"]),
        }
        for i in range(N_CUSTOMERS)
    ]


def make_transactions(
    rng: random.Random, articles: list[dict], customers: list[dict]
) -> list[dict]:
    """1,000 transaction rows: normal rows plus the planted edge cases in HM_EDGE."""
    base_price = {a["article_id"]: rng.uniform(0.01, 0.06) for a in articles}
    n_days = (END - START).days

    def normal_row() -> dict:
        art = rng.choice(articles)["article_id"]
        markdown = rng.choice([0, 0, 0, 0.1, 0.2, 0.3, 0.5])
        return {
            "t_dat": (START + timedelta(days=rng.randint(0, n_days))).isoformat(),
            "customer_id": rng.choice(customers)["customer_id"],
            "article_id": art,
            "price": f"{base_price[art] * (1 - markdown):.6f}",
            "sales_channel_id": str(rng.choice([1, 2])),
        }

    n_normal = HM_TOTAL_ROWS - sum(HM_EDGE.values())
    rows = [normal_row() for _ in range(n_normal)]
    edge: list[dict] = []
    edge += [{**normal_row(), "price": "0"} for _ in range(HM_EDGE["zero_price"])]
    edge += [{**normal_row(), "price": "-0.0150"} for _ in range(HM_EDGE["negative_price"])]
    edge += [{**normal_row(), "customer_id": ""} for _ in range(HM_EDGE["missing_customer"])]
    edge += [{**normal_row(), "t_dat": "2019-13-45"} for _ in range(HM_EDGE["bad_date"])]
    edge += [
        {**normal_row(), "article_id": "0999999001"} for _ in range(HM_EDGE["unknown_article"])
    ]
    edge += [dict(rows[i]) for i in range(HM_EDGE["exact_duplicate"])]
    for _ in range(HM_EDGE["unpadded_article_id"]):
        r = normal_row()
        edge.append({**r, "article_id": r["article_id"].lstrip("0")})
    outlier_art = articles[0]["article_id"]
    edge += [
        {**normal_row(), "article_id": outlier_art, "price": f"{base_price[outlier_art] * 10:.6f}"}
        for _ in range(HM_EDGE["outlier_price"])
    ]
    rows += edge
    rng.shuffle(rows)
    return rows


def make_myntra(rng: random.Random) -> list[dict]:
    """200 Myntra-style listings with the columns of ronakbokaria/myntra-products-dataset."""

    def listing(i: int) -> dict:
        mrp = rng.choice([499, 799, 999, 1299, 1999, 2499, 3999])
        disc = rng.choice([0, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        brand = rng.choice(BRANDS)
        return {
            "id": str(10_000_000 + i),
            "name": f"{brand} Women {rng.choice(PRODUCT_TYPES)[0]}",
            "img": f"https://example.invalid/img/{i}.jpg",
            "asin": f"A{i:07d}",
            "price": str(round(mrp * (1 - disc))),
            "mrp": str(mrp),
            "rating": f"{rng.uniform(3.0, 4.8):.1f}",
            "ratingTotal": str(rng.randint(0, 5000)) if rng.random() > 0.1 else "",
            "discount": f"{int(disc * 100)}" if disc else "",
            "seller": brand,
            "purl": f"https://example.invalid/p/{i}",
        }

    n_normal = MY_TOTAL_ROWS - sum(MY_EDGE.values())
    rows = [listing(i) for i in range(n_normal)]
    nxt = iter(range(n_normal, 10_000))
    rows += [{**listing(next(nxt)), "name": ""} for _ in range(MY_EDGE["missing_name"])]
    # Duplicates of the first rows with a different price; the first occurrence must win.
    rows += [{**rows[i], "price": "1"} for i in range(MY_EDGE["duplicate_id"])]
    rows += [{**listing(next(nxt)), "price": "0"} for _ in range(MY_EDGE["zero_price"])]
    rows += [{**listing(next(nxt)), "mrp": ""} for _ in range(MY_EDGE["missing_mrp"])]
    rows += [
        {**listing(next(nxt)), "price": "900", "mrp": "500"}
        for _ in range(MY_EDGE["mrp_below_price"])
    ]
    rows += [
        {**listing(next(nxt)), "price": "40", "mrp": "1000"}
        for _ in range(MY_EDGE["extreme_discount"])
    ]
    rows += [
        {**listing(next(nxt)), "price": "Rs. 1,299", "mrp": "Rs. 2,599"}
        for _ in range(MY_EDGE["formatted_price"])
    ]
    return rows


def main() -> None:
    """Write all fixture CSVs."""
    rng = random.Random(SEED)
    articles = make_articles(rng)
    customers = make_customers(rng)
    _write(HERE / "hm" / "articles.csv", articles)
    _write(HERE / "hm" / "customers.csv", customers)
    _write(HERE / "hm" / "transactions_train.csv", make_transactions(rng, articles, customers))
    _write(HERE / "myntra" / "myntra_products.csv", make_myntra(rng))


if __name__ == "__main__":
    main()
