"""DuckDB persistence for receipt expenses."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
from dateutil import parser as dateparser

DB_PATH = str(Path(__file__).parent / "expenses.duckdb")

TABLE = "expenses"


def normalize_date(raw: str) -> str:
    """Normalize a receipt date string to ISO YYYY-MM-DD.

    Receipts use all sorts of formats (01/06/2024, June 1 2024, 2024.06.01).
    A leading 4-digit number is treated as year-first (unambiguous); otherwise
    we bias toward day-first, which fits most non-US receipts. Unparseable
    values are returned unchanged so nothing is silently lost.
    """
    if not raw or not str(raw).strip():
        return ""
    raw = str(raw).strip()
    year_first = raw[:4].isdigit()
    try:
        dt = dateparser.parse(raw, yearfirst=year_first, dayfirst=not year_first)
    except (ValueError, OverflowError, TypeError):
        return raw
    # Guard against OCR misreads of the year (e.g. '0016-06-26'). A date far
    # outside a plausible range almost certainly means a misread digit, so keep
    # the raw string flagged rather than emitting a wrong, confident-looking date.
    if not (2000 <= dt.year <= 2099):
        return raw
    return dt.strftime("%Y-%m-%d")


def _connect():
    return duckdb.connect(DB_PATH)


def init_db() -> None:
    """Create the expenses table if it doesn't exist."""
    with _connect() as con:
        con.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE} (
                id BIGINT,
                source_file VARCHAR,
                date VARCHAR,
                item VARCHAR,
                amount_purchased DOUBLE,
                price_per_item DOUBLE,
                total_price DOUBLE,
                category VARCHAR,
                inserted_at TIMESTAMP DEFAULT now()
            )
            """
        )
        # Sequence for stable ids across appends.
        con.execute("CREATE SEQUENCE IF NOT EXISTS expense_id_seq START 1")

    # Clean up any dates stored before normalization existed.
    normalize_existing_dates()


def normalize_existing_dates() -> int:
    """Re-normalize every stored date to YYYY-MM-DD.

    Rows written before date normalization was added may hold raw strings like
    '14-06-2026' or '16/06/2026'. This rewrites them in place. Returns the
    number of rows changed. Safe to run repeatedly (already-normal dates are
    left untouched).
    """
    with _connect() as con:
        try:
            dates = con.execute(
                f"SELECT DISTINCT date FROM {TABLE}"
            ).fetchdf()["date"].tolist()
        except duckdb.CatalogException:
            return 0

        changed = 0
        for raw in dates:
            norm = normalize_date(raw)
            if norm != raw:
                con.execute(
                    f"UPDATE {TABLE} SET date = ? WHERE date = ?",
                    [norm, raw],
                )
                changed += 1
        return changed


def append_rows(rows: list[dict]) -> int:
    """Append a batch of expense rows. Returns number of rows written."""
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    for col in ["source_file", "date", "item", "category"]:
        if col not in df:
            df[col] = ""
    for col in ["amount_purchased", "price_per_item", "total_price"]:
        if col not in df:
            df[col] = 0.0

    with _connect() as con:
        con.register("incoming", df)
        con.execute(
            f"""
            INSERT INTO {TABLE}
              (id, source_file, date, item, amount_purchased,
               price_per_item, total_price, category)
            SELECT
              nextval('expense_id_seq'),
              source_file, date, item, amount_purchased,
              price_per_item, total_price, category
            FROM incoming
            """
        )
    return len(df)


def load_all() -> pd.DataFrame:
    with _connect() as con:
        try:
            return con.execute(
                f"SELECT * FROM {TABLE} ORDER BY date, id"
            ).fetchdf()
        except duckdb.CatalogException:
            return pd.DataFrame()


def spend_by_date() -> pd.DataFrame:
    with _connect() as con:
        return con.execute(
            f"""
            SELECT
              COALESCE(CAST(TRY_CAST(date AS DATE) AS VARCHAR), date) AS day,
              SUM(total_price) AS total
            FROM {TABLE}
            GROUP BY 1
            ORDER BY day
            """
        ).fetchdf()


def spend_by_day_category() -> pd.DataFrame:
    """Spend per day broken down by category.

    Normalizes the stored date string to a day (date only, no time). If a value
    can't be parsed as a date it's kept as-is so nothing is silently dropped.
    """
    with _connect() as con:
        return con.execute(
            f"""
            SELECT
              COALESCE(CAST(TRY_CAST(date AS DATE) AS VARCHAR), date) AS day,
              COALESCE(NULLIF(TRIM(category), ''), 'Uncategorized') AS category,
              SUM(total_price) AS total
            FROM {TABLE}
            GROUP BY 1, 2
            ORDER BY day, category
            """
        ).fetchdf()


def spend_by_category() -> pd.DataFrame:
    with _connect() as con:
        return con.execute(
            f"""
            SELECT
              COALESCE(NULLIF(TRIM(category), ''), 'Uncategorized') AS category,
              SUM(total_price) AS total
            FROM {TABLE}
            GROUP BY 1
            ORDER BY total DESC
            """
        ).fetchdf()
