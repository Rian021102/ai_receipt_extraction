"""DuckDB persistence for receipt expenses."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

DB_PATH = str(Path(__file__).parent / "expenses.duckdb")

TABLE = "expenses"


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
