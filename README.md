# Ledger — Receipt Expense Tracker

A Dash app that turns receipt photos into a categorized expense ledger backed by
DuckDB, with a dashboard of charts.

## Pieces

```
main.py        # FastAPI + qwen2.5vl extraction API (from earlier)
db.py          # DuckDB persistence (init, append, aggregation queries)
app.py         # Dash UI: upload -> edit -> categorize -> submit -> visualize
expenses.duckdb  # created on first run
```

## Workflow

1. **Capture tab** — drop a receipt image. It's sent to the extraction API and
   the line items fill the table.
2. Type a **Category** into each row (free text — whatever scheme you like).
   You can also edit any extracted value, add rows, or delete rows.
3. **Submit to ledger** — the rows are appended to DuckDB and the table clears
   for the next receipt.
4. Repeat for each receipt; every submit *appends* (nothing is overwritten).
5. **Dashboard tab** —
   - a bar chart of total spend per receipt date, and
   - a donut chart of spend per category (the center names your biggest one).

## Run

Three things, in order:

```bash
# 1. Model
ollama pull qwen2.5vl:7b

# 2. Extraction API (terminal 1)
uvicorn main:app --reload          # http://127.0.0.1:8000

# 3. Dash app (terminal 2)
python app.py                      # http://127.0.0.1:8050
```

If your API runs elsewhere:

```bash
RECEIPT_API_URL=http://host:port python app.py
```

## Dependencies

```bash
pip install dash duckdb plotly requests pandas fastapi uvicorn ollama python-multipart
```

## Notes

- Line totals are recomputed (qty x unit price) by the API before they reach the
  table, so the math is trustworthy even if the model misreads a printed total.
- Empty categories roll up as **Uncategorized** in the donut.
- The DuckDB file is local and append-only via these flows; to inspect it:
  ```python
  import duckdb; duckdb.connect("expenses.duckdb").sql("SELECT * FROM expenses")
  ```
