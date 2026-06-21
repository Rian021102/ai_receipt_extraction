"""Receipt expense tracker — Dash app (self-contained).

Workflow:
  1. Upload a receipt image.
  2. The vision model (qwen2.5vl:7b via Ollama) runs in-process; rows fill the table.
  3. Pick a Category for each row from the dropdown.
  4. Submit -> DuckDB appends the rows.
  5. Repeat for the next image.
  6. Dashboard tab: grouped daily bar chart (by category) + spend-by-category donut.

No separate API needed — extraction runs inside this app. Requires Ollama
running locally with the model pulled:  ollama pull qwen2.5vl:7b

Run:
    python app.py
Then open http://127.0.0.1:8050
"""

from __future__ import annotations

import base64
import json

import dash
import ollama
import plotly.graph_objects as go
from dateutil import parser as dateparser
from dash import Input, Output, State, callback, dash_table, dcc, html
from dash.exceptions import PreventUpdate

import db

db.init_db()

# ---------------------------------------------------------------------------
# Extraction (in-process, was previously the FastAPI /extract endpoint)
# ---------------------------------------------------------------------------
MODEL_NAME = "qwen2.5vl:latest"

EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "date": {"type": "string"},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "item": {"type": "string"},
                    "amount_purchased": {"type": "number"},
                    "price_per_item": {"type": "number"},
                    "total_price": {"type": "number"},
                },
                "required": ["item", "amount_purchased",
                             "price_per_item", "total_price"],
            },
        },
        "printed_total": {"type": "number"},
    },
    "required": ["date", "items"],
}

PROMPT = (
    "Extract from this receipt/image:\n"
    "- date: the single transaction date, applied to all items\n"
    "- for each item: item name, amount_purchased (quantity), "
    "price_per_item (unit price), total_price (amount_purchased x price_per_item)\n"
    "- printed_total: the grand total printed on the receipt, if shown\n"
    "Return only valid JSON matching the schema."
)


def normalize_date(raw: str) -> str:
    """Normalize a receipt date string to ISO YYYY-MM-DD.

    Receipts use all sorts of formats (01/06/2024, June 1 2024, 2024.06.01).
    dayfirst=True biases ambiguous cases like 03/04/2024 toward day/month,
    which fits most non-US receipts. If parsing fails, return the raw string
    unchanged so nothing is silently lost.
    """
    if not raw or not raw.strip():
        return ""
    raw = raw.strip()
    # A leading 4-digit number means a year-first format (2024-06-01), which is
    # unambiguous — parse it as-is. Otherwise the format is day/month-style and
    # we bias toward day-first (fits most non-US receipts).
    year_first = raw[:4].isdigit()
    try:
        dt = dateparser.parse(
            raw,
            yearfirst=year_first,
            dayfirst=not year_first,
        )
        return dt.strftime("%Y-%m-%d")
    except (ValueError, OverflowError, TypeError):
        return raw


def extract_receipt(image_b64: str) -> dict:
    """Run the vision model on a base64 image and return parsed data.

    Line totals are recomputed (qty x unit price) so the math is trustworthy
    even if the model misreads a printed figure.
    """
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": PROMPT, "images": [image_b64]}],
        format=EXTRACTION_SCHEMA,
        options={"temperature": 0},
    )
    data = json.loads(response["message"]["content"])

    items = []
    for raw in data.get("items", []):
        qty = float(raw.get("amount_purchased") or 0)
        unit = float(raw.get("price_per_item") or 0)
        items.append(
            {
                "item": raw.get("item", ""),
                "amount_purchased": qty,
                "price_per_item": unit,
                "total_price": round(qty * unit, 2),
            }
        )
    return {"date": normalize_date(data.get("date", "")), "items": items}

# ---------------------------------------------------------------------------
# Design tokens — "ledger" aesthetic: warm paper, ink, a single ledger-red.
# ---------------------------------------------------------------------------
INK = "#1c1a17"
PAPER = "#f6f2e9"
PANEL = "#fffdf7"
RULE = "#d8cfbc"
RED = "#9c2b1e"        # ledger red — debits
GREEN = "#3f6b4f"      # ledger green — confirmations
MUTED = "#6f685c"
ACCENT_SEQ = ["#9c2b1e", "#c97b3c", "#3f6b4f", "#5a7a8c", "#8a6d3b",
              "#7a4b63", "#456b6b", "#a8893f"]

CATEGORIES = ["Groceries", "Food", "Hobby", "Clothes", "Restaurant"]

TABLE_COLUMNS = [
    {"name": "Item", "id": "item", "editable": True},
    {"name": "Qty", "id": "amount_purchased", "type": "numeric", "editable": True},
    {"name": "Unit price", "id": "price_per_item", "type": "numeric", "editable": True},
    {"name": "Line total", "id": "total_price", "type": "numeric", "editable": True},
    {"name": "Category", "id": "category", "editable": True,
     "presentation": "dropdown"},
]

app = dash.Dash(__name__, title="Ledger — Receipt Expenses")
server = app.server

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def upload_tab():
    return html.Div(
        [
            html.Div(
                [
                    dcc.Upload(
                        id="upload-image",
                        children=html.Div(
                            [
                                html.Div("Drop a receipt here", className="up-title"),
                                html.Div("or click to browse  ·  jpg, png",
                                         className="up-sub"),
                            ]
                        ),
                        multiple=False,
                        className="uploader",
                    ),
                    html.Div(id="extract-status", className="status"),
                ],
                className="upload-col",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("Current receipt", className="eyebrow"),
                            html.Span(id="current-source", className="src-name"),
                        ],
                        className="table-head",
                    ),
                    dash_table.DataTable(
                        id="receipt-table",
                        columns=TABLE_COLUMNS,
                        data=[],
                        editable=True,
                        row_deletable=True,
                        dropdown={
                            "category": {
                                "options": [
                                    {"label": c, "value": c} for c in CATEGORIES
                                ]
                            }
                        },
                        style_as_list_view=True,
                        style_header={
                            "backgroundColor": "transparent",
                            "borderBottom": f"1.5px solid {INK}",
                            "color": MUTED,
                            "fontWeight": "600",
                            "fontSize": "11px",
                            "letterSpacing": "0.08em",
                            "textTransform": "uppercase",
                            "fontFamily": "var(--mono)",
                        },
                        style_cell={
                            "backgroundColor": "transparent",
                            "borderBottom": f"1px solid {RULE}",
                            "color": INK,
                            "fontFamily": "var(--body)",
                            "fontSize": "14px",
                            "padding": "10px 12px",
                            "textAlign": "left",
                        },
                        style_cell_conditional=[
                            {"if": {"column_id": c},
                             "fontFamily": "var(--mono)", "textAlign": "right",
                             "width": "100px"}
                            for c in ["amount_purchased", "price_per_item", "total_price"]
                        ],
                        style_data_conditional=[
                            {"if": {"column_id": "category"},
                             "fontStyle": "italic", "color": RED},
                        ],
                    ),
                    html.Div(
                        [
                            html.Div(id="running-total", className="rtotal"),
                            html.Div(
                                [
                                    html.Button("Add row", id="add-row",
                                                className="btn ghost"),
                                    html.Button("Submit to ledger", id="submit-rows",
                                                className="btn solid"),
                                ],
                                className="btn-row",
                            ),
                        ],
                        className="table-foot",
                    ),
                    html.Div(id="submit-status", className="status"),
                ],
                className="table-col",
            ),
        ],
        className="grid",
    )


def dashboard_tab():
    return html.Div(
        [
            html.Div(
                [
                    html.Span("01", className="chart-num"),
                    html.Span("Daily spending by category", className="chart-title"),
                ],
                className="chart-head",
            ),
            dcc.Graph(id="bar-chart", config={"displayModeBar": False}),
            html.Div(
                [
                    html.Span("02", className="chart-num"),
                    html.Span("Where it goes", className="chart-title"),
                ],
                className="chart-head",
            ),
            dcc.Graph(id="donut-chart", config={"displayModeBar": False}),
        ],
        className="dash-col",
    )


app.layout = html.Div(
    [
        html.Header(
            [
                html.Div(
                    [
                        html.Span("❧", className="mark"),
                        html.H1("Ledger", className="wordmark"),
                    ],
                    className="brand",
                ),
                html.P("Receipts in, categorized out.", className="tagline"),
            ],
            className="masthead",
        ),
        dcc.Tabs(
            id="tabs",
            value="capture",
            className="tabs",
            children=[
                dcc.Tab(label="Capture", value="capture", className="tab",
                        selected_className="tab--on"),
                dcc.Tab(label="Dashboard", value="dashboard", className="tab",
                        selected_className="tab--on"),
            ],
        ),
        html.Div(id="tab-body"),
        dcc.Store(id="current-source-store"),
    ],
    className="app",
)


# ---------------------------------------------------------------------------
# Tab routing
# ---------------------------------------------------------------------------

@callback(Output("tab-body", "children"), Input("tabs", "value"))
def render_tab(tab):
    return dashboard_tab() if tab == "dashboard" else upload_tab()


# ---------------------------------------------------------------------------
# Upload -> extract
# ---------------------------------------------------------------------------

@callback(
    Output("receipt-table", "data"),
    Output("extract-status", "children"),
    Output("extract-status", "className"),
    Output("current-source", "children"),
    Output("current-source-store", "data"),
    Input("upload-image", "contents"),
    State("upload-image", "filename"),
    prevent_initial_call=True,
)
def on_upload(contents, filename):
    if not contents:
        raise PreventUpdate

    # contents looks like "data:image/jpeg;base64,...."
    header, b64 = contents.split(",", 1)

    try:
        data = extract_receipt(b64)
    except ollama.ResponseError as e:
        return (dash.no_update,
                f"Ollama error: {e}. Is the model pulled?  ollama pull {MODEL_NAME}",
                "status err", dash.no_update, dash.no_update)
    except json.JSONDecodeError as e:
        return (dash.no_update, f"Model returned invalid JSON: {e}",
                "status err", dash.no_update, dash.no_update)
    except Exception as e:  # noqa: BLE001
        return (dash.no_update, f"Extraction failed: {e}",
                "status err", dash.no_update, dash.no_update)

    rows = []
    for it in data.get("items", []):
        rows.append(
            {
                "item": it["item"],
                "amount_purchased": it["amount_purchased"],
                "price_per_item": it["price_per_item"],
                "total_price": it["total_price"],
                "category": "",
            }
        )

    date = data.get("date", "")
    msg = f"Extracted {len(rows)} item(s) · receipt date {date or 'unknown'}. Add categories, then submit."
    return rows, msg, "status ok", filename, {"source": filename, "date": date}


# ---------------------------------------------------------------------------
# Add row / running total
# ---------------------------------------------------------------------------

@callback(
    Output("receipt-table", "data", allow_duplicate=True),
    Input("add-row", "n_clicks"),
    State("receipt-table", "data"),
    prevent_initial_call=True,
)
def add_row(n, rows):
    if not n:
        raise PreventUpdate
    rows = rows or []
    rows.append({"item": "", "amount_purchased": 1,
                 "price_per_item": 0, "total_price": 0, "category": ""})
    return rows


@callback(
    Output("running-total", "children"),
    Input("receipt-table", "data"),
)
def running_total(rows):
    if not rows:
        return ""
    total = 0.0
    for r in rows:
        try:
            total += float(r.get("total_price") or 0)
        except (TypeError, ValueError):
            pass
    return f"Receipt total  {total:,.2f}"


# ---------------------------------------------------------------------------
# Submit -> DuckDB
# ---------------------------------------------------------------------------

@callback(
    Output("submit-status", "children"),
    Output("submit-status", "className"),
    Output("receipt-table", "data", allow_duplicate=True),
    Output("extract-status", "children", allow_duplicate=True),
    Input("submit-rows", "n_clicks"),
    State("receipt-table", "data"),
    State("current-source-store", "data"),
    prevent_initial_call=True,
)
def submit(n, rows, src):
    if not n:
        raise PreventUpdate
    if not rows:
        return "Nothing to submit — upload a receipt first.", "status err", dash.no_update, dash.no_update

    src = src or {}
    payload = []
    for r in rows:
        payload.append(
            {
                "source_file": src.get("source", "upload"),
                "date": src.get("date", ""),
                "item": r.get("item", ""),
                "amount_purchased": float(r.get("amount_purchased") or 0),
                "price_per_item": float(r.get("price_per_item") or 0),
                "total_price": float(r.get("total_price") or 0),
                "category": r.get("category", ""),
            }
        )

    written = db.append_rows(payload)
    msg = f"Wrote {written} row(s) to the ledger. Upload the next receipt."
    # Clear the table for the next image.
    return msg, "status ok", [], "Ready for the next receipt."


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _empty_fig(note):
    fig = go.Figure()
    fig.add_annotation(text=note, showarrow=False,
                       font=dict(color=MUTED, size=14, family="Georgia"))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        height=300, margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


@callback(Output("bar-chart", "figure"), Input("tabs", "value"))
def bar_chart(tab):
    if tab != "dashboard":
        raise PreventUpdate
    df = db.spend_by_day_category()
    if df.empty:
        return _empty_fig("No expenses yet — submit a receipt first.")

    days = sorted(df["day"].unique())
    categories = list(df["category"].unique())

    fig = go.Figure()
    for i, cat in enumerate(categories):
        sub = df[df["category"] == cat].set_index("day")["total"]
        fig.add_bar(
            name=cat,
            x=days,
            y=[float(sub.get(d, 0)) for d in days],
            marker_color=ACCENT_SEQ[i % len(ACCENT_SEQ)],
            hovertemplate=f"%{{x}}<br>{cat}: %{{y:,.2f}}<extra></extra>",
        )

    fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK, family="Georgia"),
        xaxis=dict(title="", gridcolor="rgba(0,0,0,0)", tickangle=-30,
                   type="category"),
        yaxis=dict(title="", gridcolor=RULE, zerolinecolor=RULE),
        height=380, margin=dict(l=40, r=20, t=10, b=70), bargap=0.35,
        bargroupgap=0.08,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0,
                    font=dict(family="var(--mono)", size=11)),
    )
    return fig


@callback(Output("donut-chart", "figure"), Input("tabs", "value"))
def donut_chart(tab):
    if tab != "dashboard":
        raise PreventUpdate
    df = db.spend_by_category()
    if df.empty:
        return _empty_fig("No categories yet.")
    fig = go.Figure(
        go.Pie(
            labels=df["category"], values=df["total"], hole=0.62,
            marker=dict(colors=ACCENT_SEQ, line=dict(color=PANEL, width=2)),
            textinfo="label+percent", textfont=dict(family="Georgia", size=12),
            hovertemplate="%{label}<br>%{value:,.2f} (%{percent})<extra></extra>",
            sort=True,
        )
    )
    top = df.iloc[0]
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK, family="Georgia"), showlegend=False,
        height=380, margin=dict(l=20, r=20, t=20, b=20),
        annotations=[dict(text=f"{top['category']}<br><b>most</b>",
                          x=0.5, y=0.5, showarrow=False,
                          font=dict(size=14, color=INK, family="Georgia"))],
    )
    return fig


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

app.index_string = """<!DOCTYPE html>
<html>
<head>
{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
:root{
  --ink:#1c1a17; --paper:#f6f2e9; --panel:#fffdf7; --rule:#d8cfbc;
  --red:#9c2b1e; --muted:#6f685c;
  --display:'Georgia','Times New Roman',serif;
  --body:'Iowan Old Style','Palatino Linotype','Georgia',serif;
  --mono:'SF Mono','JetBrains Mono','Courier New',monospace;
}
*{box-sizing:border-box;}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--body);}
.app{max-width:1100px;margin:0 auto;padding:36px 28px 80px;}
.masthead{border-bottom:2px solid var(--ink);padding-bottom:18px;margin-bottom:0;}
.brand{display:flex;align-items:baseline;gap:12px;}
.mark{color:var(--red);font-size:26px;line-height:1;}
.wordmark{font-family:var(--display);font-weight:700;font-size:40px;
  letter-spacing:-0.02em;margin:0;}
.tagline{font-family:var(--mono);font-size:12px;color:var(--muted);
  letter-spacing:0.04em;margin:6px 0 0;text-transform:uppercase;}
.tabs{border-bottom:1px solid var(--rule)!important;margin-top:8px;}
.tab{border:none!important;background:transparent!important;
  font-family:var(--mono)!important;font-size:12px!important;
  text-transform:uppercase;letter-spacing:0.08em;color:var(--muted)!important;
  padding:14px 18px!important;}
.tab--on{color:var(--ink)!important;border-bottom:2px solid var(--red)!important;
  font-weight:700!important;}
.grid{display:grid;grid-template-columns:300px 1fr;gap:32px;margin-top:28px;}
.uploader{border:1.5px dashed var(--rule);border-radius:2px;background:var(--panel);
  padding:38px 18px;text-align:center;cursor:pointer;transition:border-color .15s;}
.uploader:hover{border-color:var(--red);}
.up-title{font-family:var(--display);font-size:18px;}
.up-sub{font-family:var(--mono);font-size:11px;color:var(--muted);margin-top:6px;
  letter-spacing:0.04em;}
.table-head{display:flex;align-items:baseline;gap:12px;margin-bottom:6px;}
.eyebrow{font-family:var(--mono);font-size:10px;text-transform:uppercase;
  letter-spacing:0.1em;color:var(--muted);}
.src-name{font-family:var(--display);font-size:16px;}
.table-foot{display:flex;justify-content:space-between;align-items:center;
  margin-top:16px;padding-top:14px;border-top:1.5px solid var(--ink);}
.rtotal{font-family:var(--mono);font-size:13px;letter-spacing:0.04em;}
.btn-row{display:flex;gap:10px;}
.btn{font-family:var(--mono);font-size:12px;letter-spacing:0.04em;
  text-transform:uppercase;padding:10px 16px;border-radius:2px;cursor:pointer;
  border:1px solid var(--ink);transition:all .15s;}
.btn.ghost{background:transparent;color:var(--ink);}
.btn.ghost:hover{background:var(--ink);color:var(--paper);}
.btn.solid{background:var(--red);color:#fff;border-color:var(--red);}
.btn.solid:hover{background:#7e2117;}
.status{font-family:var(--mono);font-size:12px;margin-top:14px;line-height:1.5;}
.status.ok{color:#3f6b4f;}
.status.err{color:var(--red);}
.dash-col{margin-top:28px;}
.chart-head{display:flex;align-items:baseline;gap:14px;margin:26px 0 4px;}
.chart-num{font-family:var(--mono);font-size:12px;color:var(--red);}
.chart-title{font-family:var(--display);font-size:22px;}
/* DataTable category dropdown */
.Select-control{background:var(--panel)!important;border:1px solid var(--rule)!important;
  border-radius:2px!important;font-family:var(--body)!important;}
.Select-menu-outer{background:var(--panel)!important;border:1px solid var(--rule)!important;
  font-family:var(--body)!important;z-index:1000!important;}
.Select-value-label,.Select-option{color:var(--ink)!important;}
.Select-option.is-focused{background:var(--paper)!important;}
.Select-option.is-selected{background:var(--rule)!important;}
.dash-spreadsheet-container .dash-spreadsheet-inner{overflow:visible!important;}
@media(max-width:760px){.grid{grid-template-columns:1fr;}}
</style>
</head>
<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body>
</html>"""


if __name__ == "__main__":
    app.run(debug=True, port=8050)
