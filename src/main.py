"""Receipt extraction API — self-contained.

The vision model (qwen2.5vl:7b via Ollama) runs inside this FastAPI app.

Run:
    uvicorn main:app --reload

Docs: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import base64
import json
from typing import Optional

import ollama
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Config
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
                "required": [
                    "item",
                    "amount_purchased",
                    "price_per_item",
                    "total_price",
                ],
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


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class LineItem(BaseModel):
    item: str
    amount_purchased: float
    price_per_item: float
    total_price: float


class ExtractRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded image bytes.")
    source: str = Field("upload", description="Identifier (e.g. filename).")


class ReceiptResult(BaseModel):
    source: str
    date: str
    items: list[LineItem]
    printed_total: Optional[float] = None
    computed_total: float
    total_match: Optional[bool] = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Receipt Extraction API",
    description="Extract structured line items from receipt images (base64) using a local VLM.",
    version="1.0.0",
)


def _run_model(image_b64: str, source: str) -> ReceiptResult:
    """Call the VLM, then recompute totals and validate."""
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": PROMPT,
                "images": [image_b64],
            }
        ],
        format=EXTRACTION_SCHEMA,
        options={"temperature": 0},
    )

    data = json.loads(response["message"]["content"])

    items: list[LineItem] = []
    for raw in data.get("items", []):
        total = round(
            float(raw["amount_purchased"]) * float(raw["price_per_item"]), 2
        )
        items.append(
            LineItem(
                item=raw["item"],
                amount_purchased=raw["amount_purchased"],
                price_per_item=raw["price_per_item"],
                total_price=total,
            )
        )

    computed_total = round(sum(it.total_price for it in items), 2)
    printed = data.get("printed_total")
    match = abs(float(printed) - computed_total) < 0.01 if printed is not None else None

    return ReceiptResult(
        source=source,
        date=data.get("date", ""),
        items=items,
        printed_total=printed,
        computed_total=computed_total,
        total_match=match,
    )


@app.get("/health")
def health() -> dict:
    """Liveness + check the model is pulled."""
    try:
        models = [m.model for m in ollama.list().models]
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Cannot reach Ollama: {e}")

    ready = MODEL_NAME in models
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "model_ready": ready,
        "available_models": models,
        "hint": None if ready else f"Run: ollama pull {MODEL_NAME}",
    }


@app.post("/extract", response_model=ReceiptResult)
def extract(req: ExtractRequest) -> ReceiptResult:
    """Extract one receipt from a base64-encoded image."""
    try:
        base64.b64decode(req.image_base64, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data.")

    try:
        return _run_model(req.image_base64, req.source)
    except ollama.ResponseError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama error: {e}. Try: ollama pull {MODEL_NAME}",
        )
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=502, detail=f"Model returned invalid JSON: {e}")