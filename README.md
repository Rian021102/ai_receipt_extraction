# Receipt Extraction API

A small FastAPI service that extracts structured line items from receipt images
using a local vision model (`qwen2.5vl:7b`) through Ollama.

## Layout

```
src/
  extractor.py   # core logic — reusable from any app (CLI, worker, notebook)
  main.py        # FastAPI app — thin HTTP layer over extractor.py
  requirements.txt
```

The split matters: `extractor.py` has no FastAPI dependency, so you can import
`extract_from_path`, `results_to_dataframe`, etc. directly in other applications
later. `main.py` only handles HTTP concerns.

## Setup

```bash
# 1. Make sure Ollama is running and the model is pulled
ollama pull qwen2.5vl:7b
ollama list            # confirm the exact tag

# 2. Install deps
pip install -r requirements.txt
```

## Run

```bash
uvicorn main:app --reload
```

- Interactive docs: http://127.0.0.1:8000/docs
- Health check:      http://127.0.0.1:8000/health

## Endpoints

| Method | Path             | Purpose                                            |
|--------|------------------|----------------------------------------------------|
| GET    | `/health`        | Liveness + checks the model is pulled              |
| POST   | `/extract`       | Single uploaded image → structured result          |
| POST   | `/extract-path`  | Single image by server-side file path              |
| POST   | `/extract-batch` | Many uploads → items table + per-image grand totals|

## Examples

Health:

```bash
curl http://127.0.0.1:8000/health
```

Single upload:

```bash
curl -X POST http://127.0.0.1:8000/extract \
  -F "file=@/home/rian/images/receipt.jpg"
```

Server-side path:

```bash
curl -X POST "http://127.0.0.1:8000/extract-path?path=/home/rian/images/receipt.jpg"
```

Batch:

```bash
curl -X POST http://127.0.0.1:8000/extract-batch \
  -F "files=@/home/rian/images/a.jpg" \
  -F "files=@/home/rian/images/b.jpg"
```

## Reusing the core in another app

```python
from extractor import extract_from_path, results_to_dataframe, grand_totals

result = extract_from_path("/home/rian/images/receipt.jpg")
df = results_to_dataframe([result])
print(grand_totals(df))
```

## Notes

- Line totals are **recomputed** server-side as `amount_purchased x price_per_item`;
  the model's own `total_price` is ignored to avoid arithmetic mistakes.
- `total_match` compares the recomputed grand total against the receipt's printed
  total — a `false` flags an image worth eye-checking.
- If `/health` shows `model_ready: false`, run the `ollama pull` it suggests.
