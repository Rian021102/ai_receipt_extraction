import cv2
import base64
import json
import re
import pandas as pd
from pathlib import Path
from ollama import chat


ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# How many times to re-ask the model when its output fails arithmetic validation.
MAX_EXTRACT_RETRIES = 2

# Allowed relative error between the summed line totals and the receipt total.
RECEIPT_TOTAL_TOLERANCE = 0.02


def enhance_receipt_image(image_path: str, output_path: str) -> str:
    """
    Lightly prepare a receipt image for a vision model.

    Vision LLMs are trained on natural color images, not binarized OCR-style
    scans. So we keep the image in color and only upscale small images, plus a
    gentle contrast bump. Aggressive grayscale/denoise/threshold pipelines tend
    to destroy faint print and make extraction LESS consistent.
    """

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # 1. Upscale only if the image is small (helps the model read small text).
    h, w = img.shape[:2]
    if max(h, w) < 1500:
        img = cv2.resize(
            img,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC,
        )

    # 2. Mild contrast/brightness improvement in color via CLAHE on the L channel.
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    img = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)

    cv2.imwrite(output_path, img)

    return output_path


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_json_from_response(response_text: str) -> str:
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text)
    if json_match:
        return json_match.group(1).strip()
    return response_text.strip()


def normalize_number(value):
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return int(value)

    value = str(value)
    value = value.replace("Rp", "")
    value = value.replace(".", "")
    value = value.replace(",", "")
    value = value.replace(" ", "")
    value = value.strip()

    if value == "":
        return None

    try:
        return int(float(value))
    except ValueError:
        return None


def validate_receipt(data) -> bool:
    """
    Cheap sanity checks so we can retry when a small model miscounts.

    Returns True only when the data looks internally consistent:
    - there is at least one item,
    - every item has a numeric line_total,
    - the summed line totals match receipt_total within tolerance
      (skipped if receipt_total is missing).
    """
    if not isinstance(data, dict):
        return False

    items = data.get("items") or []
    if not items:
        return False

    line_totals = [normalize_number(i.get("line_total")) for i in items]
    if any(lt is None for lt in line_totals):
        return False

    receipt_total = normalize_number(data.get("receipt_total"))
    if receipt_total:
        summed = sum(line_totals)
        if abs(summed - receipt_total) / receipt_total > RECEIPT_TOTAL_TOLERANCE:
            return False

    return True


def extract_receipt(image_path):
    enhanced_dir = Path("/home/rian/python_project/myvenv/ai_receipt_extraction/enhanced_images")
    enhanced_dir.mkdir(parents=True, exist_ok=True)

    enhanced_path = enhanced_dir / f"enhanced_{Path(image_path).name}"

    # Enhance image first
    enhance_receipt_image(
        image_path=str(image_path),
        output_path=str(enhanced_path)
    )

    # Send enhanced image to Ollama
    img_b64 = encode_image_to_base64(str(enhanced_path))

    prompt = """
You are extracting structured data from a supermarket receipt.

Return ONLY valid JSON. No markdown. No explanation.

Use exactly this schema:

{
  "purchase_date": "YYYY-MM-DD",
  "items": [
    {
      "item_name": "string",
      "quantity": number,
      "unit": "kg or pcs",
      "unit_price": integer,
      "line_total": integer
    }
  ],
  "receipt_total": integer
}

Rules:
- purchase_date must come from the purchase date printed on the receipt.
- Apply the same purchase_date to every item.
- Extract only purchased items.
- Do not include discount rows.
- Do not include TOTAL, BCA, tax, saving, payment method, member number, or receipt metadata.
- For weighted items, quantity is the kg value.
- For normal items, quantity is the number of pieces.
- If quantity is not shown, use 1.
- unit must be either "kg" or "pcs".
- unit_price is the price per kg or per piece.
- line_total is the total amount paid for that item line.
- line_total should equal unit_price * quantity. If they disagree, recheck the
  numbers you read before answering.
- The sum of all line_total values should equal receipt_total.
- Numbers must be integers without dots, commas, Rp, or currency symbols.
- Date format must be YYYY-MM-DD.

Example output:
{
  "purchase_date": "2024-05-01",
  "items": [
    {"item_name": "Apel Fuji", "quantity": 2, "unit": "pcs", "unit_price": 15000, "line_total": 30000},
    {"item_name": "Daging Sapi", "quantity": 0.5, "unit": "kg", "unit_price": 120000, "line_total": 60000}
  ],
  "receipt_total": 90000
}
"""

    last_data = None
    for _ in range(MAX_EXTRACT_RETRIES + 1):
        response = chat(
            model="qwen2.5vl:latest",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [img_b64],
                }
            ],
            format="json",
            options={
                "temperature": 0,
                "seed": 42,
                "num_ctx": 4096,
            },
        )

        json_str = extract_json_from_response(response.message.content)

        try:
            last_data = json.loads(json_str)
        except json.JSONDecodeError:
            continue

        if validate_receipt(last_data):
            return last_data

    # No attempt passed validation; return the best effort (or raise if none parsed).
    if last_data is None:
        raise json.JSONDecodeError("No valid JSON returned by model", json_str, 0)

    return last_data


def flatten_receipt(data):
    rows = []

    purchase_date = data.get("purchase_date")
    receipt_total = normalize_number(data.get("receipt_total"))

    for item in data.get("items", []):
        rows.append({
            "date": purchase_date,
            "item_name": item.get("item_name"),
            "quantity": item.get("quantity"),
            "unit": item.get("unit"),
            "unit_price": normalize_number(item.get("unit_price")),
            "line_total": normalize_number(item.get("line_total")),
            "receipt_total": receipt_total,
        })

    return rows


def main():
    images_dir = Path("/home/rian/python_project/myvenv/ai_receipt_extraction/images")
    output_path = Path("/home/rian/python_project/myvenv/ai_receipt_extraction/data/extracted_receipts.xlsx")

    all_rows = []

    for p in sorted(images_dir.iterdir()):
        if not p.is_file():
            continue

        if p.suffix.lower() not in ALLOWED_EXTS:
            print(f"Skipping unsupported file: {p.name}")
            continue

        print(f"Processing: {p.name}")

        try:
            receipt_data = extract_receipt(p)
            rows = flatten_receipt(receipt_data)
            all_rows.extend(rows)

        except json.JSONDecodeError as e:
            print(f"[JSON ERROR] {p.name}: {e}")

        except Exception as e:
            print(f"[ERROR] {p.name}: {e}")

    if all_rows:
        df = pd.DataFrame(all_rows)

        columns = [
            "date",
            "item_name",
            "quantity",
            "unit",
            "unit_price",
            "line_total",
            "receipt_total",
        ]

        df = df[columns]

        print(df)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(output_path, index=False)

        print(f"Saved to: {output_path}")

    else:
        print("No valid receipt rows extracted.")


if __name__ == "__main__":
    main()