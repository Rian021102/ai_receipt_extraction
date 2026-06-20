"""Client: send a folder of receipt images to the API and print results as a DataFrame.

Usage:
    python run_folder.py /home/rian/images
    python run_folder.py /home/rian/images --url http://127.0.0.1:8000
    python run_folder.py /home/rian/images --csv out.csv
"""

from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

import pandas as pd
import requests

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def encode(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract receipts from a folder via the API.")
    ap.add_argument("folder", help="Path to the folder containing receipt images.")
    ap.add_argument("--url", default="http://127.0.0.1:8000", help="API base URL.")
    ap.add_argument("--csv", help="Optional path to also save the item table as CSV.")
    args = ap.parse_args()

    folder = Path(args.folder).expanduser()
    if not folder.is_dir():
        print(f"Not a folder: {folder}", file=sys.stderr)
        return 1

    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        print(f"No images found in {folder}", file=sys.stderr)
        return 1

    rows: list[dict] = []
    failures: list[tuple[str, str]] = []

    for img in images:
        try:
            resp = requests.post(
                f"{args.url}/extract",
                json={"image_base64": encode(img), "source": img.name},
                timeout=300,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:  # noqa: BLE001
            failures.append((img.name, str(e)))
            print(f"{img.name}: FAILED ({e})", file=sys.stderr)
            continue

        for it in data["items"]:
            rows.append(
                {
                    "source_file": data["source"],
                    "date": data["date"],
                    "item": it["item"],
                    "amount_purchased": it["amount_purchased"],
                    "price_per_item": it["price_per_item"],
                    "total_price": it["total_price"],
                }
            )

    df = pd.DataFrame(rows)

    if df.empty:
        print("\nNo data extracted — all images failed.")
        return 1

    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

    print("\n=== Items ===")
    print(df.to_string(index=False))

    totals = df.groupby(["source_file", "date"])["total_price"].sum().reset_index()
    totals.columns = ["source_file", "date", "grand_total"]
    print("\n=== Grand totals ===")
    print(totals.to_string(index=False))

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nSaved item table to {args.csv}")

    if failures:
        print(f"\n{len(failures)} file(s) failed.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())