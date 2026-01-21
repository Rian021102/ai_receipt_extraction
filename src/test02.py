from ollama import chat
import base64
import json
import re
import pandas as pd
from pathlib import Path
''''
Process all images in a directory, extract information using an Ollama vision model,
and compile results into a single table.
'''''

# Only process common image formats Ollama vision models typically accept
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def encode_image_to_base64(image_path: str) -> str:
    """Read an image file and return a base64-encoded string (no data URI prefix)."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_json_from_response(response_text: str) -> str:
    """Extract JSON from response that might contain markdown code blocks."""
    # Remove markdown code block if present
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text)
    if json_match:
        return json_match.group(1).strip()
    return response_text.strip()


def main():
    # images_dir = Path("/Users/rianrachmanto/miniforge3/project/images/")
    images_dir=Path("/images/")

    rows = []  # accumulate results across images

    for p in sorted(images_dir.iterdir()):
        # Skip directories and non-images (also skips .DS_Store)
        if not p.is_file():
            continue
        if p.suffix.lower() not in ALLOWED_EXTS:
            print(f"Skipping (not supported image ext): {p.name}")
            continue

        try:
            img_b64 = encode_image_to_base64(str(p))

            response = chat(
                model="gemma3:4b",
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Extract the following information from this receipt/transfer image and return as JSON:\n"
                            '{"recipient": "name of recipient", "amount": "transfer amount", "date": "transaction date"}\n'
                            "Return only valid JSON without any markdown formatting or explanation."
                        ),
                        # IMPORTANT: images must be base64 strings (or bytes), NOT file paths
                        "images": [img_b64],
                    }
                ],
            )

            raw_text = response.message.content
            print(f"\n=== {p.name} ===")
            print(raw_text)

            json_str = extract_json_from_response(raw_text)

            # If the model returns nothing or non-JSON, json.loads will raise
            data = json.loads(json_str)

            # Normalize to list of dicts
            if isinstance(data, dict):
                data["_source_file"] = p.name
                rows.append(data)
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        item["_source_file"] = p.name
                        rows.append(item)
            else:
                print(f"Unexpected JSON type from {p.name}: {type(data)}")

        except json.JSONDecodeError as e:
            print(f"[JSON ERROR] {p.name}: {e}")
        except Exception as e:
            print(f"[OLLAMA/IO ERROR] {p.name}: {e}")

    if rows:
        df = pd.DataFrame(rows)
        print("\n=== Combined table ===")
        #change the value of the column recipient to person 1, person 2 and so on
        df['recipient'] = ['person ' + str(i+1) for i in range(len(df))]
        #remove source_file and account_number and account columns
        print(df)
    else:
        print("\nNo valid rows extracted.")


if __name__ == "__main__":
    main()
