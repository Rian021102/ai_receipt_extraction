from ollama import chat
import base64
import json
import re
import pandas as pd

'''''
Process a single image, extract information using an Ollama vision model,
and display the results in a table.
'''''

def encode_image_to_base64(image_path: str) -> str:
    """Read an image file and return a base64-encoded string (no data URI prefix)."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def extract_json_from_response(response_text: str) -> str:
    """Extract JSON from response that might contain markdown code blocks."""
    # Remove markdown code block if present
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
    if json_match:
        return json_match.group(1).strip()
    return response_text.strip()

def main():

    image_path='your_directory/receipt1.jpg'  # Replace with your image path
    response = chat(
        model="gemma3:4b",
        messages=[
            {
                "role": "user",
                "content":"extract recipient and the amount of transfer in json format. Return only valid JSON without any markdown formatting.",
                "images": [image_path],
            },
        ],
    )
    print(response.message.content)

    # make the output from json to table

    json_str = extract_json_from_response(response.message.content)
    data = json.loads(json_str)
    df = pd.DataFrame([data] if isinstance(data, dict) else data)
    print(df)

if __name__ == "__main__":
    main()
