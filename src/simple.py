import ollama
import json
import pandas as pd
from pathlib import Path

img_dir = Path('/home/rian/python_project/myvenv/ai_receipt_extraction/images')

schema = {
    'type': 'object',
    'properties': {
        'date': {'type': 'string'},
        'items': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'item': {'type': 'string'},
                    'amount_purchased': {'type': 'number'},
                    'price_per_item': {'type': 'number'},
                    'total_price': {'type': 'number'}
                },
                'required': ['item', 'amount_purchased', 'price_per_item', 'total_price']
            }
        },
        'printed_total': {'type': 'number'}
    },
    'required': ['date', 'items']
}

prompt = (
    "Extract from this receipt/image:\n"
    "- date: the single transaction date, applied to all items\n"
    "- for each item: item name, amount_purchased (quantity), "
    "price_per_item (unit price), total_price (amount_purchased x price_per_item)\n"
    "- printed_total: the grand total printed on the receipt, if shown\n"
    "Return only valid JSON matching the schema."
)

rows = []

for img in sorted(img_dir.glob('*.jpg')):
    try:
        response = ollama.chat(
            model='qwen2.5vl:latest',
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [str(img)]
            }],
            format=schema,
            options={'temperature': 0}
        )
        data = json.loads(response['message']['content'])
    except (json.JSONDecodeError, ollama.ResponseError) as e:
        print(f"{img.name}: FAILED ({e})")
        continue

    for it in data['items']:
        total = round(it['amount_purchased'] * it['price_per_item'], 2)
        rows.append({
            'source_file': img.name,
            'date': data['date'],
            'item': it['item'],
            'amount_purchased': it['amount_purchased'],
            'price_per_item': it['price_per_item'],
            'total_price': total
        })

df = pd.DataFrame(rows)
print(df.to_string(index=False))

# Save item-level data
df.to_csv(img_dir / 'extracted.csv', index=False)

# Per-image grand totals
totals = df.groupby(['source_file', 'date'])['total_price'].sum().reset_index()
totals.columns = ['source_file', 'date', 'grand_total']
print('\n' + totals.to_string(index=False))

# Save totals
totals.to_csv(img_dir / 'totals.csv', index=False)