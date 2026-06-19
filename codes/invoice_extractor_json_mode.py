"""
Invoice Data Extractor
======================
Uses OpenAI's JSON mode (response_format={"type": "json_object"}) to extract
name, date, and amount from invoice text.
Verifies every response is valid JSON before returning it.
"""

import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# ── Sample invoices ────────────────────────────────────────────────────────────
SAMPLE_INVOICES = [
    """
    INVOICE #1042
    Date: May 15, 2025
    Billed To: Acme Corporation
    Services: Software Consulting — Q2 Retainer
    Amount Due: $4,750.00
    """,
    """
    Tax Invoice
    Issued: 03/22/2025
    Client: Sarah Johnson
    Description: Website Redesign Project
    Total: USD 12,300
    Payment due within 30 days.
    """,
    """
    Invoice No: INV-2025-089
    To: Global Logistics Ltd.
    Invoice Date: 2025-04-01
    Item: Annual SaaS Subscription
    Subtotal: $899.00
    Tax (10%): $89.90
    Grand Total: $988.90
    """,
]

SYSTEM_PROMPT = """
You are an invoice data extraction assistant.
Extract the following fields from the invoice text provided by the user:
  - name:   the client or company name being billed
  - date:   the invoice date (use ISO format YYYY-MM-DD if possible)
  - amount: the final total amount due as a number (no currency symbols)

Always respond with ONLY a JSON object in this exact shape:
{
  "name":   "<string>",
  "date":   "<string>",
  "amount": <number>
}
If a field cannot be found, use null for its value.
""".strip()


def extract_invoice_data(invoice_text: str) -> dict:
    """
    Send invoice text to OpenAI with JSON mode enabled.
    Returns a validated Python dict with keys: name, date, amount.
    Raises ValueError if the response is not valid JSON.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},   # ← JSON mode ON
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": invoice_text.strip()},
        ],
        temperature=0,   # deterministic output for data extraction
    )

    raw = response.choices[0].message.content

    # ── Verify it is valid JSON ────────────────────────────────────────────────
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model returned invalid JSON: {e}\nRaw output: {raw}") from e

    # ── Verify expected keys are present ──────────────────────────────────────
    expected_keys = {"name", "date", "amount"}
    missing = expected_keys - data.keys()
    if missing:
        raise ValueError(f"JSON is missing expected keys: {missing}\nGot: {data}")

    return data


def main():
    print("=" * 60)
    print("  Invoice Data Extractor — OpenAI JSON Mode")
    print("=" * 60)

    for i, invoice in enumerate(SAMPLE_INVOICES, start=1):
        print(f"\n── Invoice #{i} (raw text) {'─' * 30}")
        print(invoice.strip())

        print(f"\n── Extracted JSON {'─' * 38}")
        try:
            result = extract_invoice_data(invoice)

            # Pretty-print the validated JSON
            print(json.dumps(result, indent=2))

            # Confirm JSON validity explicitly
            json.dumps(result)   # will raise if somehow invalid
            print("✓ Valid JSON confirmed.")

        except ValueError as e:
            print(f"✗ Extraction error: {e}")

    print("\n" + "=" * 60)
    print("All invoices processed.")


if __name__ == "__main__":
    main()


# Note:

# Question:
# In @invoice_extractor_json_mode.py  why response_format={"type": "json_object"} is needed when the below prompt is there

# Always respond with ONLY a JSON object in this exact shape:
# {
#   "name":   "<string>",
#   "date":   "<string>",
#   "amount": <number>
# }

# Answer:
# Good question. The prompt instruction alone is unreliable — the model might still:

# Prefix the JSON with "Here is the extracted data:" or similar
# Add a trailing explanation after the closing }
# Wrap it in a markdown code block (```json ... ```)
# Occasionally drift from the format under edge cases
# response_format={"type": "json_object"} is a hard API-level constraint. 
# It forces the model's output to be parseable JSON — guaranteed — 
# before the response even reaches your code. 
# It's not prompting the model to try to output JSON; 
# it's the API refusing to return anything that isn't valid JSON.

# So the two layers serve different roles:

# Layer	          |   What it does
#----------------------------------------------------------------------------------------
# System prompt	  | Tells the model what shape the JSON should be (name, date, amount)
# response_format |	Guarantees the output is JSON at all — no prose, no markdown fences

# The prompt handles structure; response_format handles validity. 
# You need both — response_format alone would give you valid JSON but with no 
# guarantee of the right keys, and the prompt alone would give you the right keys but 
# no guarantee of clean JSON.



