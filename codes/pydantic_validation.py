"""
Invoice Data Extractor — Pydantic Validation
==============================================
Uses OpenAI's JSON mode to extract invoice data, then validates
and parses the JSON output with a Pydantic model (5 fields).
"""

import json
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError, field_validator

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

# ── Pydantic model ─────────────────────────────────────────────────────────────
class InvoiceData(BaseModel):
    name: str                        # client or company being billed
    date: str                        # invoice date (ISO YYYY-MM-DD preferred)
    amount: float                    # final total as a number
    invoice_number: Optional[str]    # invoice ID/reference, if present
    description: Optional[str]       # service or item description, if present

    @field_validator("amount")
    @classmethod
    def amount_must_be_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"amount must be positive, got {v}")
        return v

    @field_validator("date")
    @classmethod
    def date_must_be_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("date field must not be empty")
        return v.strip()


SYSTEM_PROMPT = """
You are an invoice data extraction assistant.
Extract the following fields from the invoice text provided by the user:
  - name:           the client or company name being billed
  - date:           the invoice date (use ISO format YYYY-MM-DD if possible)
  - amount:         the final total amount due as a number (no currency symbols)
  - invoice_number: the invoice reference number or ID (e.g. #1042, INV-2025-089)
  - description:    a brief description of the service or item billed

Always respond with ONLY a JSON object in this exact shape:
{
  "name":           "<string>",
  "date":           "<string>",
  "amount":         <number>,
  "invoice_number": "<string or null>",
  "description":    "<string or null>"
}
If a field cannot be found, use null for its value.
""".strip()


def extract_invoice_data(invoice_text: str) -> InvoiceData:
    """
    Send invoice text to OpenAI with JSON mode enabled.
    Returns a validated InvoiceData Pydantic model instance.
    Raises ValueError on bad JSON, ValidationError on schema mismatch.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": invoice_text.strip()},
        ],
        temperature=0,
    )

    raw = response.choices[0].message.content

    # Parse JSON
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model returned invalid JSON: {e}\nRaw output: {raw}") from e

    # Validate and parse with Pydantic — raises ValidationError on failure
    return InvoiceData(**data)


def main():
    print("=" * 60)
    print("  Invoice Extractor — Pydantic Validation")
    print("=" * 60)

    for i, invoice in enumerate(SAMPLE_INVOICES, start=1):
        print(f"\n── Invoice #{i} (raw text) {'─' * 30}")
        print(invoice.strip())

        print(f"\n── Extracted & Validated {'─' * 33}")
        try:
            result = extract_invoice_data(invoice)
            print(result.model_dump_json(indent=2))
            print("✓ Pydantic validation passed.")
        except ValidationError as e:
            print(f"✗ Pydantic validation error:\n{e}")
        except ValueError as e:
            print(f"✗ Extraction error: {e}")

    print("\n" + "=" * 60)
    print("All invoices processed.")


if __name__ == "__main__":
    main()


# Note:

# Question: Tell me the flow of @pydantic_validation.py  especially when pydantic validation take s place

# Answer:
# main()
#   └─ for each invoice string in SAMPLE_INVOICES
#        └─ extract_invoice_data(invoice)
#             │
#             ├─ 1. OpenAI API call
#             │     → response_format={"type": "json_object"} guarantees raw string is valid JSON
#             │
#             ├─ 2. json.loads(raw)
#             │     → converts the JSON string into a plain Python dict
#             │     → e.g. {"name": "Acme", "date": "2025-05-15", "amount": 4750.0, ...}
#             │     → raises ValueError if somehow not valid JSON
#             │
#             └─ 3. InvoiceData(**data)   ← PYDANTIC VALIDATION HAPPENS HERE
#                   │
#                   ├─ Unpacks the dict as keyword args into the model
#                   ├─ Checks every field's type (str, float, Optional[str])
#                   ├─ Runs @field_validator("amount") → rejects if negative
#                   ├─ Runs @field_validator("date")   → rejects if blank/empty
#                   └─ Returns an InvoiceData instance if all passes
#                      OR raises ValidationError with details if anything fails

#   └─ result.model_dump_json(indent=2)
#        → serialises the validated InvoiceData instance back to a pretty JSON string for printing
