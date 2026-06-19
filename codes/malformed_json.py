"""
Invoice Data Extractor — Pydantic Validation
=============================================
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
  "total_amount":    <number>,
  "invoice_number": "<string or null>",
  "description":    "<string or null>"
}
If a field cannot be found, use null for its value.
""".strip()


def extract_invoice_data(invoice_text: str) -> InvoiceData:
    """
    Send invoice text to OpenAI with JSON mode enabled.
    Returns a validated InvoiceData Pydantic model instance.
    On Pydantic ValidationError, retries once with a correction prompt.
    Raises ValueError on bad JSON, ValidationError if retry also fails.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": invoice_text.strip()},
    ]

    def call_api(msgs: list) -> str:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=msgs,
            temperature=0,
        )
        return response.choices[0].message.content

    def parse_json(raw: str) -> dict:
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Model returned invalid JSON: {e}\nRaw output: {raw}") from e

    # ── First attempt ──────────────────────────────────────────────────────────
    raw = call_api(messages)
    data = parse_json(raw)

    try:
        return InvoiceData(**data)
    except ValidationError as first_error:
        print(f"first_error:{first_error}")
        print(f"  [!] Pydantic validation failed: {first_error.error_count()} error(s). Retrying with correction prompt...")
        print(f"raw:{raw}")
        # ── Build correction prompt and retry ──────────────────────────────────
        correction = (
            f"Your previous response was:\n{raw}\n\n"
            f"It failed validation with these errors:\n{first_error}\n\n"
            "Please fix the JSON so every key matches the required schema exactly "
            "(e.g. use 'amount', not 'total_amount') and return only the corrected JSON object."
        )
        messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user",      "content": correction},
        ]

        print(f"messages in retry:{messages}")

        retry_raw = call_api(messages)
        retry_data = parse_json(retry_raw)
        return InvoiceData(**retry_data)   # raises ValidationError if still wrong


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

# Question:
# In the retry section why is the "assistant" role needed? What is its role?
# What happens if we do not add it while retrying?

# Answer:
# The retry messages list is structured as:
#   [
#     {"role": "system",    "content": SYSTEM_PROMPT},
#     {"role": "user",      "content": invoice_text},    # original request
#     {"role": "assistant", "content": raw},             # model's bad response
#     {"role": "user",      "content": correction},      # correction prompt
#   ]
#
# What the "assistant" role does:
# It reconstructs the conversation history honestly. The model sees:
# "I said X, then the user told me X was wrong."
# This gives the correction proper context — the model understands it OWNS
# that previous output and must fix it, rather than seeing some random JSON
# snippet pasted by the user.
#
# What happens without it:
#   [
#     {"role": "system", "content": SYSTEM_PROMPT},
#     {"role": "user",   "content": invoice_text},
#     # assistant turn missing — two user turns in a row
#     {"role": "user",   "content": correction},
#   ]
#
# In this specific code it would still work — because the correction message
# already embeds `raw` as text ("Your previous response was:\n{raw}"), so the
# model can still see the bad JSON. But there are real downsides:
#
# - Two consecutive "user" turns is an unnatural conversation shape —
#   the model wasn't designed for it.
# - The model treats the bad JSON as something the USER wrote, not something
#   IT produced — so it may be more defensive or confused about what to "fix".
# - The conversational grounding is weaker: the model doesn't see this as
#   correcting its own mistake, just as a user pasting some JSON and asking for edits.
#
# Think of it like a code review: telling someone "you wrote this buggy function,
# fix it" lands differently than "here is a buggy function someone wrote, fix it" —
# even if the function text is identical in both cases.
