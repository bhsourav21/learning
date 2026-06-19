"""
Invoice Data Extractor — OpenAI Structured Output
==================================================
Uses OpenAI's structured output feature (client.beta.chat.completions.parse)
to extract and parse invoice data directly into a Pydantic model.

Difference from JSON mode (pydantic_validation.py):
  JSON mode    → response_format={"type": "json_object"}
                 Guarantees valid JSON, but NOT a specific schema.
                 You still manually call json.loads() + InvoiceData(**data).

  Structured   → response_format=InvoiceData  (the Pydantic class itself)
  Output         The API enforces the schema server-side before returning.
                 response.choices[0].message.parsed is already an InvoiceData
                 instance — no json.loads(), no InvoiceData(**data) needed.
"""

from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, field_validator

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
    Total: USD $500
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


# No JSON shape needed in the prompt — the schema is sent to the API directly
# via response_format=InvoiceData, so the model already knows the exact shape.
SYSTEM_PROMPT = (
    "You are an invoice data extraction assistant. "
    "Extract the client name, invoice date (ISO YYYY-MM-DD if possible), "
    "final total amount (as a number, no currency symbols), invoice reference number, "
    "and a brief service/item description from the invoice text. "
    "Use null for any field that cannot be found."
)


def extract_invoice_data(invoice_text: str) -> InvoiceData:
    """
    Send invoice text to OpenAI using structured output.
    The API enforces the InvoiceData schema server-side.
    Returns an InvoiceData instance directly from message.parsed —
    no json.loads() or manual Pydantic construction required.
    """
    # response = client.beta.chat.completions.parse(   # ← .parse(), not .create()
    response = client.chat.completions.parse(   # ← .parse(), not .create()
        model="gpt-4o-mini",
        response_format=InvoiceData,                 # ← Pydantic class, not {"type": "json_object"}
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": invoice_text.strip()},
        ],
        temperature=0,
    )

    message = response.choices[0].message

    if message.refusal:
        raise ValueError(f"Model refused to extract: {message.refusal}")

    return message.parsed   # ← already a validated InvoiceData instance


def main():
    print("=" * 60)
    print("  Invoice Extractor — OpenAI Structured Output")
    print("=" * 60)

    for i, invoice in enumerate(SAMPLE_INVOICES, start=1):
        print(f"\n── Invoice #{i} (raw text) {'─' * 30}")
        print(invoice.strip())

        print(f"\n── Extracted & Validated {'─' * 33}")
        try:
            result = extract_invoice_data(invoice)
            print(result.model_dump_json(indent=2))
            print("✓ Structured output parsed successfully.")
        except ValueError as e:
            print(f"✗ Extraction error: {e}")

    print("\n" + "=" * 60)
    print("All invoices processed.")


if __name__ == "__main__":
    main()


# Note:

# Difference from JSON mode (pydantic_validation.py):

# |                    | JSON mode (pydantic_validation.py)          | Structured output (openai_structured_output.py)     |
# |--------------------|---------------------------------------------|-----------------------------------------------------|
# | API call           | client.chat.completions.create()            | client.beta.chat.completions.parse()                |
# | response_format    | {"type": "json_object"}                     | InvoiceData  (the Pydantic class itself)            |
# | Schema enforcement | Your code, after the response               | API server-side, before the response                |
# | Parsing steps      | json.loads() → InvoiceData(**data)          | None — message.parsed is already an InvoiceData     |
# | Wrong key possible?| Yes (see malformed_json.py for demo)        | No — the API rejects it before returning            |
#
# The ValidationError catch is also gone — with structured output, if the response
# comes back at all, message.parsed is guaranteed to be a valid InvoiceData instance.
# The only thing to check is message.refusal, for cases where the model declines to answer.


# Observation (discovered by testing):
#
# Can @field_validator rules be triggered to fail in structured output? → Almost never.
#
# What we tested:
#   1. Invoice with amount "ABCD" (non-numeric text)
#      → Model returned 0.0, not a failure. Said "✓ Structured output parsed successfully."
#
#   2. Invoice with amount "-$500" (credit note / negative value)
#      → Model returned 500.0 (stripped the negative sign), not a failure.
#      → amount_must_be_positive never even saw a negative value to reject.
#
# Why this happens:
#   The model doesn't blindly copy the invoice text into JSON fields.
#   It semantically interprets the data to fit the schema.
#   For a float field named "amount", it understands the context means
#   "a positive payment value", so it coerces accordingly.
#   It would rather return a "reasonable" value than fail.
#
# Key insight — Structured output coerces silently; JSON mode fails loudly:
#
#   | Scenario                    | JSON mode + Pydantic          | Structured output              |
#   |-----------------------------|-------------------------------|--------------------------------|
#   | Wrong key in response       | ValidationError raised        | Impossible — API enforces keys |
#   | Wrong type in response      | ValidationError raised        | Impossible — API enforces types|
#   | Negative amount in invoice  | ValidationError raised        | Model coerces to positive      |
#   | Unparseable amount ("ABCD") | ValidationError raised        | Model coerces to 0.0           |
#   | Silent wrong value          | Unlikely                      | Real risk — no error, bad data |
#
# Conclusion:
#   @field_validator decorators are effectively dead code in structured output —
#   the model coerces values in a way that naturally satisfies business rules
#   before they even run. Structured output prevents crashes but may silently
#   return wrong data. JSON mode + Pydantic is stricter for business logic validation.
