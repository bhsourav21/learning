import dspy
from pydantic import BaseModel, Field

lm = dspy.LM(
    "ollama_chat/gemma3:4b",
    api_base="http://localhost:11434",
    api_key="",
)
dspy.configure(lm=lm)

dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)

class LineItem(BaseModel):
    description: str
    quantity: int
    unit_price: float

class Invoice(BaseModel):
    vendor: str
    invoice_number: str
    currency: str = Field(description="ISO-4217 code, e.g. USD")
    items: list[LineItem]
    total: float

class ExtraInvoice(dspy.Signature):
    raw_text: str = dspy.InputField()
    invoice: Invoice = dspy.OutputField()

ocr_text = """
NORTHWIND TECH SUPPLY
1420 Harbor Way, Suite 300
Seattle, WA 98101

INVOICE                      Invoice No:  INV-2026-04871
                             Date:        2026-03-14
                             Terms:       Net 30
Bill To:
  Lakeside Analytics Ltd.
  88 Granville St, Vancouver BC

QTY   DESCRIPTION                        UNIT PRICE      AMOUNT
---------------------------------------------------------------
 2    USB-C to HDMI Adapter, 4K/60Hz          24.99       49.98
 1    Mechanical Keyboard                      89.50       89.50
      (Brown Switch, US layout)
 3    27in 4K IPS Monitor                    329.00      987.00
 4    Laptop Stand - Aluminum                  18.75       75.00
 5    HDMI Cable 2m                             7.20       36.00
---------------------------------------------------------------
                                   SUBTOTAL  USD  1,237.48
                                   TAX 8.5%  USD    105.19
                                   TOTAL     USD  1,342.67

Remit in USD. Thank you for your business.
"""

extract = dspy.Predict(ExtraInvoice)
inv = extract(raw_text=ocr_text).invoice

print(f"vendor:{inv.vendor}")
print(f"invoice_number:{inv.invoice_number}")
print(f"currency:{inv.currency}")
print(f"total:{inv.total}")
print(f"number of items:{len(inv.items)}")

i = 0
for item in inv.items:
    i = i + 1
    print(f"item#:{i}")
    print(f"description:{item.description}")
    print(f"quantity:{item.quantity}")
    print(f"unit_price:{item.unit_price}")


