import os
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from openai import OpenAI
from ragas import EvaluationDataset
from ragas import evaluate
from ragas.metrics import (
Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall,
)

load_dotenv()

evaluator_llm= LangchainLLMWrapper(
    ChatOpenAI(model="gpt-4o-mini", temperature=0)
    )

evaluator_embeddings= LangchainEmbeddingsWrapper(OpenAIEmbeddings())

rows = [
    {
        "user_input": "What was Acme Corp's ARR and Net Revenue Retention (NRR) in FY2024, and what is the FY2025 target for NRR?",
        "response": (
            "Acme Corp closed FY2024 with ARR of $98.4M, up from $79.2M in FY2023. "
            "Net Revenue Retention for FY2024 was 118% (versus 112% the prior year), "
            "and the company is targeting 120%+ NRR for FY2025."
        ),
        "retrieved_contexts": [
            "Acme Corp is a B2B SaaS company founded in 2015, headquartered in San Francisco, CA. "
            "The company builds cloud-native workflow automation and data integration software for "
            "mid-market and enterprise customers across North America, Europe, and Asia-Pacific. Acme "
            "serves 1,400+ paying customers across 38 countries with an ARR of $98.4M as of Q4 2024.",
            "Metric: ARR | FY2024: $98.4M | FY2023: $79.2M | Target FY2025: $125M\n"
            "Metric: Net Revenue Retention (NRR) | FY2024: 118% | FY2023: 112% | Target FY2025: 120%+\n"
            "Metric: Average Contract Value (ACV) | FY2024: $68,000 | FY2023: $59,000 | Target FY2025: $80,000",
        ],
        "reference": (
            "Acme Corp's ARR was $98.4M in FY2024, up from $79.2M in FY2023. NRR was 118% in FY2024 "
            "(versus 112% in FY2023), and the FY2025 target for NRR is 120%+."
        ),
    },
    {
        "user_input": "What were Globex Industries' on-time delivery rate and defect rate (PPM) in FY2024, and what are the FY2025 targets?",
        "response": (
            "Globex Industries hit 96.8% on-time delivery in FY2024 (up from 94.2% in FY2023) with a "
            "defect rate of 12 PPM (down from 28 PPM). For FY2025 they're targeting 98% OTD and under "
            "10 PPM in defects."
        ),
        "retrieved_contexts": [
            "Metric: On-Time Delivery (OTD) | FY2024: 96.8% | FY2023: 94.2% | Target FY2025: 98%\n"
            "Metric: Defect Rate (PPM) | FY2024: 12 | FY2023: 28 | Target FY2025: <10\n"
            "Metric: Capacity Utilisation | FY2024: 84% | FY2023: 79% | Target FY2025: 88%",
            "Globex Industries is a precision manufacturing company founded in 1982, headquartered in "
            "Detroit, MI. The company designs and manufactures advanced industrial components, "
            "sub-assemblies, and turnkey systems for the automotive, aerospace, and defence sectors. "
            "Globex operates 6 plants across the US and Mexico, employing 4,200 staff globally.",
        ],
        "reference": (
            "In FY2024, Globex Industries achieved an on-time delivery rate of 96.8% (up from 94.2% in "
            "FY2023) and a defect rate of 12 PPM (down from 28 PPM in FY2023). The FY2025 targets are "
            "98% OTD and a defect rate below 10 PPM."
        ),
    },
    {
        "user_input": "How much Total Payment Volume (TPV) did Hooli Ventures process in FY2024, and what is the FY2025 target?",
        "response": (
            "Hooli Ventures processed $4.2B in Total Payment Volume during FY2024, more than double the "
            "$1.9B processed in FY2023. Their FY2025 target is to grow TPV to $9B."
        ),
        "retrieved_contexts": [
            "Hooli Ventures is a fintech startup founded in 2019, headquartered in New York, NY. The "
            "company operates an API-first payments infrastructure platform serving neobanks, B2B "
            "marketplaces, and embedded finance providers. Hooli processed $4.2B in annualised payment "
            "volume in 2024 and raised a $75M Series C in March 2024.",
            "Metric: TPV (Total Payment Volume) | FY2024: $4.2B | FY2023: $1.9B | Target FY2025: $9B\n"
            "Metric: Revenue | FY2024: $38.4M | FY2023: $16.2M | Target FY2025: $82M\n"
            "Metric: Take Rate | FY2024: 0.91% | FY2023: 0.85% | Target FY2025: 0.91%",
        ],
        "reference": (
            "Hooli Ventures processed $4.2B in Total Payment Volume (TPV) in FY2024, up from $1.9B in "
            "FY2023. The FY2025 target is to grow TPV to $9B."
        ),
    },
    {
        "user_input": "What are Initech Solutions' four service lines, and what share of revenue does each represent?",
        "response": (
            "Initech Solutions organizes its business into four service lines: Cloud & Infrastructure "
            "(35% of revenue), Cybersecurity (25%), Data & AI (25%), and Applications (15%)."
        ),
        "retrieved_contexts": [
            "Dimension: Service Lines | Detail: Cloud & Infra (35%), Cybersecurity (25%), Data & AI "
            "(25%), Apps (15%)\n"
            "Dimension: Revenue Model | Detail: T&M (40%), Fixed-fee project (35%), Managed services "
            "retainer (25%)\n"
            "Dimension: Avg. Engagement Size | Detail: $850K; typical duration 6–18 months",
            "Initech Solutions is an IT consulting and managed services firm founded in 2003, "
            "headquartered in Austin, TX. The company provides digital transformation advisory, cloud "
            "migration, cybersecurity, and enterprise application services to Fortune 1000 clients. "
            "Initech has 1,800 consultants across 12 offices in the US, UK, and India.",
        ],
        "reference": (
            "Initech Solutions' four service lines are Cloud & Infrastructure (35% of revenue), "
            "Cybersecurity (25%), Data & AI (25%), and Applications (15%)."
        ),
    },
    {
        "user_input": "How much did Umbrella Dynamics spend on R&D in FY2024, and what percentage of revenue did that represent?",
        "response": (
            "Umbrella Dynamics spent $42.3M on R&D in FY2024, which was 29.8% of revenue — a slight "
            "decrease from 31.2% of revenue in FY2023, though the FY2025 target R&D spend rises to $52M "
            "(31% of revenue)."
        ),
        "retrieved_contexts": [
            "Umbrella Dynamics is a pharmaceutical and biotechnology company founded in 1998, "
            "headquartered in Basel, Switzerland. The company is focused on developing novel "
            "therapeutics in oncology, rare diseases, and immunology. Umbrella has 3 drug candidates in "
            "Phase II/III trials and 2 approved products generating $142M in revenue. R&D Investment: "
            "$42.3M in FY2024 (29.8% of revenue).",
            "Metric: R&D Spend | FY2024: $42.3M | FY2023: $36.8M | Target FY2025: $52M\n"
            "Metric: R&D as % of Revenue | FY2024: 29.8% | FY2023: 31.2% | Target FY2025: 31%\n"
            "Metric: Pipeline Value (NPV) | FY2024: $2.1B | FY2023: $1.7B | Target FY2025: $2.8B",
        ],
        "reference": (
            "Umbrella Dynamics invested $42.3M in R&D in FY2024, representing 29.8% of revenue (down "
            "from 31.2% of revenue in FY2023). The FY2025 target R&D spend is $52M, or about 31% of "
            "revenue."
        ),
    },
    {
        "user_input": "At Acme Corp, what approval level, PO requirements, and turnaround SLA apply to a $30,000 expenditure?",
        "response": (
            "A $30,000 purchase at Acme Corp falls into the $25,001–$100,000 tier, which requires CFO "
            "approval, a purchase order plus three competing quotes, and has a 5-business-day turnaround "
            "SLA."
        ),
        "retrieved_contexts": [
            "All company expenditure must be pre-approved in accordance with the following authorisation "
            "matrix. Purchase orders are mandatory for all vendor engagements above $1,000. Three "
            "competing quotes are required above $10,000.",
            "Expenditure Amount: $5,001 – $25,000 | Approver Level: VP / Director | PO Required: Yes | "
            "Turnaround SLA: 3 business days\n"
            "Expenditure Amount: $25,001 – $100,000 | Approver Level: CFO | PO Required: Yes + 3 quotes "
            "| Turnaround SLA: 5 business days\n"
            "Expenditure Amount: Above $100,000 | Approver Level: CEO + Board approval | PO Required: "
            "Yes + 3 quotes | Turnaround SLA: 10 business days",
        ],
        "reference": (
            "A $30,000 expenditure at Acme Corp falls in the $25,001–$100,000 bracket, requiring CFO "
            "approval, a purchase order, three competing quotes, and a 5-business-day turnaround SLA."
        ),
    },
    {
        "user_input": "How many days of annual/earned leave do Globex Industries employees get per year, and how many days can be carried forward or encashed?",
        "response": (
            "Globex Industries employees get 21 days of annual/earned leave each year. They can carry "
            "forward up to 10 days, and encashment is permitted up to a maximum of 30 days."
        ),
        "retrieved_contexts": [
            "Leave Type: Annual / Earned Leave | Annual Entitlement: 21 days | Carry Forward: Up to 10 "
            "days | Encashment: Yes (max 30 days)\n"
            "Leave Type: Sick / Medical Leave | Annual Entitlement: 10 days | Carry Forward: Not "
            "permitted | Encashment: No\n"
            "Leave Type: Casual Leave | Annual Entitlement: 8 days | Carry Forward: Not permitted | "
            "Encashment: No",
            "Standard working hours at Globex Industries are 9:00 AM to 6:00 PM local time, Monday "
            "through Friday, with a one-hour lunch break. Flexible work arrangements are available "
            "subject to manager approval and role requirements.",
        ],
        "reference": (
            "Globex Industries employees receive 21 days of annual/earned leave per year. Up to 10 days "
            "can be carried forward, and encashment is allowed up to a maximum of 30 days."
        ),
    },
    {
        "user_input": "What is Hooli Ventures' MFA policy, and what is the SLA for acknowledging and containing a P1 (critical) security incident?",
        "response": (
            "Hooli Ventures requires Multi-Factor Authentication on all corporate accounts with no "
            "exceptions. For P1 (critical) incidents, the SLA is 1 hour to acknowledge and 4 hours to "
            "contain."
        ),
        "retrieved_contexts": [
            "Information security at Hooli Ventures is governed by the Information Security Management "
            "System (ISMS). All employees must complete annual security awareness training. The "
            "following controls are mandatory: Multi-Factor Authentication (MFA) on all corporate "
            "accounts — no exceptions. Endpoint Detection & Response (EDR) agent on all managed devices.",
            "Password policy: minimum 14 characters, no reuse of last 12, rotated every 90 days. All "
            "data classified as 'Restricted' or above must be encrypted at rest and in transit. Incident "
            "response SLA: P1 (Critical) — 1-hour acknowledgement, 4-hour containment.",
        ],
        "reference": (
            "Hooli Ventures mandates MFA on all corporate accounts with no exceptions. For P1 (Critical) "
            "security incidents, the SLA is a 1-hour acknowledgement and 4-hour containment."
        ),
    },
    {
        "user_input": "What was Initech Solutions' Net Profit After Tax in FY2024, and how much did it grow year-over-year?",
        "response": (
            "Initech Solutions reported Net Profit After Tax of $11.6M in FY2024, up 68.1% from $6.9M in "
            "FY2023."
        ),
        "retrieved_contexts": [
            "Line Item: EBIT (Operating Profit) | FY2024 (USD M): 14.8 | FY2023 (USD M): 9.9 | YoY "
            "Growth: +49.5%\n"
            "Line Item: Tax Expense (27% effective rate) | FY2024 (USD M): (4.3) | FY2023 (USD M): (2.6) "
            "| YoY Growth: +65.4%\n"
            "Line Item: Net Profit After Tax | FY2024 (USD M): 11.6 | FY2023 (USD M): 6.9 | YoY Growth: "
            "+68.1%\n"
            "Line Item: Net Margin % | FY2024: 8.1% | FY2023: 5.8% | YoY Growth: +230 bps",
            "Initech Solutions's financial operations are governed by a three-tier oversight structure: "
            "the Board Audit Committee (strategic oversight), the CFO office (policy & compliance), and "
            "Departmental Finance Business Partners (operational execution). All financial statements "
            "are prepared in accordance with IFRS.",
        ],
        "reference": (
            "Initech Solutions' Net Profit After Tax was $11.6M in FY2024, up 68.1% from $6.9M in "
            "FY2023."
        ),
    },
    {
        "user_input": "How does Acme Corp's FY2024 ARR compare to Hooli Ventures' FY2024 revenue, and what industries are the two companies in?",
        "response": (
            "Acme Corp (B2B SaaS, workflow automation software) reported FY2024 ARR of $98.4M, while "
            "Hooli Ventures (fintech, API-first payments infrastructure) reported FY2024 revenue of "
            "$38.4M — so Acme's figure was more than 2.5x higher. Note these aren't perfectly "
            "apples-to-apples: ARR is recurring subscription revenue annualized, while Hooli's $38.4M is "
            "actual revenue earned from a much larger $4.2B in payment volume processed."
        ),
        "retrieved_contexts": [
            "Acme Corp is a B2B SaaS company founded in 2015, headquartered in San Francisco, CA... Acme "
            "serves 1,400+ paying customers across 38 countries with an ARR of $98.4M as of Q4 2024. "
            "Metric: ARR | FY2024: $98.4M | FY2023: $79.2M | Target FY2025: $125M",
            "Hooli Ventures is a fintech startup founded in 2019, headquartered in New York, NY... Hooli "
            "processed $4.2B in annualised payment volume in 2024. Metric: Revenue | FY2024: $38.4M | "
            "FY2023: $16.2M | Target FY2025: $82M",
        ],
        "reference": (
            "Acme Corp (Software & SaaS) reported FY2024 ARR of $98.4M, versus Hooli Ventures (Fintech & "
            "Payments) which reported FY2024 revenue of $38.4M. Acme's top-line figure was higher, "
            "though the two metrics aren't directly comparable: Acme's number is annualized recurring "
            "subscription revenue, while Hooli's revenue is derived from a take rate on $4.2B of total "
            "payment volume processed."
        ),
    },
]


dataset= EvaluationDataset.from_list(rows)

print(f"dataset:{dataset}")


result= evaluate(
    dataset=dataset,
    metrics=[
        Faithfulness(), AnswerRelevancy(), 
        ContextPrecision(), ContextRecall(),
    ],
    llm=evaluator_llm,
    embeddings=evaluator_embeddings,
)

print("result")
print(result) # aggregate scores across all 10 rows
df= result.to_pandas() # per-row breakdown, useful for diagnosis
df.to_csv("ragas_results.csv", index=False)