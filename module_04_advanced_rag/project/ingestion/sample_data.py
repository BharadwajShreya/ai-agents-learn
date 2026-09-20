"""
Sample Data — Synthetic Financial Reports
==========================================

DESIGN DECISION: Why synthetic data + real PDFs?

We create synthetic markdown documents for TWO reasons:
  1. IMMEDIATE TESTING: We can test chunking/retrieval right now without
     downloading anything. Never block development on data availability.
  2. GROUND TRUTH: We KNOW what's in these docs, so we can create
     perfect evaluation datasets (Phase 4). With real PDFs, creating
     ground truth is manual and slow.

We ALSO use real PDFs (downloaded separately) because:
  1. They have REAL charts, tables, and complex layouts
  2. They test our multimodal pipeline with actual visual content
  3. They make the demo more impressive

This pattern — synthetic for dev, real for integration testing — is
standard practice in production ML systems.
"""

import os
import sys
from pathlib import Path

# Ensure Windows stdout handles UTF-8 characters properly
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass



# =============================================================================
# Synthetic Financial Reports (markdown format)
# =============================================================================

ACME_Q3_REPORT = """
# Acme Corporation — Q3 2025 Earnings Report

## Executive Summary

Acme Corporation reported strong third-quarter results for fiscal year 2025, with revenue reaching $4.2 billion, representing a 15% increase year-over-year. The company's strategic investments in cloud infrastructure and AI-powered services continued to drive growth across all major business segments.

EBITDA for the quarter was $1.2 billion, with an EBITDA margin of 28.6%, expanding 180 basis points from Q3 2024's margin of 26.8%. This margin improvement reflects the company's ongoing focus on operational efficiency and the higher-margin nature of its growing cloud business.

## Revenue Breakdown by Segment

### Cloud Services Division
Cloud services revenue reached $2.1 billion in Q3 2025, up 28% year-over-year from $1.64 billion in Q3 2024. This segment now represents 50% of total revenue, up from 45% a year ago. Key growth drivers include:
- Enterprise SaaS subscriptions grew 35% to $1.2 billion
- Infrastructure-as-a-Service (IaaS) revenue increased 22% to $650 million
- Professional services and consulting contributed $250 million, up 15%

The cloud division signed 47 new enterprise contracts worth over $1 million each during the quarter, compared to 32 such contracts in Q3 2024.

### Hardware Division
Hardware revenue was $1.3 billion, essentially flat year-over-year ($1.28 billion in Q3 2024). The hardware business continues its transition from on-premises servers to cloud-optimized infrastructure:
- Server and networking equipment: $800 million (down 5%)
- AI accelerator chips (new product line): $350 million (up 180% from $125 million)
- Storage solutions: $150 million (down 12%)

The AI accelerator segment is now the fastest-growing hardware category, driven by enterprise demand for on-premises AI inference capabilities.

### Services Division
Services revenue reached $800 million, up 12% from $714 million in Q3 2024. This includes:
- Managed IT services: $450 million (up 8%)
- Cybersecurity services: $200 million (up 25%)
- Training and certification: $150 million (up 10%)

## Financial Highlights

| Metric | Q3 2025 | Q3 2024 | Change |
|--------|---------|---------|--------|
| Total Revenue | $4.2B | $3.65B | +15% |
| Cloud Revenue | $2.1B | $1.64B | +28% |
| Hardware Revenue | $1.3B | $1.28B | +1.6% |
| Services Revenue | $800M | $714M | +12% |
| Gross Profit | $2.52B | $2.08B | +21% |
| Gross Margin | 60% | 57% | +300bps |
| EBITDA | $1.2B | $978M | +22.7% |
| EBITDA Margin | 28.6% | 26.8% | +180bps |
| Net Income | $756M | $612M | +23.5% |
| EPS (diluted) | $3.15 | $2.55 | +23.5% |
| Free Cash Flow | $890M | $725M | +22.8% |
| R&D Spending | $630M | $548M | +15% |

## Strategic Initiatives

### AI and Machine Learning
Acme invested $630 million in R&D during Q3, with approximately 40% ($252 million) directed toward AI and machine learning capabilities. Key milestones include:
- Launched "Acme AI Studio," an enterprise platform for building and deploying AI applications
- Released the Acme Foundation Model v2.0, a 70-billion parameter language model optimized for enterprise use cases
- Acquired NeuralPath Inc. for $180 million, adding 45 AI researchers and proprietary training data

### Geographic Expansion
International revenue grew 22% year-over-year to $1.47 billion (35% of total revenue):
- North America: $2.73B (65% of revenue)
- Europe: $840M (20% of revenue, up 18% YoY)
- Asia Pacific: $420M (10% of revenue, up 32% YoY)
- Rest of World: $210M (5% of revenue, up 28% YoY)

### Supply Chain
The company maintained a strong supply chain position with:
- Primary chip suppliers: NVIDIA (45%), Intel (30%), AMD (25%)
- Average component lead time reduced from 16 weeks to 12 weeks
- Strategic inventory buffer of $450 million in critical components

## Risk Factors

1. **Regulatory Risk:** The EU AI Act may impact Acme's AI product offerings in European markets. Compliance costs are estimated at $50-75 million annually.
2. **Competition:** Increasing competition from hyperscalers (AWS, Azure, GCP) in the cloud segment could pressure margins.
3. **Customer Concentration:** Top 10 customers represent 28% of total revenue, with the largest single customer at 6%.
4. **Talent Retention:** The competitive market for AI talent resulted in a 15% increase in engineering compensation costs.
5. **Currency Risk:** A 10% strengthening of the USD would reduce international revenue by approximately $147 million.

## Outlook

For Q4 2025, Acme expects:
- Revenue of $4.4-4.6 billion (12-17% YoY growth)
- EBITDA margin of 29-30%
- Cloud revenue growth of 25-30%
- Capital expenditure of $400-450 million, primarily for data center expansion

For full-year 2025, the company raised its guidance:
- Full-year revenue: $16.5-16.8 billion (previously $16.0-16.5 billion)
- Full-year EBITDA margin: 28-29%

*Note: All financial figures are in US dollars unless otherwise stated. Non-GAAP measures exclude stock-based compensation and acquisition-related costs.*
"""

ACME_Q2_REPORT = """
# Acme Corporation — Q2 2025 Earnings Report

## Executive Summary

Acme Corporation delivered solid second-quarter results with revenue of $3.9 billion, up 13% year-over-year. Cloud services continued to be the primary growth engine, while the hardware division showed early signs of recovery through its AI accelerator product line.

EBITDA was $1.05 billion with a margin of 26.9%, compared to $920 million and 26.2% in Q2 2024.

## Revenue Breakdown by Segment

### Cloud Services Division
Cloud revenue was $1.85 billion, up 25% year-over-year:
- Enterprise SaaS: $1.05 billion (up 30%)
- IaaS: $580 million (up 20%)
- Professional services: $220 million (up 12%)

Net new annual recurring revenue (ARR) additions were $320 million, the highest in company history. Customer churn rate improved to 3.2% from 4.1% a year ago.

### Hardware Division
Hardware revenue was $1.28 billion, down 2% year-over-year:
- Server and networking: $850 million (down 8%)
- AI accelerators: $280 million (up 155%)
- Storage: $150 million (down 10%)

### Services Division
Services revenue was $770 million, up 10% year-over-year:
- Managed IT: $430 million
- Cybersecurity: $195 million (up 22%)
- Training: $145 million

## Financial Highlights

| Metric | Q2 2025 | Q2 2024 | Change |
|--------|---------|---------|--------|
| Total Revenue | $3.9B | $3.45B | +13% |
| Cloud Revenue | $1.85B | $1.48B | +25% |
| Hardware Revenue | $1.28B | $1.31B | -2.3% |
| Services Revenue | $770M | $700M | +10% |
| EBITDA | $1.05B | $920M | +14.1% |
| EBITDA Margin | 26.9% | 26.2% | +70bps |
| Net Income | $672M | $580M | +15.9% |
| Free Cash Flow | $780M | $650M | +20% |

## Key Developments

### Product Launches
- **Acme CloudShield:** New cloud-native security platform, generating $45M in first-quarter bookings
- **Acme Edge AI Box:** On-premises AI inference device for manufacturing and retail, shipping to first 200 customers

### Partnerships
- Signed strategic partnership with TechGiant Corp for joint go-to-market in healthcare vertical
- Expanded relationship with GlobalBank Inc., adding AI-powered fraud detection ($25M annual contract)

### Workforce
- Total employees: 42,500 (up from 39,800 in Q2 2024)
- Engineering headcount: 18,200 (43% of workforce)
- New AI research lab opened in London with 120 researchers

## Guidance

For Q3 2025, Acme expects:
- Revenue of $4.1-4.3 billion
- EBITDA margin of 27.5-28.5%
- Cloud growth of 26-30%
"""

TECHGIANT_REPORT = """
# TechGiant Corporation — Annual Report 2024

## Company Overview

TechGiant Corporation is a multinational technology company specializing in enterprise software, cloud computing, and artificial intelligence solutions. Founded in 2005, TechGiant has grown to serve over 15,000 enterprise customers across 85 countries.

## Financial Performance FY2024

### Annual Revenue Summary
Total revenue for fiscal year 2024 was $12.8 billion, representing 18% growth over FY2023's $10.85 billion. This marks the fifth consecutive year of double-digit revenue growth.

| Metric | FY2024 | FY2023 | FY2022 | Change (YoY) |
|--------|--------|--------|--------|-------------|
| Total Revenue | $12.8B | $10.85B | $9.2B | +18% |
| Subscription Revenue | $8.96B | $7.17B | $5.75B | +25% |
| Professional Services | $2.56B | $2.39B | $2.21B | +7% |
| Hardware & Other | $1.28B | $1.29B | $1.24B | -0.8% |
| Gross Margin | 72% | 70% | 68% | +200bps |
| Operating Income | $2.82B | $2.17B | $1.66B | +30% |
| Net Income | $2.18B | $1.63B | $1.20B | +34% |
| Free Cash Flow | $3.05B | $2.28B | $1.75B | +34% |

### Revenue by Geography
- Americas: $7.68B (60%, up 16%)
- EMEA: $3.84B (30%, up 20%)
- APAC: $1.28B (10%, up 25%)

## Product Portfolio

### TechGiant Cloud Platform (TCP)
The flagship cloud platform grew 28% to $6.4 billion in annual revenue. TCP now hosts over 2,500 enterprise applications and processes 15 billion API calls daily.

Key metrics:
- 99.99% uptime SLA achievement
- Average customer contract value: $425,000/year (up from $380,000)
- 850 new enterprise customers added in FY2024

### TechGiant AI Suite
The AI product line, launched in Q2 2023, reached $1.2 billion in annual revenue:
- TG-Copilot (AI assistant for developers): $480M
- TG-Analytics (AI-powered business intelligence): $420M
- TG-Vision (computer vision for manufacturing): $180M
- TG-Language (NLP services for customer support): $120M

### Enterprise Security
Security products generated $1.8 billion, up 22%:
- Identity and access management: $720M
- Threat detection and response: $540M
- Data protection and compliance: $360M
- Cloud security posture management: $180M

## Strategic Investments

### Acquisitions
TechGiant completed three acquisitions in FY2024:
1. **DataForge Analytics** ($450M) — Adds real-time data pipeline capabilities
2. **SecureAI Labs** ($280M) — AI-powered threat detection technology
3. **CloudBridge Systems** ($190M) — Multi-cloud management platform

### R&D Investment
R&D spending was $2.56 billion (20% of revenue):
- AI and Machine Learning: $1.02B (40% of R&D)
- Cloud infrastructure: $640M (25%)
- Security: $512M (20%)
- Developer tools: $384M (15%)

The company holds 3,200 patents, with 480 new patents filed in FY2024.

## Competitive Landscape

TechGiant competes primarily with:
- **Acme Corporation:** Direct competitor in cloud and AI. TechGiant has larger cloud revenue but Acme leads in AI accelerator hardware.
- **GlobalSoft Inc.:** Stronger in traditional enterprise software but behind in cloud transition.
- **CloudFirst Ltd.:** Pure-play cloud competitor with faster growth but smaller scale.

Market share estimates (enterprise cloud):
- Hyperscalers (AWS/Azure/GCP): 65%
- TechGiant: 12%
- Acme Corporation: 8%
- Others: 15%

## Supplier Relationships

Key technology suppliers:
- NVIDIA: Primary GPU supplier for AI infrastructure (60% of AI compute)
- Intel: CPU supplier for cloud servers (45% of server CPUs)
- Samsung: Memory and storage components
- TSMC: Custom chip fabrication for TG-Accelerator chips

TechGiant is Acme Corporation's largest customer, accounting for approximately 6% of Acme's total revenue through infrastructure purchases.

## ESG and Corporate Responsibility

- Carbon neutral operations achieved in FY2024
- $150M invested in renewable energy for data centers
- 45% of leadership positions held by underrepresented groups
- $50M annual commitment to STEM education programs

## Outlook FY2025

- Revenue guidance: $14.5-15.0 billion (13-17% growth)
- Subscription revenue growth: 20-25%
- Operating margin target: 23-25%
- R&D investment: maintain at 20% of revenue
- Planned CapEx: $1.8-2.0 billion (data center expansion in APAC)
"""


def create_sample_documents(output_dir: str):
    """Create sample financial report documents for the RAG system."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    documents = {
        "acme_q3_2025_earnings.md": ACME_Q3_REPORT,
        "acme_q2_2025_earnings.md": ACME_Q2_REPORT,
        "techgiant_annual_2024.md": TECHGIANT_REPORT,
    }

    for filename, content in documents.items():
        filepath = output_path / filename
        filepath.write_text(content.strip(), encoding="utf-8")
        print(f"  ✅ Created: {filename} ({len(content):,} chars)")

    print(f"\n  📁 {len(documents)} documents created in {output_dir}")
    return list(documents.keys())


if __name__ == "__main__":
    # When run directly, create sample documents
    project_dir = Path(__file__).parent.parent
    sample_dir = project_dir / "data" / "sample_docs"
    print("Creating sample financial reports...\n")
    create_sample_documents(str(sample_dir))
    print("\n✅ Done! You can now run the ingestion pipeline.")
    print("\nTo also use real PDFs with charts, download financial")
    print("reports (10-K filings from SEC EDGAR) and place them")
    print(f"in: {sample_dir}")
