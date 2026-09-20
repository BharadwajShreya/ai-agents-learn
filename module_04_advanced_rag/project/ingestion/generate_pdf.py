"""
Generate Sample PDF with Charts and Tables
===========================================

Creates a realistic financial report PDF with:
  - Text paragraphs
  - Data tables
  - Bar charts and line charts
  - Multiple pages

This lets us test the multimodal pipeline immediately without
downloading external PDFs. Run this AFTER installing requirements.

Usage: python -m ingestion.generate_pdf
"""

import sys
from pathlib import Path

# Ensure Windows stdout handles UTF-8 characters and emojis properly
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_financial_report_pdf(output_path: str):
    """Generate a multi-page financial report PDF with charts and tables."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("❌ pymupdf required. Run: pip install pymupdf")
        return

    try:
        import matplotlib
        matplotlib.use('Agg')  # non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        import io
    except ImportError:
        print("❌ matplotlib required. Run: pip install matplotlib")
        return

    doc = fitz.open()

    # =========================================================================
    # PAGE 1: Title + Executive Summary
    # =========================================================================
    page = doc.new_page(width=595, height=842)  # A4

    # Title
    page.insert_text(
        fitz.Point(50, 60),
        "Acme Corporation",
        fontname="helv", fontsize=24, color=(0.1, 0.2, 0.5)
    )
    page.insert_text(
        fitz.Point(50, 85),
        "Q3 2025 Earnings Report",
        fontname="helv", fontsize=16, color=(0.3, 0.3, 0.3)
    )

    # Line separator
    page.draw_line(fitz.Point(50, 95), fitz.Point(545, 95),
                   color=(0.1, 0.2, 0.5), width=2)

    # Executive summary text
    summary = (
        "Acme Corporation reported strong third-quarter results for fiscal year 2025, "
        "with revenue reaching $4.2 billion, representing a 15% increase year-over-year. "
        "The company's strategic investments in cloud infrastructure and AI-powered services "
        "continued to drive growth across all major business segments.\n\n"
        "EBITDA for the quarter was $1.2 billion, with an EBITDA margin of 28.6%, expanding "
        "180 basis points from Q3 2024. This margin improvement reflects the company's "
        "ongoing focus on operational efficiency and the higher-margin nature of its growing "
        "cloud business.\n\n"
        "Key highlights:\n"
        "• Cloud services revenue grew 28% YoY to $2.1 billion\n"
        "• AI accelerator chip revenue surged 180% to $350 million\n"
        "• Free cash flow of $890 million, up 22.8% YoY\n"
        "• Raised full-year revenue guidance to $16.5-16.8 billion"
    )
    rect = fitz.Rect(50, 110, 545, 400)
    page.insert_textbox(rect, summary, fontname="helv", fontsize=10,
                        color=(0.2, 0.2, 0.2))

    # Revenue bar chart
    fig, ax = plt.subplots(figsize=(5, 3))
    quarters = ['Q1 2024', 'Q2 2024', 'Q3 2024', 'Q4 2024', 'Q1 2025', 'Q2 2025', 'Q3 2025']
    revenue = [3.1, 3.45, 3.65, 3.8, 3.75, 3.9, 4.2]
    colors = ['#1a365d'] * 4 + ['#2b6cb0'] * 3
    ax.bar(quarters, revenue, color=colors, width=0.6)
    ax.set_ylabel('Revenue ($B)', fontsize=9)
    ax.set_title('Quarterly Revenue Trend', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 5)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('$%.1fB'))
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.tight_layout()

    # Save chart to bytes and insert into PDF
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)

    img_rect = fitz.Rect(50, 420, 545, 720)
    page.insert_image(img_rect, stream=buf.read())

    page.insert_text(
        fitz.Point(50, 740),
        "Figure 1: Quarterly Revenue Trend (FY2024-FY2025)",
        fontname="helv", fontsize=8, color=(0.4, 0.4, 0.4)
    )

    # =========================================================================
    # PAGE 2: Revenue Breakdown + Pie Chart
    # =========================================================================
    page2 = doc.new_page(width=595, height=842)

    page2.insert_text(
        fitz.Point(50, 50),
        "Revenue Breakdown by Segment",
        fontname="helv", fontsize=16, color=(0.1, 0.2, 0.5)
    )
    page2.draw_line(fitz.Point(50, 60), fitz.Point(545, 60),
                    color=(0.1, 0.2, 0.5), width=1)

    # Table data
    table_text = (
        "Metric                    Q3 2025      Q3 2024      Change\n"
        "─────────────────────────────────────────────────────────────\n"
        "Total Revenue             $4.2B        $3.65B       +15.1%\n"
        "Cloud Services            $2.1B        $1.64B       +28.0%\n"
        "  Enterprise SaaS         $1.2B        $889M        +35.0%\n"
        "  IaaS                    $650M        $533M        +22.0%\n"
        "  Professional Services   $250M        $217M        +15.0%\n"
        "Hardware                  $1.3B        $1.28B       +1.6%\n"
        "  Server & Networking     $800M        $842M        -5.0%\n"
        "  AI Accelerators         $350M        $125M        +180%\n"
        "  Storage Solutions       $150M        $170M        -12.0%\n"
        "Services                  $800M        $714M        +12.0%\n"
        "  Managed IT              $450M        $417M        +8.0%\n"
        "  Cybersecurity           $200M        $160M        +25.0%\n"
        "  Training                $150M        $136M        +10.0%\n"
        "─────────────────────────────────────────────────────────────\n"
        "Gross Profit              $2.52B       $2.08B       +21.2%\n"
        "Gross Margin              60.0%        57.0%        +300bps\n"
        "EBITDA                    $1.2B        $978M        +22.7%\n"
        "EBITDA Margin             28.6%        26.8%        +180bps\n"
        "Net Income                $756M        $612M        +23.5%\n"
        "EPS (diluted)             $3.15        $2.55        +23.5%\n"
    )

    rect2 = fitz.Rect(50, 75, 545, 380)
    page2.insert_textbox(rect2, table_text, fontname="cour", fontsize=8,
                         color=(0.1, 0.1, 0.1))

    page2.insert_text(
        fitz.Point(50, 390),
        "Table 1: Detailed Financial Summary Q3 2025 vs Q3 2024",
        fontname="helv", fontsize=8, color=(0.4, 0.4, 0.4)
    )

    # Pie chart — revenue by segment
    fig2, ax2 = plt.subplots(figsize=(4.5, 3))
    segments = ['Cloud\n$2.1B', 'Hardware\n$1.3B', 'Services\n$800M']
    sizes = [2.1, 1.3, 0.8]
    colors_pie = ['#2b6cb0', '#4a5568', '#38a169']
    explode = (0.05, 0, 0)
    wedges, texts, autotexts = ax2.pie(
        sizes, labels=segments, colors=colors_pie,
        autopct='%1.1f%%', startangle=90, explode=explode,
        textprops={'fontsize': 8}
    )
    ax2.set_title('Q3 2025 Revenue by Segment', fontsize=11, fontweight='bold')
    plt.tight_layout()

    buf2 = io.BytesIO()
    fig2.savefig(buf2, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    buf2.seek(0)

    img_rect2 = fitz.Rect(100, 410, 500, 680)
    page2.insert_image(img_rect2, stream=buf2.read())

    page2.insert_text(
        fitz.Point(50, 695),
        "Figure 2: Revenue Distribution by Business Segment",
        fontname="helv", fontsize=8, color=(0.4, 0.4, 0.4)
    )

    # =========================================================================
    # PAGE 3: Growth Trends + Line Chart
    # =========================================================================
    page3 = doc.new_page(width=595, height=842)

    page3.insert_text(
        fitz.Point(50, 50),
        "Growth Trends & Strategic Outlook",
        fontname="helv", fontsize=16, color=(0.1, 0.2, 0.5)
    )
    page3.draw_line(fitz.Point(50, 60), fitz.Point(545, 60),
                    color=(0.1, 0.2, 0.5), width=1)

    # Multi-line chart — segment growth
    fig3, ax3 = plt.subplots(figsize=(5, 3.5))
    quarters_short = ['Q1\'24', 'Q2\'24', 'Q3\'24', 'Q4\'24', 'Q1\'25', 'Q2\'25', 'Q3\'25']
    cloud = [1.35, 1.48, 1.64, 1.72, 1.75, 1.85, 2.1]
    hardware = [1.25, 1.31, 1.28, 1.30, 1.25, 1.28, 1.3]
    services = [0.60, 0.70, 0.714, 0.73, 0.75, 0.77, 0.8]

    ax3.plot(quarters_short, cloud, 'o-', color='#2b6cb0', linewidth=2, label='Cloud', markersize=5)
    ax3.plot(quarters_short, hardware, 's-', color='#4a5568', linewidth=2, label='Hardware', markersize=5)
    ax3.plot(quarters_short, services, '^-', color='#38a169', linewidth=2, label='Services', markersize=5)

    ax3.set_ylabel('Revenue ($B)', fontsize=9)
    ax3.set_title('Revenue by Segment — Quarterly Trend', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=8, loc='upper left')
    ax3.set_ylim(0, 2.5)
    ax3.yaxis.set_major_formatter(mticker.FormatStrFormatter('$%.1fB'))
    ax3.grid(True, alpha=0.3)
    plt.xticks(fontsize=7)
    plt.tight_layout()

    buf3 = io.BytesIO()
    fig3.savefig(buf3, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig3)
    buf3.seek(0)

    img_rect3 = fitz.Rect(50, 75, 545, 380)
    page3.insert_image(img_rect3, stream=buf3.read())

    page3.insert_text(
        fitz.Point(50, 395),
        "Figure 3: Revenue by Segment — Quarterly Trend",
        fontname="helv", fontsize=8, color=(0.4, 0.4, 0.4)
    )

    # Strategic text
    strategy_text = (
        "Strategic Outlook\n\n"
        "Cloud services remain the primary growth engine, now representing 50% of total "
        "revenue compared to 45% a year ago. The company expects cloud to reach 55% of "
        "revenue by FY2026 as enterprise digital transformation accelerates.\n\n"
        "The AI accelerator business, while still small at $350M quarterly revenue, is "
        "growing at 180% YoY and represents the highest-margin hardware product. The company "
        "plans to double production capacity by Q2 2026.\n\n"
        "Risk factors include increasing competition from hyperscalers (AWS, Azure, GCP), "
        "regulatory uncertainty around the EU AI Act (estimated $50-75M annual compliance cost), "
        "and customer concentration (top 10 customers = 28% of revenue).\n\n"
        "For Q4 2025, management guides for revenue of $4.4-4.6 billion with EBITDA margin "
        "of 29-30%. Full-year 2025 guidance has been raised to $16.5-16.8 billion."
    )

    rect3 = fitz.Rect(50, 415, 545, 750)
    page3.insert_textbox(rect3, strategy_text, fontname="helv", fontsize=10,
                         color=(0.2, 0.2, 0.2))

    # =========================================================================
    # PAGE 4: EBITDA Margin + Geographic Revenue
    # =========================================================================
    page4 = doc.new_page(width=595, height=842)

    page4.insert_text(
        fitz.Point(50, 50),
        "Profitability & Geographic Analysis",
        fontname="helv", fontsize=16, color=(0.1, 0.2, 0.5)
    )
    page4.draw_line(fitz.Point(50, 60), fitz.Point(545, 60),
                    color=(0.1, 0.2, 0.5), width=1)

    # EBITDA margin chart
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(5.5, 2.8))

    # Left: EBITDA margin trend
    margins = [24.5, 26.2, 26.8, 27.1, 26.5, 26.9, 28.6]
    ax4a.bar(quarters_short, margins, color='#2b6cb0', width=0.5)
    ax4a.set_ylabel('EBITDA Margin (%)', fontsize=8)
    ax4a.set_title('EBITDA Margin Trend', fontsize=10, fontweight='bold')
    ax4a.set_ylim(20, 32)
    ax4a.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    plt.setp(ax4a.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=6)

    # Right: Geographic revenue pie
    geo_labels = ['N. America\n$2.73B', 'Europe\n$840M', 'APAC\n$420M', 'RoW\n$210M']
    geo_sizes = [2.73, 0.84, 0.42, 0.21]
    geo_colors = ['#2b6cb0', '#4299e1', '#63b3ed', '#bee3f8']
    ax4b.pie(geo_sizes, labels=geo_labels, colors=geo_colors,
             autopct='%1.0f%%', startangle=90, textprops={'fontsize': 7})
    ax4b.set_title('Revenue by Geography', fontsize=10, fontweight='bold')

    plt.tight_layout()

    buf4 = io.BytesIO()
    fig4.savefig(buf4, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig4)
    buf4.seek(0)

    img_rect4 = fitz.Rect(30, 75, 565, 340)
    page4.insert_image(img_rect4, stream=buf4.read())

    page4.insert_text(
        fitz.Point(50, 355),
        "Figure 4: EBITDA Margin Trend and Geographic Revenue Distribution",
        fontname="helv", fontsize=8, color=(0.4, 0.4, 0.4)
    )

    # Supplier info text
    supplier_text = (
        "Supply Chain & Key Suppliers\n\n"
        "The company maintains strategic supplier relationships to ensure component "
        "availability and competitive pricing:\n\n"
        "• NVIDIA: 45% of AI compute chips (primary GPU supplier)\n"
        "• Intel: 30% of server CPUs\n"
        "• AMD: 25% of server CPUs\n\n"
        "Average component lead time has been reduced from 16 weeks to 12 weeks "
        "through improved supply chain management and strategic inventory buffers "
        "of $450 million in critical components.\n\n"
        "TechGiant Corporation remains Acme's largest customer, accounting for "
        "approximately 6% of total revenue through infrastructure purchases. "
        "Top 10 customers represent 28% of total revenue."
    )

    rect4 = fitz.Rect(50, 375, 545, 650)
    page4.insert_textbox(rect4, supplier_text, fontname="helv", fontsize=10,
                         color=(0.2, 0.2, 0.2))

    # Footer
    page4.insert_text(
        fitz.Point(50, 780),
        "Note: All financial figures in US dollars. Non-GAAP measures exclude "
        "stock-based compensation and acquisition-related costs.",
        fontname="helv", fontsize=7, color=(0.5, 0.5, 0.5)
    )

    # Save PDF
    doc.save(output_path)
    doc.close()
    print(f"  ✅ Generated PDF: {output_path}")
    print(f"     Pages: 4, Charts: 5, Tables: 1")


if __name__ == "__main__":
    output_dir = Path(__file__).parent.parent / "data" / "sample_docs"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = str(output_dir / "acme_q3_2025_earnings_report.pdf")

    print("=" * 60)
    print("  GENERATING SAMPLE PDF WITH CHARTS")
    print("=" * 60)
    print()

    generate_financial_report_pdf(output_file)

    print("\n✅ PDF ready for multimodal RAG testing!")
    print(f"   Open it to see: bar charts, line charts, pie charts, and tables")
