# output/pdf_generator.py

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.units import inch
import os


def generate_pdf(report_text: str, references: str) -> str:
    """
    Generate a research-style PDF with:
    - Clear bold headings
    - Formal spacing
    - Academic visual hierarchy
    """

    os.makedirs("outputs", exist_ok=True)
    path = "outputs/final_report.pdf"

    doc = SimpleDocTemplate(
        path,
        pagesize=A4,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50
    )

    styles = getSampleStyleSheet()

    # ----------------------------
    # Body text style
    # ----------------------------
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=11,
        leading=15,
        alignment=TA_LEFT,
        spaceAfter=10
    )

    # ----------------------------
    # Heading style (MAIN SECTIONS)
    # ----------------------------
    heading_style = ParagraphStyle(
        "Heading",
        parent=styles["Normal"],
        fontSize=15,
        leading=20,
        spaceBefore=24,
        spaceAfter=14,
        fontName="Helvetica-Bold",
        wordWrap="CJK"   # keeps long headings visually balanced
    )

    story = []

    # ----------------------------
    # Render report content
    # ----------------------------
    for line in report_text.split("\n"):
        line = line.strip()

        if not line:
            story.append(Spacer(1, 0.15 * inch))
            continue

        # Markdown section heading
        if line.startswith("## "):
            heading_text = line.replace("## ", "")
            story.append(Paragraph(heading_text, heading_style))
        else:
            story.append(Paragraph(line, body_style))

    # ----------------------------
    # References page
    # ----------------------------
    story.append(PageBreak())
    story.append(Paragraph("REFERENCES", heading_style))
    story.append(Spacer(1, 0.2 * inch))

    for ref in references.split("\n"):
        if ref.strip():
            story.append(Paragraph(ref, body_style))

    doc.build(story)
    return path
