# output/pdf_generator.py

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.units import inch
import os
import re


def generate_pdf(report_text: str, references: str) -> str:
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

    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=11,
        leading=15,
        alignment=TA_LEFT,
        spaceAfter=10
    )

    heading_style = ParagraphStyle(
        "Heading",
        parent=styles["Normal"],
        fontSize=15,
        leading=20,
        spaceBefore=22,
        spaceAfter=14,
        fontName="Helvetica-Bold"
    )

    story = []

    lines = report_text.split("\n")

    for line in lines:
        line = line.strip()

        if not line:
            story.append(Spacer(1, 0.15 * inch))
            continue

        if line.startswith("## "):
            heading = line.replace("## ", "")
            story.append(Paragraph(heading, heading_style))
        else:
            story.append(Paragraph(line, body_style))

    # References page
    story.append(PageBreak())
    story.append(Paragraph("REFERENCES", heading_style))
    story.append(Spacer(1, 0.2 * inch))

    for ref in references.split("\n"):
        if ref.strip():
            story.append(Paragraph(ref, body_style))

    doc.build(story)
    return path
