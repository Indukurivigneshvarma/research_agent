from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.units import inch
import re
import os


def _format_text(text: str):
    """
    Convert markdown-style headings into
    BOLD + UPPERCASE headings for PDF.
    """
    lines = text.split("\n")
    formatted = []

    for line in lines:
        line = line.strip()

        # Markdown headings → bold uppercase
        if re.match(r"^#{1,3}\s+", line):
            heading = re.sub(r"^#{1,3}\s+", "", line)
            formatted.append(f"<b>{heading.upper()}</b>")
            formatted.append("")  # spacing
        else:
            formatted.append(line)

    return "\n".join(formatted)


def generate_pdf(report_text: str, references: str, confidence_text: str):
    os.makedirs("outputs", exist_ok=True)
    file_path = "outputs/final_report.pdf"

    doc = SimpleDocTemplate(
        file_path,
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

    story = []

    # ---- MAIN REPORT ----
    formatted_report = _format_text(report_text)

    for block in formatted_report.split("\n\n"):
        if block.strip():
            story.append(Paragraph(block, body_style))
            story.append(Spacer(1, 0.15 * inch))

    # ---- CONFIDENCE / LIMITATIONS ----
    if confidence_text:
        story.append(PageBreak())
        story.append(Paragraph("<b>CONFIDENCE & LIMITATIONS</b>", body_style))
        story.append(Spacer(1, 0.2 * inch))

        for line in confidence_text.split("\n"):
            if line.strip():
                story.append(Paragraph(line, body_style))

    # ---- REFERENCES PAGE ----
    story.append(PageBreak())
    story.append(Paragraph("<b>REFERENCES</b>", body_style))
    story.append(Spacer(1, 0.2 * inch))

    for ref in references.split("\n"):
        if ref.strip():
            story.append(Paragraph(ref, body_style))

    doc.build(story)

    return file_path
