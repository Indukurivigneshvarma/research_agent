from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch


def generate_pdf(
    report_text: str,
    output_path: str = "report.pdf",
) -> str:
    """
    Generates a clean academic-style PDF.

    Expected markers in report_text:
    - @@TITLE@@
    - @@Section Heading@@
    """

    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="TitleStyle",
            fontSize=18,
            leading=22,
            spaceBefore=24,
            spaceAfter=24,
            alignment=TA_CENTER,
            fontName="Helvetica-Bold",
        )
    )

    styles.add(
        ParagraphStyle(
            name="SectionHeader",
            fontSize=13,
            leading=16,
            spaceBefore=18,
            spaceAfter=12,
            fontName="Helvetica-Bold",
        )
    )

    styles.add(
        ParagraphStyle(
            name="ReportBody",
            fontSize=11,
            leading=15,
            spaceBefore=6,
            spaceAfter=6,
        )
    )

    doc = SimpleDocTemplate(
        output_path,
        pagesize=LETTER,
        rightMargin=1 * inch,
        leftMargin=1 * inch,
        topMargin=1 * inch,
        bottomMargin=1 * inch,
    )

    story = []

    lines = report_text.splitlines()
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if not line:
            story.append(Spacer(1, 12))
            i += 1
            continue

        # ---------- TITLE ----------
        if line == "@@TITLE@@":
            title_text = lines[i + 1].strip()
            story.append(Paragraph(title_text, styles["TitleStyle"]))
            story.append(Spacer(1, 24))
            i += 3  # skip @@TITLE@@, title, @@TITLE@@
            continue

        # ---------- SECTION HEADER ----------
        if line.startswith("@@") and line.endswith("@@"):
            heading = line.strip("@")
            story.append(Paragraph(heading, styles["SectionHeader"]))
            story.append(Spacer(1, 12))
            i += 1
            continue

        # ---------- BODY PARAGRAPH ----------
        story.append(Paragraph(line, styles["ReportBody"]))
        i += 1

    doc.build(story)
    return output_path
