import os
from datetime import datetime

from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

REPORT_FOLDER = "reports"

os.makedirs(REPORT_FOLDER, exist_ok=True)


def generate_pdf_report(image_name, prediction, confidence, report):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"report_{timestamp}.pdf"

    filepath = os.path.join(REPORT_FOLDER, filename)

    doc = SimpleDocTemplate(filepath)

    styles = getSampleStyleSheet()

    story = []

    story.append(Paragraph("<b>AI Medical Report Assistant</b>", styles["Title"]))

    story.append(Paragraph("<br/>", styles["BodyText"]))

    story.append(Paragraph(f"<b>Image Name:</b> {image_name}", styles["BodyText"]))

    story.append(Paragraph(f"<b>Prediction:</b> {prediction}", styles["BodyText"]))

    story.append(Paragraph(f"<b>Confidence:</b> {confidence:.2f}%", styles["BodyText"]))

    story.append(Paragraph("<br/>", styles["BodyText"]))

    report = report.replace("\n", "<br/>")

    story.append(Paragraph(report, styles["BodyText"]))

    story.append(Paragraph("<br/>", styles["BodyText"]))

    story.append(
        Paragraph(
            "<b>Disclaimer:</b> This report is AI-generated and is not a medical diagnosis.",
            styles["BodyText"]
        )
    )

    doc.build(story)

    return filepath