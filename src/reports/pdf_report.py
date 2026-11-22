from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

def customer_pdf(customer, probability, risk, reasons, actions):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("RetainAI — Customer Retention Brief", styles["Title"]),
        Spacer(1, 12),
        Paragraph(f"Modelled churn probability: {probability:.1%}", styles["Heading2"]),
        Paragraph(f"Risk band: {risk}", styles["Normal"]),
        Spacer(1, 12),
        Paragraph("Customer profile", styles["Heading2"]),
    ]
    rows = [["Feature", "Value"]] + [[str(k), str(v)] for k, v in customer.items()]
    table = Table(rows, repeatRows=1, colWidths=[180, 300])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1f4e79")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
    ]))
    story += [table, Spacer(1, 14), Paragraph("Model explanation (SHAP)", styles["Heading2"])]
    for item in reasons:
        story.append(Paragraph(f"• {item['text']} (SHAP: {item['shap_value']:.4f})", styles["BodyText"]))
    story += [Spacer(1, 12), Paragraph("Suggested actions — review before use", styles["Heading2"])]
    for item in actions:
        story.append(Paragraph(f"• {item['action']} — {item['reason']}", styles["BodyText"]))
    story += [Spacer(1, 12), Paragraph(
        "This report explains model behaviour, not causation. Suggested actions are hypotheses. "
        "Probability estimates are not guarantees.", styles["Italic"]
    )]
    doc.build(story)
    buffer.seek(0)
    return buffer

def batch_pdf(summary_rows):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [Paragraph("RetainAI — Batch Prediction Summary", styles["Title"]), Spacer(1, 12)]
    rows = [["Risk band", "Customers"]] + [[str(k), str(v)] for k,v in summary_rows.items()]
    table = Table(rows, colWidths=[240, 180])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1f4e79")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
    ]))
    story.append(table)
    doc.build(story)
    buffer.seek(0)
    return buffer
