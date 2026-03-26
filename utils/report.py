from fpdf import FPDF
import os

def generate_pdf():
    os.makedirs("outputs/reports", exist_ok=True)

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, "Sentiment Analysis Report", ln=True)

    pdf.image("outputs/images/pie.png", x=10, y=30, w=100)
    pdf.image("outputs/images/bar.png", x=10, y=100, w=100)

    pdf.output("outputs/reports/report.pdf")