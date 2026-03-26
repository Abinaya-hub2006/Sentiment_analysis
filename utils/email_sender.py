import smtplib
from email.message import EmailMessage
import os
from dotenv import load_dotenv

# load env
load_dotenv()

def send_email(receiver_email, pdf_path):

    sender_email = os.getenv("EMAIL")
    app_password = os.getenv("APP_PASSWORD")

    msg = EmailMessage()
    msg["Subject"] = "Sentiment Analysis Report"
    msg["From"] = sender_email
    msg["To"] = receiver_email

    msg.set_content("Your sentiment analysis report is attached.")

    # attach PDF
    with open(pdf_path, "rb") as f:
        file_data = f.read()

    msg.add_attachment(
        file_data,
        maintype="application",
        subtype="pdf",
        filename="report.pdf"
    )

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(sender_email, app_password)
            smtp.send_message(msg)

        return True

    except Exception as e:
        print("Email Error:", e)
        return False