import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from dotenv import load_dotenv

load_dotenv()
def send_email(content):
    message = Mail(
        from_email=os.getenv("MAIL_FROM"),
        to_emails=os.getenv("MAIL_TO"),
        subject= "Weekly Chatbot Summary",
        html_content=content)
    
    try:
        sg = SendGridAPIClient(os.environ.get('SENDGRID_API_KEY'))
        response = sg.send(message)
        print(response.status_code)
        print(response.body)
        print(response.headers)
    except Exception as e:
        print(e.body)