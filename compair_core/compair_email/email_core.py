import os

from redmail import EmailSender

EMAIL_HOST = f"{os.environ.get('EMAIL_HOST')}"
EMAIL_USER = f"{os.environ.get('EMAIL_USER')}"
EMAIL_PW = f"{os.environ.get('EMAIL_PW')}"

emailer = EmailSender(
    host=EMAIL_HOST,  
    port=587, 
    username=EMAIL_USER,
    password=EMAIL_PW,
    use_starttls=True 
)
