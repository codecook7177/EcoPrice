"""
EcoPrice - Email & Account Verification Service
Handles cryptographic token generation, verification, and email dispatching
via standard SMTP (compatible with Gmail App Passwords, SendGrid, Resend, etc.)
with a zero-configuration developer simulation fallback for local testing.
"""

import os
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadTimeSignature

logger = logging.getLogger("ecoprice.email")


class EmailService:
    def __init__(self):
        self.salt = "ecoprice-account-verification-salt"

    def _get_serializer(self, secret_key: str) -> URLSafeTimedSerializer:
        return URLSafeTimedSerializer(secret_key)

    def generate_token(self, email: str, secret_key: str) -> str:
        """Generates a secure, cryptographically signed URL-safe token."""
        serializer = self._get_serializer(secret_key)
        return serializer.dumps(email.strip().lower(), salt=self.salt)

    def verify_token(self, token: str, secret_key: str, max_age_seconds: int = 86400) -> Optional[str]:
        """
        Verifies the cryptographic token.
        Returns the email address if valid and unexpired (default 24 hours).
        Returns None if expired, tampered with, or invalid.
        """
        serializer = self._get_serializer(secret_key)
        try:
            email = serializer.loads(token, salt=self.salt, max_age=max_age_seconds)
            return email.strip().lower()
        except (SignatureExpired, BadTimeSignature, Exception) as e:
            logger.warning(f"[EmailService] Token verification failed: {e}")
            return None

    def is_smtp_configured(self) -> bool:
        """Checks if real SMTP credentials are present in the environment."""
        username = (os.environ.get("SMTP_USERNAME") or os.environ.get("MAIL_USERNAME") or "").strip()
        password = (os.environ.get("SMTP_PASSWORD") or os.environ.get("MAIL_PASSWORD") or "").strip()
        return bool(username and password)

    def send_confirmation_email(
        self,
        to_email: str,
        manager_name: str,
        confirm_url: str
    ) -> Tuple[bool, str]:
        """
        Dispatches a verification email.
        If SMTP environment variables are configured, sends via SMTP.
        If SMTP is not configured, logs to console and returns simulation status.
        """
        smtp_user = (os.environ.get("SMTP_USERNAME") or os.environ.get("MAIL_USERNAME") or "").strip()
        smtp_pass = (os.environ.get("SMTP_PASSWORD") or os.environ.get("MAIL_PASSWORD") or "").strip().replace(" ", "")
        smtp_host = (os.environ.get("SMTP_SERVER") or "smtp.gmail.com").strip()
        smtp_port = int(os.environ.get("SMTP_PORT", 587))
        sender_name = os.environ.get("MAIL_SENDER_NAME", "EcoPrice Intelligence")
        sender_email = (os.environ.get("MAIL_SENDER_EMAIL") or smtp_user or "noreply@ecoprice.local").strip()

        subject = "Confirm Your EcoPrice Store Manager Account"
        
        # Plain text fallback
        text_body = f"""Hello {manager_name},

Thank you for registering your retail store with EcoPrice.

To activate your account and access the store management dashboard, please confirm your email address by opening the link below:

{confirm_url}

This verification link will expire in 24 hours.

If you did not create an account on EcoPrice, you can safely ignore this message.

EcoPrice Automated Verification System
"""

        # Aesthetic HTML Body (matches EcoPrice typography and branding)
        html_body = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{
    font-family: 'Space Grotesk', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
    margin: 0;
    padding: 30px 15px;
  }}
  .email-card {{
    max-width: 540px;
    margin: 0 auto;
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 32px 28px;
  }}
  .badge {{
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.12em;
    color: #10b981;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.3);
    padding: 3px 8px;
    border-radius: 3px;
    margin-bottom: 18px;
  }}
  h1 {{
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 24px;
    color: #ffffff;
    margin: 0 0 14px 0;
    font-weight: 700;
  }}
  p {{
    font-size: 14px;
    line-height: 1.6;
    color: #8b949e;
    margin: 0 0 20px 0;
  }}
  .btn-confirm {{
    display: inline-block;
    background-color: #10b981;
    color: #ffffff !important;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-decoration: none;
    padding: 13px 26px;
    border-radius: 4px;
    margin: 12px 0 24px 0;
  }}
  .url-box {{
    word-break: break-all;
    background: rgba(0, 0, 0, 0.3);
    border: 1px solid #30363d;
    padding: 10px 12px;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: #79c0ff;
    border-radius: 4px;
    margin-top: 15px;
  }}
  .footer {{
    border-top: 1px solid #21262d;
    margin-top: 28px;
    padding-top: 16px;
    font-size: 11px;
    color: #6e7681;
    font-family: 'Space Mono', monospace;
  }}
</style>
</head>
<body>
  <div class="email-card">
    <div class="badge">ECOPRICE.IDENTITY // VERIFICATION</div>
    <h1>Confirm Your Store Account</h1>
    <p>Hello {manager_name},</p>
    <p>Thank you for registering with EcoPrice. To finalize your account setup and access real-time demand forecasts and dynamic pricing tools, please confirm your email address below.</p>
    <div>
      <a href="{confirm_url}" class="btn-confirm" target="_blank">CONFIRM ACCOUNT &rarr;</a>
    </div>
    <p style="font-size: 12px; margin-bottom: 6px;">Or paste this link into your browser:</p>
    <div class="url-box">{confirm_url}</div>
    <div class="footer">
      This link is valid for 24 hours. If you did not create an account on EcoPrice, no action is needed.
    </div>
  </div>
</body>
</html>"""

        if not (smtp_user and smtp_pass):
            # Development / Simulation Mode
            print(f"\n" + "=" * 70)
            print(f"[EmailService SIMULATION MODE]")
            print(f"Recipient: {to_email} ({manager_name})")
            print(f"Subject:   {subject}")
            print(f"Confirm Link: {confirm_url}")
            print("=" * 70 + "\n")
            return False, confirm_url

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = f"{sender_name} <{sender_email}>"
            msg["To"] = to_email

            part1 = MIMEText(text_body, "plain", "utf-8")
            part2 = MIMEText(html_body, "html", "utf-8")
            msg.attach(part1)
            msg.attach(part2)

            if smtp_port == 465:
                with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=12) as server:
                    server.login(smtp_user, smtp_pass)
                    server.send_message(msg)
            else:
                with smtplib.SMTP(smtp_host, smtp_port, timeout=12) as server:
                    server.starttls()
                    server.login(smtp_user, smtp_pass)
                    server.send_message(msg)

            logger.info(f"[EmailService] Real confirmation email sent to {to_email}")
            return True, "Email dispatched to recipient's inbox."
        except Exception as e:
            logger.error(f"[EmailService] Failed to send email via SMTP: {e}")
            # Fall back to simulation return so the user is never blocked
            return False, confirm_url


email_service = EmailService()
