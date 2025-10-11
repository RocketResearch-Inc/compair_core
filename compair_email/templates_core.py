"""Minimal email templates for the core edition."""

ACCOUNT_VERIFY_TEMPLATE = """
<p>Hi {{user_name}},</p>
<p>Please verify your Compair account by clicking the link below:</p>
<p><a href="{{verify_link}}">Verify my account</a></p>
<p>Thanks!</p>
""".strip()

PASSWORD_RESET_TEMPLATE = """
<p>We received a request to reset your password.</p>
<p>Your password reset code is: <strong>{{reset_code}}</strong></p>
""".strip()
