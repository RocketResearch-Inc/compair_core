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

GROUP_INVITATION_TEMPLATE = """
<p>{{inviter_name}} invited you to join the group {{group_name}}.</p>
<p><a href="{{invitation_link}}">Accept invitation</a></p>
""".strip()

GROUP_JOIN_TEMPLATE = """
<p>{{user_name}} has joined your group.</p>
""".strip()

INDIVIDUAL_INVITATION_TEMPLATE = """
<p>{{inviter_name}} invited you to Compair.</p>
<p><a href="{{referral_link}}">Join now</a></p>
""".strip()

REFERRAL_CREDIT_TEMPLATE = """
<p>Hi {{user_name}},</p>
<p>Great news! You now have {{referral_credits}} referral credits.</p>
""".strip()
