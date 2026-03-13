import random, json, sys

# =========================================================
# PHISHING EMAIL TEMPLATES — Structured, multi-line format
# =========================================================

PHISHING_EMAIL_TEMPLATES = [
    # 1. IT Security Alert
    (
        "IT Security Alert – Immediate Action Required",
        """Dear {name},

Our IT Security monitoring system has flagged your account for unusual login activity from an unrecognized device. To protect your data and prevent unauthorized access, we have temporarily restricted access to certain features.

Please verify your identity and restore full access by clicking the link below within the next 24 hours. Failure to do so may result in a temporary account suspension pending manual review.

Click here to verify your account now:
https://corp-it-security-verify.net/auth?token={token}

Regards,
IT Security Operations
{dept} Department""",
        "IT Security Alert"
    ),
    # 2. Payroll Issue
    (
        "Urgent: Payroll Information Update Required",
        """Dear {name},

Our payroll processing team has identified a discrepancy with your direct deposit information on file. As a result, your upcoming paycheck scheduled for this Friday may be delayed unless the information is verified and corrected.

To avoid any disruption to your payment, please log in to the employee payroll portal immediately and confirm your banking details. This process takes less than two minutes to complete.

Update your payroll information here:
https://payroll-employee-portal-secure.com/verify?id={token}

Best regards,
Payroll & Compensation Team
{dept} Department""",
        "Payroll Issue"
    ),
    # 3. HR Compliance Deadline
    (
        "Action Required: Annual Compliance Training Deadline",
        """Dear {name},

Our records indicate that you have not yet completed this year's mandatory compliance training, which is required for all employees per company policy. The deadline to complete this training is end of business this Friday.

Employees who do not complete the training by the deadline may have their system access suspended pending completion. The training takes approximately 20 minutes and can be done from your workstation.

Please access the training portal here and complete your certification:
https://hr-compliance-training-portal.net/login?user={token}

Kind regards,
Human Resources Department
{dept} Team""",
        "HR Compliance"
    ),
    # 4. Account Login Warning
    (
        "Warning: New Sign-In to Your Corporate Account",
        """Dear {name},

We detected a sign-in to your corporate account from a location and device that does not match your usual activity. This sign-in occurred today and may indicate that your credentials have been compromised.

If this was you, no action is necessary. However, if you do not recognize this activity, you must reset your password immediately to secure your account and prevent unauthorized access to sensitive company data.

Reset your password immediately using the secure link below:
https://account-security-corp-reset.net/recover?token={token}

Security Team,
Corporate IT & Cybersecurity
{dept} Department""",
        "Account Login Warning"
    ),
    # 5. Package / Delivery Scam
    (
        "Your Package Could Not Be Delivered – Action Needed",
        """Dear {name},

We attempted to deliver a package addressed to you today but were unable to complete the delivery due to an incomplete address on file. Your package is currently being held at our regional distribution center.

To reschedule delivery and confirm your shipping address, please visit the link below and update your information within 48 hours. After this window, the package will be returned to the sender.

Confirm your delivery details here:
https://delivery-management-track.com/reschedule?ref={token}

Thank you for your prompt attention.

Sincerely,
Delivery Services Team
{dept} Logistics""",
        "Delivery Notification"
    ),
]

# =========================================================
# LEGITIMATE EMAIL TEMPLATES — Structured, professional
# =========================================================

LEGIT_EMAIL_TEMPLATES = [
    # 1. Meeting Recap
    (
        "Summary and Next Steps from Today's Planning Meeting",
        """Hi {name},

Thank you all for a productive session this morning as we aligned on our upcoming project milestones. I've compiled the key takeaways and action items from our discussion for everyone's reference.

Please review your assigned tasks and ensure they are added to the project tracker by Wednesday. If you encounter any blockers or require additional resources, don't hesitate to reach out and we can schedule a quick sync.

Looking forward to a strong sprint ahead!

Best regards,
{sender}
{dept} Team"""
    ),
    # 2. Report Share
    (
        "Monthly Performance Report – {dept} Department",
        """Hi {name},

Please find attached the monthly performance report for the {dept} department covering the previous month's KPIs and operational metrics. The report highlights areas of strong performance as well as opportunities for improvement we should address this quarter.

I'd appreciate your review of the figures and any initial thoughts before our leadership review meeting next Thursday. Section 3 in particular outlines the budget variance that we'll need to discuss.

Let me know if you have any questions or would like to walk through the data together.

Warm regards,
{sender}
{dept} Analytics Team"""
    ),
    # 3. Project Check-in
    (
        "Quick Check-In: {dept} Project Status Update",
        """Hi {name},

I wanted to touch base on the current status of the project and ensure we're still aligned on the deliverables for this sprint. Based on last week's stand-up, a few items were flagged as at risk due to resource constraints.

Can you send over a quick update on where things stand with your tasks by end of day today? It would be great to have a clear picture before our stakeholder call tomorrow morning so we can address any concerns proactively.

Thanks in advance for the update — really appreciate your hard work on this!

Best,
{sender}
{dept} Project Office"""
    ),
    # 4. Onboarding Welcome
    (
        "Welcome to the Team! Your Onboarding Details Inside",
        """Hi {name},

We're thrilled to have you officially joining the {dept} team! Your first week is going to be packed with introductions, system setup, and getting familiar with our workflows and tools.

Attached you'll find your onboarding schedule for the first two weeks, including your mandatory orientation sessions and introductory meetings with your teammates and manager. Please take a few moments to review it and flag any scheduling conflicts ahead of time.

Don't hesitate to reach out to HR or your manager if you need any assistance getting settled in. We're so excited to have you on board!

Warmly,
{sender}
Human Resources & {dept} Team"""
    ),
    # 5. Performance Review Reminder
    (
        "Reminder: Annual Performance Review Submissions Due Friday",
        """Hi {name},

This is a friendly reminder that annual performance self-assessments are due this Friday at 5:00 PM. The review forms are available in the HR portal under the 'Performance' section, and your manager has already been assigned to complete your manager-side review.

Please ensure your self-assessment is thorough and reflects your key achievements, challenges, and growth goals for the coming year. These reviews are an important part of your career development and are used during our compensation planning process.

Thank you for completing this on time — it helps keep the entire review cycle on track.

Best regards,
{sender}
Human Resources
{dept} Department"""
    ),
]

# =========================================================
# SMS TEMPLATES
# =========================================================

PHISHING_SMS_TEMPLATES = [
    "[{bank}] ALERT: Suspicious transaction of ${amount} detected on your account. Verify now or your card will be blocked: {link}",
    "HR NOTICE: Your direct deposit details need immediate update or Friday's paycheck will be delayed. Confirm here: {link}",
    "{bank}: Your account access has been suspended due to failed login attempts. Restore access: {link}",
    "DELIVERY FAILED: Package #{token} could not be delivered. Reschedule here: {link}",
    "IT SECURITY: Your VPN credentials expire today. Renew in the next 2 hrs to avoid lockout: {link}",
]

LEGIT_SMS_TEMPLATES = [
    "Hey {name}, just a reminder our team lunch is at noon today. See you there!",
    "Hi, your appointment is confirmed for tomorrow at 2:30 PM. Reply CANCEL to cancel.",
    "Your package has been delivered and left at the front door. Have a great day!",
    "Hey, are we still on for the 3pm call? Let me know if you need to reschedule.",
    "Just a heads up — the office will be closing early at 3pm this Friday for the company event.",
]

# =========================================================
# FILL-IN VALUES
# =========================================================

NAMES = ["Alex", "Jordan", "Sam", "Taylor", "Morgan", "Casey", "Riley", "Drew"]
DEPTS = ["IT", "Security", "Compliance", "HR", "Finance"]
SENDERS = ["Sarah Mitchell", "James O'Brien", "Linda Chen", "Marcus Webb", "Daniel Park"]
BANKS = ["Chase", "Wells Fargo", "Bank of America", "Citibank"]
LINKS = ["https://verify-account-secure.net/r?t=", "https://corp-portal-auth.com/verify?id=", "https://secure-update-now.net/a?ref="]
AMOUNTS = ["247.00", "1,349.50", "89.99", "3,200.00"]


def _token():
    return "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=12))


def paraphrase(s: str) -> str:
    swaps = {
        "urgent": "important",
        "immediate": "prompt",
        "suspended": "disabled",
        "confirm": "verify",
        "notice": "alert",
        "immediately": "as soon as possible",
        "Failure to": "If you do not",
    }
    out = s
    for k, v in swaps.items():
        out = out.replace(k, v).replace(k.title(), v.title())
    return out


def obfuscate(s: str) -> str:
    """Light obfuscation — replaces a few characters to evade naive keyword filters."""
    return (
        s.replace("verify", "v3rify")
         .replace("Verify", "V3rify")
         .replace("account", "acc0unt")
         .replace("Account", "Acc0unt")
         .replace("password", "p@ssword")
         .replace("Password", "P@ssword")
         .replace("click", "cl1ck")
         .replace("Click", "Cl1ck")
    )


def gen(channel="email", level="template", return_label=False, is_phishing=None):
    if is_phishing is None:
        is_phishing = random.choice([True, False])

    name = random.choice(NAMES)
    dept = random.choice(DEPTS)
    sender = random.choice(SENDERS)
    token = _token()

    if channel == "email":
        if is_phishing:
            subject, body_template, _ = random.choice(PHISHING_EMAIL_TEMPLATES)
            body = body_template.format(name=name, dept=dept, token=token)
            text = f"Subject: {subject}\n\n{body}"
        else:
            subject, body_template = random.choice(LEGIT_EMAIL_TEMPLATES)
            subject = subject.format(dept=dept, name=name)
            body = body_template.format(name=name, dept=dept, sender=sender)
            text = f"Subject: {subject}\n\n{body}"
    else:
        # SMS
        if is_phishing:
            link = random.choice(LINKS) + token
            text = random.choice(PHISHING_SMS_TEMPLATES).format(
                bank=random.choice(BANKS),
                amount=random.choice(AMOUNTS),
                link=link,
                token=token
            )
        else:
            text = random.choice(LEGIT_SMS_TEMPLATES).format(name=name)

    if level == "paraphrase":
        text = paraphrase(text)
    elif level == "obfuscation":
        text = obfuscate(paraphrase(text))

    if return_label:
        return text, 1 if is_phishing else 0
    return text


if __name__ == "__main__":
    channel = sys.argv[1] if len(sys.argv) > 1 else "email"
    level = sys.argv[2] if len(sys.argv) > 2 else "template"
    print(json.dumps({"channel": channel, "level": level, "text": gen(channel, level)}, indent=2))