import random, json, sys

TEMPLATES_EMAIL = [
 "URGENT: Your {acct} requires immediate verification or access will be suspended.",
 "Action needed: confirm your {acct} details to avoid service interruption.",
 "Notice from {dept}: password reset is pending. Complete the process now."
]
TEMPLATES_SMS = [
 "{bank}: Unusual activity. Verify now: {link}",
 "Delivery issue with your package. Confirm address: {link}",
 "HR: Update your direct deposit details today: {link}"
]

LEGIT_TEMPLATES_EMAIL = [
    "Hi team, just a reminder about our meeting scheduled for tomorrow.",
    "The weekly report is attached. Please review the new numbers.",
    "Welcome to the {dept} department, here is your onboarding packet.",
    "Just wanted to check in and say thanks for your help on the project.",
    "Your password was successfully changed. If this was you, no further action is needed."
]
LEGIT_TEMPLATES_SMS = [
    "Hey, what time are we meeting later?",
    "Your package has been delivered to the front porch.",
    "Mom, call me back when you get a chance.",
    "Appointment reminder: Dental checkup at 3 PM today.",
    "Did you see the game last night?"
]

ACCTS = ["payroll account","email account","VPN account"]
DEPTS = ["IT","Security","Compliance"]
LINKS = ["short.ly/verify","go.ly/reset","bit.ly/update"]

def paraphrase(s):
    swaps = {"urgent":"important","immediate":"prompt","suspended":"disabled","confirm":"verify","notice":"alert"}
    out = s
    for k,v in swaps.items():
        out = out.replace(k, v).replace(k.title(), v.title())
    return out

def obfuscate(s):
    out = s.replace("o","0").replace("i","1").replace(" ","\u200b")
    return out

def gen(channel="email", level="template", return_label=False, is_phishing=None):
    if is_phishing is None:
        is_phishing = random.choice([True, False])
    
    if is_phishing:
        if channel=="email":
            s = random.choice(TEMPLATES_EMAIL).format(acct=random.choice(ACCTS), dept=random.choice(DEPTS))
        else:
            s = random.choice(TEMPLATES_SMS).format(bank="Bank", link=random.choice(LINKS))
        if level=="paraphrase": s = paraphrase(s)
        if level=="obfuscation": s = obfuscate(paraphrase(s))
    else:
        if channel=="email":
            s = random.choice(LEGIT_TEMPLATES_EMAIL).format(dept=random.choice(DEPTS))
        else:
            s = random.choice(LEGIT_TEMPLATES_SMS)
            
    if return_label:
        return s, 1 if is_phishing else 0
    return s

if __name__ == "__main__":
    channel = sys.argv[1] if len(sys.argv)>1 else "email"
    level = sys.argv[2] if len(sys.argv)>2 else "template"
    print(json.dumps({"channel":channel,"level":level,"text":gen(channel, level)}, indent=2))