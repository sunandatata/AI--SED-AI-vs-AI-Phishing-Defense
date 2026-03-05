import pandas as pd
from pathlib import Path
out = Path("data/processed"); out.mkdir(parents=True, exist_ok=True)
rows = [
 {"channel":"email","text":"URGENT: Your payroll account requires immediate verification to avoid suspension.","label":1},
 {"channel":"email","text":"Reminder: Team sync tomorrow at 10am. Agenda attached.","label":0},
 {"channel":"sms","text":"Bank Alert: Unusual activity detected. Verify now: short.ly/xyz","label":1},
 {"channel":"sms","text":"Hey, reaching late by 10min. See you soon!","label":0},
 {"channel":"email","text":"IT Notice: Password expires today. Click here to keep access.","label":1},
 {"channel":"email","text":"Thanks for submitting the draft. I left comments inline.","label":0}
]
pd.DataFrame(rows).to_csv(out/"seed_dataset.csv", index=False)
print("Wrote", out/"seed_dataset.csv")