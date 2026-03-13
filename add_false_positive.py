import pandas as pd
from pathlib import Path

csv_path = Path("data/processed/train.csv")

text = """Click this link to confirm your email address and complete setup for your candidate account
https://comcast.wd5.myworkdayjobs.com/Comcast_Careers/activate/b0whq3dzkr1yf01542nho5f19sjh475b1wo7q1vs3255s0k5tlojbax9oo97no8vsqv9ccnzgo6tlj93prf9zpg2uicmbhdpx1m9/?redirect=%2Fen-US%2FComcast_Careers%2Fjob%2FCO---Englewood%252C-183-Inverness-Dr-West%2FSoftware-Engineer--Java-_R430875%2Fapply%2FapplyManually%3Fjr_id%3D69af3910749500645093bc37%26a_t_id%3D69af8ed6afe59372c662ede1
The link will expire after 24 hours."""

# Let's add variations of legitimate emails with links
extra_texts = [
    text,
    "Please click the link below to verify your email address for your new Workday account.\nhttps://company.myworkdayjobs.com/verify",
    "Action Required: Complete your profile setup.\nTo complete your setup, click here: https://portal.legit-company.com/setup\nThis link expires in 24 hours.",
    "Confirm your email address to access the candidate portal.\nClick this link: https://careers.company.com/confirm",
]

df = pd.read_csv(csv_path)

# Ensure 'text' and 'label' columns exist
if "text" in df.columns and "label" in df.columns:
    new_rows = pd.DataFrame({"text": extra_texts, "label": [0]*len(extra_texts)}) # 0 = Benign
    df = pd.concat([df, new_rows], ignore_index=True)
    df.to_csv(csv_path, index=False)
    print(f"Added {len(extra_texts)} legitimate examples to {csv_path}")
else:
    print("Columns 'text' or 'label' not found.")
