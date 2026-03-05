# data/prepare_kaggle.py
import pandas as pd
from pathlib import Path
import re, sys

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

SEED_ROWS = [
    {"channel":"email","text":"URGENT: Your payroll account requires immediate verification to avoid suspension.","label":1},
    {"channel":"email","text":"Reminder: Team sync tomorrow at 10am. Agenda attached.","label":0},
    {"channel":"sms","text":"Bank Alert: Unusual activity detected. Verify now: short.ly/xyz","label":1},
    {"channel":"sms","text":"Hey, reaching late by 10min. See you soon!","label":0},
    {"channel":"email","text":"IT Notice: Password expires today. Click here to keep access.","label":1},
    {"channel":"email","text":"Thanks for submitting the draft. I left comments inline.","label":0}
]

def load_any(path):
    print(f"  -> trying to read {path.name}")
    # try common encodings
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"     read OK with encoding={enc} rows={len(df)} cols={list(df.columns)[:6]}")
            return df
        except Exception as e:
            pass
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    print(f"     read OK with python engine rows={len(df)} cols={list(df.columns)[:6]}")
    return df

def normalize_text_col(df):
    # try common text/message column names
    candidates = ["text","message","email_text","body","EmailText","Message","content","EmailBody","EmailBodyText"]
    for c in candidates:
        if c in df.columns:
            if c != "text":
                df = df.rename(columns={c:"text"})
            return df
    # (optional) combine subject + message if present
    lowers = {c.lower(): c for c in df.columns}
    if "subject" in lowers and "message" in lowers:
        subj = lowers["subject"]; msg = lowers["message"]
        df["text"] = (df[subj].astype(str) + " — " + df[msg].astype(str))
        return df
    raise ValueError(f"Could not find a text/content column in columns: {list(df.columns)[:10]}")

def normalize_label_col(df):
    # map various labels to {0,1} where 1=phish/spam
    lab = None
    for c in df.columns:
        lc = c.lower()
        if lc in ["label","labels","class","target","is_phishing","is_spam","spam","y","phishing"]:
            lab = c; break
        if lc in ["type","category"]:
            lab = c; break
    if lab is None:
        for c in df.columns:
            if re.search("phish|spam|ham", c, re.I):
                lab = c; break
    if lab is None:
        raise ValueError(f"Could not find a label column in columns: {list(df.columns)[:10]}")

    if lab != "label":
        df = df.rename(columns={lab:"label"})

    def to01(x):
        s = str(x).strip().lower()
        if s in ["1","true","phishing","phish","spam","malicious","bad","yes","fraud"]: return 1
        if s in ["0","false","legit","ham","benign","no","normal","good","notspam"]: return 0
        if "spam" in s: return 1
        if "ham"  in s: return 0
        try:
            return 1 if float(s) >= 0.5 else 0
        except:
            return None

    df["label"] = df["label"].map(to01)
    before = len(df)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    print(f"     label normalization: kept {len(df)}/{before}")
    return df

def prepare_from_raw():
    frames = []
    csvs = list(RAW.glob("*.csv"))
    if not csvs:
        print("No CSVs found in data/raw — will fall back to seed dataset.")
        return None

    for csv in csvs:
        print(f"Reading: {csv.name}")
        try:
            df = load_any(csv)
            df = normalize_text_col(df)
            df = normalize_label_col(df)
            ch = "sms" if re.search("sms", csv.name, re.I) else "email"
            df["channel"] = ch
            frames.append(df[["text","label","channel"]].dropna())
            print(f"  -> usable rows added: {len(frames[-1])}")
        except Exception as e:
            print(f"  !! skipped {csv.name} because: {e}")

    if not frames:
        print("No usable CSVs after parsing — will fall back to seed dataset.")
        return None

    all_df = pd.concat(frames, ignore_index=True)
    all_df["text"] = all_df["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    all_df = all_df[all_df["text"].str.len() > 3]
    return all_df

def write_out(df, name="train.csv"):
    out_path = OUT / name
    df.to_csv(out_path, index=False)
    print("Wrote", out_path, "| rows:", len(df))

def main():
    df = prepare_from_raw()
    if df is None:
        print(">>> Using seed dataset to create data/processed/train.csv")
        df = pd.DataFrame(SEED_ROWS)
    write_out(df, "train.csv")

if __name__ == "__main__":
    main()
