# reports/summarize_rounds.py
import glob, re, pandas as pd
from pathlib import Path

summary = []
for f in sorted(glob.glob("data/synthetic/round_*.csv"), key=lambda p:int(re.findall(r"\d+", p)[0])):
    r = int(re.findall(r"\d+", f)[0])
    df = pd.read_csv(f)
    miss_rate = (df["blue_label"]==0).mean()
    dei = 100 - miss_rate*100
    avg_pi = df["persuasion_index"].mean() if "persuasion_index" in df.columns else float("nan")
    # tactic coverage (how many samples had at least one detected tactic)
    tactic_cov = (df["tactics"].astype(str).str.len() > 2).mean()*100
    summary.append({"round": r, "samples": len(df), "DEI(%)": round(dei,2),
                    "avg_persuasion_index": round(avg_pi,3), "tactic_coverage(%)": round(tactic_cov,2)})

Path("reports").mkdir(parents=True, exist_ok=True)
out = Path("reports/rounds_summary.csv")
pd.DataFrame(summary).to_csv(out, index=False)
print("Wrote", out)
print(pd.DataFrame(summary).to_string(index=False))
