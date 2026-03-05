# loop/train_evolution.py
import os, json, pandas as pd, subprocess
from pathlib import Path

ROUNDS = 5
SAMPLES_PER_ROUND = 300
MODEL_OUT = "blue_agent/models/baseline.joblib"

def run(cmd):
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    Path("data/evolution").mkdir(parents=True, exist_ok=True)
    print("Starting AI-vs-AI co-evolution...")
    for r in range(1, ROUNDS + 1):
        print(f"\n=== Round {r} ===")
        run(["python","-m","loop.run_round","--round_id",str(r),"--n",str(SAMPLES_PER_ROUND)])
        df = pd.read_csv(f"data/synthetic/round_{r}.csv")
        misses = df[df["blue_label"]==0][["text","channel"]].copy()
        if misses.empty:
            print("Blue caught everything; skipping retrain.")
            continue
        misses["label"] = 1  # treat missed ones as phishing positives
        master = Path("data/processed/hardneg_master.csv")
        if master.exists():
            old = pd.read_csv(master)
            comb = pd.concat([old, misses], ignore_index=True)
        else:
            comb = misses
        comb.to_csv(master, index=False)
        print("Hard negatives total:", len(comb))
        # Retrain quickly: base + hard negatives
        base = pd.read_csv("data/processed/train.csv")
        mix = pd.concat([base, comb], ignore_index=True)
        tmp = Path("data/processed/train_mixed.csv")
        mix.to_csv(tmp, index=False)
        run(["python","blue_agent/train_baseline.py","--input",str(tmp),"--model_out",MODEL_OUT])

    print("\nEvolution complete. See data/synthetic/ and data/processed/hardneg_master.csv")

if __name__ == "__main__":
    main()
