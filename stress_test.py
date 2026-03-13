
import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import json
import pandas as pd
from loop.run_round import run_round
from red_agent.red_agent_framework import AdaptiveRedAgent

def run_stress_test(num_rounds=100):
    print(f"Starting Stress Test: {num_rounds} rounds...")
    
    # Paths
    cfg_path = ROOT / "configs" / "experiment.json"
    model_path = ROOT / "blue_agent" / "models" / "baseline.joblib"
    roberta_path = ROOT / "blue_agent" / "models" / "roberta" / "final"
    t5_path = ROOT / "red_agent" / "models" / "t5_phish"
    
    # Initialize Adaptive Red Agent
    red_agent = AdaptiveRedAgent()
    
    history = []
    
    for rid in range(1, num_rounds + 1):
        print(f"--- Round {rid} ---")
        # Run a round with 50 samples for speed (total 5000 samples)
        df, dei, red_mode = run_round(
            round_id=rid,
            n=50,
            threshold=0.5,
            cfg_path=str(cfg_path),
            model_type="baseline", # Use baseline for speed in stress test
            model_path=str(model_path),
            roberta_path=str(roberta_path),
            t5_path=str(t5_path),
            red_agent_obj=red_agent
        )
        
        # Analyze feedback for evolution
        red_agent.analyze_feedback(df)
        
        history.append({
            "Round": rid,
            "DEI": dei,
            "AttackSuccess": (1 - dei) * 100
        })
        
    history_df = pd.DataFrame(history)
    history_df.to_csv(ROOT / "data" / "synthetic" / "stress_test_results.csv", index=False)
    print("Stress test complete. Results saved to data/synthetic/stress_test_results.csv")

if __name__ == "__main__":
    run_stress_test(100)
