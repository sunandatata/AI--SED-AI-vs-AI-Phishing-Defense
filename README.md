# AI²-SED: Co‑Evolving Generator–Defender Simulator (Starter)

**Novelty axes:** co‑evolution curriculum, tactic-aware explanations, multi-channel (email+SMS),
edge-ready detector, versioned synthetic set.

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python red_agent/generate_seed.py
python blue_agent/train_baseline.py --input data/processed/seed_dataset.csv --model_out blue_agent/models/baseline.joblib
python loop/run_round.py --round_id 1

streamlit run app/app.py
```