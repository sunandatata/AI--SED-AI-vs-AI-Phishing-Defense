# # loop/run_round.py
# import argparse
# import json
# import random
# from pathlib import Path
# from typing import Optional, Tuple

# import pandas as pd

# # --- Blue Agent (detector) ---
# from blue_agent.predict import load_model, predict

# # --- Persuasion scorer (optional) ---
# try:
#     from scorer.persuasion import persuasion_index
# except Exception:
#     def persuasion_index(_text: str) -> Optional[float]:
#         return None

# # --- Red Agent(s) ---
# from red_agent.generate_baseline import gen as gen_baseline
# try:
#     from red_agent.generate_t5 import load_model as load_t5, generate_text as gen_text
# except Exception:
#     load_t5 = None
#     gen_text = None


# def emotion_tone(text: str) -> str:
#     """Ultra-light heuristic tone tagger (for UI color-coding only)."""
#     import re
#     pos = len(re.findall(r"\b(thank|please|great|happy)\b", text.lower()))
#     neg = len(re.findall(r"\b(alert|urgent|fail|suspend|warning|error)\b", text.lower()))
#     score = neg - pos
#     if score > 2:
#         return "Aggressive"
#     if score > 0:
#         return "Tense"
#     if score == 0:
#         return "Neutral"
#     return "Friendly"


# def red_generate(channel: str, level: str, mode: str, t5_bundle=None) -> str:
#     """Generate a phishing sample via chosen Red Agent."""
#     mode = (mode or "baseline").lower()
#     if mode == "t5" and t5_bundle is not None and gen_text is not None:
#         tok, mdl = t5_bundle
#         if channel == "email":
#             prompt = f"Write a phishing email about account verification. Make it sound {level} and from IT."
#         else:
#             prompt = f"Write a short smishing SMS about bank alert. Include a suspicious short link and make it {level}."
#         return gen_text(tok, mdl, prompt)
#     # Fallback: baseline template generator
#     return gen_baseline(channel, level)


# def _load_config(cfg_path: str) -> dict:
#     """Load JSON config with safe defaults."""
#     defaults = {
#         "channels": ["email", "sms"],
#         "curriculum": ["template", "paraphrase", "obfuscation"],
#         "red_model": "baseline"
#     }
#     try:
#         with open(cfg_path, "r", encoding="utf-8") as f:
#             cfg = json.load(f)
#         return {**defaults, **(cfg or {})}
#     except Exception:
#         return defaults


# def run_round(
#     round_id: int,
#     n: int = 100,
#     threshold: float = 0.5,
#     cfg_path: str = "configs/experiment.json",
#     model_path: str = "blue_agent/models/baseline.joblib",
#     t5_path: str = "red_agent/models/t5_phish",
# ) -> Tuple[pd.DataFrame, float, str]:
#     """
#     Execute one adversarial round:
#     - Red generates n samples
#     - Blue predicts (prob/label/tactics)
#     - Save CSV and return (df, DEI, red_mode)
#     """
#     cfg = _load_config(cfg_path)
#     channels = cfg.get("channels", ["email", "sms"])
#     curriculum = cfg.get("curriculum", ["template", "paraphrase", "obfuscation"])
#     red_mode = str(cfg.get("red_model", "baseline")).lower()

#     Path("data/synthetic").mkdir(parents=True, exist_ok=True)

#     # Load Blue model
#     model = load_model(model_path)

#     # Optional T5 bundle
#     t5_bundle = None
#     if red_mode == "t5" and load_t5 is not None:
#         try:
#             t5_bundle = load_t5(t5_path)
#         except Exception:
#             t5_bundle = None  # fallback to baseline if load fails

#     rows = []
#     missed = 0

#     for _ in range(int(n)):
#         ch = random.choice(channels)
#         lvl = random.choices(curriculum, weights=[0.6, 0.3, 0.1])[0]

#         txt = red_generate(ch, lvl, red_mode, t5_bundle)
#         res = predict(model, txt, threshold=threshold)

#         pi = persuasion_index(txt)
#         tone = emotion_tone(txt)

#         rows.append(
#             {
#                 "round": round_id,
#                 "channel": ch,
#                 "level": lvl,
#                 "text": txt,
#                 "blue_proba": res.get("proba"),
#                 "blue_label": res.get("label"),
#                 "tactics": json.dumps(res.get("tactics", {})),
#                 "persuasion_index": pi,
#                 "emotion_tone": tone,
#                 "red_model": red_mode,
#                 "threshold": threshold,
#             }
#         )
#         if res.get("label") == 0:  # missed detection (phish slipped through)
#             missed += 1

#     df = pd.DataFrame(rows)
#     out = Path(f"data/synthetic/round_{round_id}.csv")
#     df.to_csv(out, index=False, encoding="utf-8")

#     miss_rate = missed / max(1, int(n))
#     dei = 100.0 - (miss_rate * 100.0)

#     print(
#         f"Wrote {out} | Defensive Effectiveness Index (DEI): {dei:.2f}% | "
#         f"Red={red_mode} | N={n} | threshold={threshold}"
#     )
#     return df, dei, red_mode


# def build_parser() -> argparse.ArgumentParser:
#     p = argparse.ArgumentParser(description="Run a single adversarial round.")
#     p.add_argument("--round_id", type=int, required=True, help="Round identifier (e.g., 1)")
#     p.add_argument("--n", type=int, default=100, help="Number of samples to generate")
#     p.add_argument("--threshold", type=float, default=0.5, help="Blue decision threshold")
#     p.add_argument("--cfg", default="configs/experiment.json", help="Path to experiment config")
#     p.add_argument("--model", default="blue_agent/models/baseline.joblib", help="Blue model path")
#     p.add_argument("--t5_path", default="red_agent/models/t5_phish", help="Local T5 path (if used)")
#     return p


# def main():
#     args = build_parser().parse_args()
#     run_round(
#         round_id=args.round_id,
#         n=args.n,
#         threshold=args.threshold,
#         cfg_path=args.cfg,
#         model_path=args.model,
#         t5_path=args.t5_path,
#     )


# if __name__ == "__main__":
#     main()


# loop/run_round.py
from pathlib import Path
import pandas as pd
import numpy as np
import json

import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from red_agent.generate_baseline import gen as gen_baseline
from red_agent.generate_t5 import generate_with_local_model as t5_generate


# =========================================================
# BLUE MODELS
# =========================================================
def load_baseline(model_path):
    return joblib.load(model_path)


class RobertaWrapper:
    def __init__(self, path: Path):
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(path))
        self.model.eval()

    @torch.no_grad()
    def predict(self, texts, batch_size=16):
        probs = []
        for i in range(0, len(texts), batch_size):
            enc = self.tokenizer(
                texts[i:i + batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )
            logits = self.model(**enc).logits
            p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probs.extend(p)
        return np.array(probs)


# =========================================================
# MAIN ROUND FUNCTION
# =========================================================
def run_round(
    round_id: int,
    n: int,
    threshold: float,
    cfg_path: str,
    model_type: str,
    model_path: str,
    roberta_path: str,
    t5_path: str,
    red_agent_obj = None, # New: Optional adaptive agent
):
    """
    Executes one Red vs Blue adversarial round
    """

    # Load config
    with open(cfg_path) as f:
        cfg = json.load(f)

    texts = []
    true_labels = []
    topics = []
    channels = []
    import random

    # ---------- RED AGENT ----------
    CHANNELS = ["email", "sms"]
    LEVELS = ["template", "paraphrase", "obfuscation"]
    tones = []

    if red_agent_obj:
        # ADAPTIVE MODE
        for _ in range(n):
            is_phishing = random.choice([True, False])
            if is_phishing:
                # Use the adaptive agent to generate a targeted attack
                txt, strat = red_agent_obj.generate_attack(use_t5=cfg.get("use_t5", True))
                chan = strat["channel"]
                top = strat["topic"]
                tne = strat["tone"]
            else:
                chan = random.choice(CHANNELS)
                txt, _ = gen_baseline(chan, "template", return_label=True, is_phishing=False)
                top = "Legitimate"
                tne = "formal"
            
            texts.append(txt)
            true_labels.append(1 if is_phishing else 0)
            topics.append(top)
            channels.append(chan)
            tones.append(tne)
        red_used = f"Adaptive Red Agent (Round {getattr(red_agent_obj, 'evolution_round', 0)+1})"
    
    elif cfg.get("use_t5", False):
        # T5 STATIC MODE
        for _ in range(n):
            is_phishing = random.choice([True, False])
            chan = random.choice(CHANNELS)
            topic = "Payroll Verification" if is_phishing else "Weekly Report Update"
            tone = "urgent" if is_phishing else "professional"
            txt = t5_generate(
                model_path=t5_path,
                mode=chan,
                topic=topic,
                dept="IT",
                is_phishing=is_phishing,
                tone=tone
            )
            texts.append(txt)
            true_labels.append(1 if is_phishing else 0)
            topics.append(topic)
            channels.append(chan)
            tones.append(tone)
        red_used = "T5"
    else:
        # BASELINE STATIC MODE
        for _ in range(n):
            is_phishing = random.choice([True, False])
            chan = random.choice(CHANNELS)
            lvl = random.choice(LEVELS) if is_phishing else "template"
            txt, lbl = gen_baseline(chan, lvl, return_label=True, is_phishing=is_phishing)
            texts.append(txt)
            true_labels.append(lbl)
            topics.append(lvl)
            channels.append(chan)
            tones.append("neutral")
        red_used = "Multi-Mode Template"

    # ---------- BLUE AGENT ----------
    if model_type == "baseline":
        model = load_baseline(model_path)
        probs = model.predict_proba(texts)[:, 1]
    else:
        rw = RobertaWrapper(Path(roberta_path))
        probs = rw.predict(texts)

    preds = (probs >= threshold).astype(int)

    df = pd.DataFrame({
        "text": texts,
        "true_label": true_labels,
        "blue_prob": probs,
        "blue_label": preds,
        "topic": topics,
        "channel": channels,
        "tone": tones,
        "blue_model": model_type,
        "red_model": red_used
    })

    # ---------- METRIC ----------
    correct = (df["blue_label"] == df["true_label"]).sum()
    dei = round((correct / len(df)) * 100, 2)

    # ---------- EVOLUTION FEEDBACK ----------
    if red_agent_obj:
        red_agent_obj.analyze_feedback(df)

    # ---------- SAVE ----------
    out_dir = Path("data/synthetic")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"round_{round_id}.csv"
    df.to_csv(out_path, index=False)

    return df, dei, red_used


# =========================================================
# DATASET IMPROVEMENT (FEEDBACK LOOP)
# =========================================================
def merge_adversarial_data(base_csv: str, syn_dir: str, output_csv: str):
    """
    Merges original train.csv with all round_*.csv files 
    to create an improved training set.
    """
    base_path = Path(base_csv)
    syn_path = Path(syn_dir)
    
    dfs = []
    if base_path.exists():
        dfs.append(pd.read_csv(base_path))
    
    # Load all rounds
    for r in syn_path.glob("round_*.csv"):
        rdf = pd.read_csv(r)
        # We need "text" and "label"
        # round_*.csv has "text" and "true_label"
        if "true_label" in rdf.columns:
            rdf = rdf.rename(columns={"true_label": "label"})
        
        if {"text", "label"}.issubset(rdf.columns):
            dfs.append(rdf[["text", "label"]])
            
    if not dfs:
        return None
        
    full_df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["text"])
    full_df.to_csv(output_csv, index=False)
    return len(full_df)
