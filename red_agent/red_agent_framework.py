# red_agent/red_agent_framework.py
from pathlib import Path
import pandas as pd
import numpy as np
import random
from typing import List, Dict, Optional, Tuple

# Import generators
from red_agent.generate_baseline import gen as gen_baseline
from red_agent.generate_t5 import (
    generate_with_local_model as t5_generate,
    load_model as load_t5,
    generate_text as t5_gen_text,
    _build_prompt as t5_build_prompt
)

class AdaptiveRedAgent:
    """
    An AI-driven Red Agent that evolves its strategies based on historical performance.
    """
    def __init__(self, t5_path: Optional[str] = None):
        self.t5_path = t5_path
        self.t5_bundle = None
        self.successful_samples = []  # Phishing samples that bypassed Blue
        self.effective_topics = {}     # Topic -> Success count
        self.effective_channels = {}   # Channel -> Success count
        self.effective_tones = {}      # Tone -> Success count
        
        # Internal state metrics
        self.evolution_round = 0

    def load_resources(self):
        if self.t5_path and not self.t5_bundle:
            try:
                self.t5_bundle = load_t5(self.t5_path)
            except Exception:
                self.t5_bundle = None

    def analyze_feedback(self, df: pd.DataFrame):
        """
        Analyze the detection results to evolve the strategy.
        Focus on 'False Negatives' (Blue misclassified Phish as Benign).
        """
        self.evolution_round += 1
        
        # Filter for successful phishing (True label=1, Blue label=0)
        t_col = "true_label" if "true_label" in df.columns else "label"
        b_col = "blue_label"
        
        if t_col not in df.columns or b_col not in df.columns:
            return

        successes = df[(df[t_col] == 1) & (df[b_col] == 0)]
        
        # Update successful samples pool
        new_samples = successes["text"].tolist()
        self.successful_samples.extend(new_samples)
        self.successful_samples = list(set(self.successful_samples))[-100:]
        
        # Analyze effective properties
        for _, row in successes.iterrows():
            if "topic" in row:
                t = row["topic"]
                self.effective_topics[t] = self.effective_topics.get(t, 0) + 1
            if "channel" in row:
                c = row["channel"]
                self.effective_channels[c] = self.effective_channels.get(c, 0) + 1
            if "tone" in row:
                tn = row["tone"]
                self.effective_tones[tn] = self.effective_tones.get(tn, 0) + 1

    def select_strategy(self) -> Dict:
        """Determines the best channel/topic/mode based on history."""
        channels = ["email", "sms"]
        topics = ["Account Security", "Payroll", "VPN Update", "System Maintenance", "Urgent Alert", "Reward Program"]
        tones = ["urgent", "professional", "soft-sell", "aggressive", "helpful"]
        
        # Factor in historical success with smoothing
        c_weights = [self.effective_channels.get(c, 1) + 1 for c in channels]
        t_weights = [self.effective_topics.get(t, 1) + 1 for t in topics]
        tn_weights = [self.effective_tones.get(tn, 1) + 1 for tn in tones]
        
        selected_chan = random.choices(channels, weights=c_weights)[0]
        selected_topic = random.choices(topics, weights=t_weights)[0]
        selected_tone = random.choices(tones, weights=tn_weights)[0]
        
        return {
            "channel": selected_chan,
            "topic": selected_topic,
            "tone": selected_tone,
            "dept": random.choice(["IT", "HR", "Finance", "Security"]),
        }

    def generate_attack(self, use_t5: bool = False) -> Tuple[str, Dict]:
        """
        Generates an attack and returns metadata.
        """
        self.load_resources()
        strat = self.select_strategy()
        
        # Strategy selection for mutation vs fresh
        # 40% chance to evolve a previously successful sample
        if self.successful_samples and random.random() < 0.4 and use_t5 and self.t5_bundle:
            seed = random.choice(self.successful_samples)
            # Mutation phrases
            mutations = [
                f"Paraphrase this to make it more persuasive: {seed}",
                f"Rewrite this but make it sound more {strat['tone']}: {seed}",
                f"Adapt this phishing message for a {strat['channel']} context: {seed}"
            ]
            prompt = random.choice(mutations)
            txt = t5_gen_text(self.t5_bundle[0], self.t5_bundle[1], prompt)
            strat["method"] = "Adaptive Mutation"
            return txt, strat
        
        # Normal generation but with evolved strategy weights
        if use_t5 and self.t5_bundle:
            prompt = t5_build_prompt(
                strat["channel"], 
                topic=strat["topic"], 
                dept=strat["dept"], 
                tone=strat["tone"],
                is_phishing=True
            )
            txt = t5_gen_text(self.t5_bundle[0], self.t5_bundle[1], prompt)
            strat["method"] = "Selective T5"
            return txt, strat
        else:
            # Baseline template generation using paraphrase level for variety
            res = gen_baseline(strat["channel"], "paraphrase", is_phishing=True)
            txt = res[0] if isinstance(res, tuple) else res
            strat["method"] = "Heuristic Baseline"
            return txt, strat
