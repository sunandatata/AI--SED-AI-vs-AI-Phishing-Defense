
# app/app.py
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in sys.path for robust imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import os
import re
import shutil
import subprocess
import inspect
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    average_precision_score,
    f1_score,
)

# =========================================================
# PATHS
# =========================================================
DATA_DIR = ROOT / "data"
PROC_DIR = DATA_DIR / "processed"
SYN_DIR = DATA_DIR / "synthetic"

BASELINE_PATH = ROOT / "blue_agent" / "models" / "baseline.joblib"
ROBERTA_DIR = ROOT / "blue_agent" / "models" / "roberta" / "final"
T5_DIR = ROOT / "red_agent" / "models" / "t5_phish"
CFG_PATH = ROOT / "configs" / "experiment.json"

PROC_DIR.mkdir(parents=True, exist_ok=True)
SYN_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# IMPORT CORE LOGIC (safe imports)
# =========================================================
try:
    from loop.run_round import (
        run_round as run_round_func,
        merge_adversarial_data
    )
except Exception:
    run_round_func = None
    merge_adversarial_data = None

_DASH_ERR = None
try:
    try:
        from app.dashboard import render_dashboard
    except ImportError:
        from dashboard import render_dashboard
except Exception as e:
    _DASH_ERR = str(e)
    def render_dashboard():
        st.error(f"Dashboard module not available. Error: {_DASH_ERR}")

from red_agent.generate_baseline import gen as gen_baseline

try:
    from red_agent.generate_t5 import (
        model_available as t5_available_func,
        generate_with_local_model as t5_generate,
        load_model as load_t5_model,
        generate_text as t5_gen_text,
        _build_prompt as t5_build_prompt
    )
except Exception:
    def t5_available_func(_=None) -> bool: return False
    def t5_generate(**kwargs) -> str: raise RuntimeError("T5 not available.")
    load_t5_model = None
    t5_gen_text = None
    t5_build_prompt = None

# =========================================================
# BLUE EVAL HELPERS
# =========================================================
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

@st.cache_resource
def load_baseline():
    return joblib.load(BASELINE_PATH)

class RobertaWrapper:
    def __init__(self, path: Path):
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(path))
        self.model.eval()

    @torch.no_grad()
    def predict(self, texts, batch_size=16):
        probs = []
        for i in range(0, len(texts), batch_size):
            enc = self.tokenizer(texts[i:i + batch_size], return_tensors="pt", padding=True, truncation=True, max_length=256)
            logits = self.model(**enc).logits
            p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probs.extend(p)
        return np.array(probs)

@st.cache_resource
def load_roberta():
    return RobertaWrapper(ROBERTA_DIR)

@st.cache_resource
def load_t5_cached():
    if not t5_available_func(T5_DIR): return None
    try:
        t, m = load_t5_model(T5_DIR)
        return (t, m)
    except Exception: return None

def eval_predict(texts, model_type: str):
    if model_type == "baseline":
        model = load_baseline()
        return model.predict_proba(texts)[:, 1]
    else:
        rw = load_roberta()
        return rw.predict(texts)

# =========================================================
# SAFETY FILTERS
# =========================================================
BANNED_PATTERNS = [r"\bsex\b", r"\bmoan\b", r"\bpleasure\b", r"\bfuck\b", r"\bshit\b", r"\bass\b", r"\bnude\b"]
def is_safe_text(text: str) -> bool:
    t = (text or "").lower()
    return not any(re.search(p, t) for p in BANNED_PATTERNS)

def safe_enterprise_email(topic: str, dept: str) -> str:
    return f"From: {dept} Support\nSubject: Action Required – {topic}\n\nDear Employee,\nPlease verify your {topic} via the portal.\n\nRegards,\n{dept} Support"

# =========================================================
# HELPERS
# =========================================================
def find_train_csv():
    p = PROC_DIR / "train.csv"
    return p if p.exists() else None

def sample_real_phish(df):
    ph = df[df["label"] == 1]
    if ph.empty: return None
    return str(ph.sample(1).iloc[0]["text"])

def list_round_files():
    return sorted(SYN_DIR.glob("round_*.csv"), key=lambda p: p.stat().st_mtime)

def next_round_id():
    files = list_round_files()
    if not files: return 1
    ids = []
    for f in files:
        try: ids.append(int(f.stem.split("_")[-1]))
        except Exception: pass
    return (max(ids) + 1) if ids else (len(files) + 1)

def run_round_safe(**kwargs):
    if run_round_func is None: raise RuntimeError("run_round() not available.")
    sig = inspect.signature(run_round_func)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return run_round_func(**filtered)

# =========================================================
# STREAMLIT UI
# =========================================================
def main():
    st.set_page_config(page_title="AI²-SED Application", layout="wide")
    st.title("⚡ AI²-SED — AI vs AI Phishing Defense")

    with st.sidebar:
        st.subheader("Environment Status")
        T5_AVAILABLE = bool(t5_available_func(T5_DIR))
        st.write(f"Baseline: {'✅' if BASELINE_PATH.exists() else '❌'}")
        st.write(f"RoBERTa: {'✅' if ROBERTA_DIR.exists() else '❌'}")
        st.write(f"T5: {'✅' if T5_AVAILABLE else '❌'}")

    # Tabs definition
    tabs = st.tabs([
        "Detect Phishing",
        "Generate (Red Agent)",
        "Adversarial Round",
        "Bulk Attack Simulation",
        "Train Models",
        "Evaluate Models",
        "Rounds & Metrics",
        "Dashboard"
    ])
    tab_detect, tab_red, tab_round, tab_bulk, tab_train, tab_eval, tab_rounds, tab_dash = tabs

    # 1. Detect Phishing
    with tab_detect:
        st.subheader("Shield: Live Detection")
        u_text = st.text_area("Paste message below:", height=150, key="live_ta")
        c_m, c_b = st.columns([3,1])
        with c_m: m_sel = st.selectbox("Model", ["baseline", "roberta"], key="live_ms")
        with c_b:
            st.write(""); st.write("")
            if st.button("Analyze", type="primary", key="live_btn"):
                if u_text.strip():
                    p = float(eval_predict([u_text], m_sel)[0])
                    if p > 0.5: st.error(f"⚠️ **PHISHING** ({p:.1%})")
                    else: st.success(f"✅ **BENIGN** ({p:.1%})")
                else: st.warning("Enter text.")

    # 2. Generate
    with tab_red:
        st.subheader("Red Agent: Generation")
        col1, col2 = st.columns(2)
        with col1:
            chan = st.selectbox("Channel", ["email", "sms"], key="gen_chan")
            meth = st.selectbox("Method", ["Template-based", "T5 Transformer", "Dataset Sample (Real)"], key="gen_meth")
            topic = st.text_input("Topic", "Security", key="gen_topic")
            dept = st.text_input("Dept", "IT", key="gen_dept")
            p_mode = st.selectbox("Type", ["Phishing", "Legitimate", "Mixed"], key="gen_type")
        with col2:
            if st.button("Generate Sample", type="primary", key="gen_btn"):
                is_p = (p_mode == "Phishing") if p_mode != "Mixed" else bool(np.random.choice([True, False]))
                txt = None
                if meth == "Dataset Sample (Real)":
                    train_f = find_train_csv()
                    if train_f: txt = sample_real_phish(pd.read_csv(train_f))
                if not txt:
                    if meth == "T5 Transformer" and T5_AVAILABLE:
                        t5b = load_t5_cached()
                        if t5b: txt = t5_gen_text(t5b[0], t5b[1], t5_build_prompt(chan, topic=topic, dept=dept, is_phishing=is_p))
                    if not txt:
                        txt = gen_baseline(chan, "template", is_phishing=is_p)
                        if isinstance(txt, tuple): txt = txt[0]
                if txt and is_safe_text(txt): st.code(txt, language="markdown")
                else: st.warning("Safety block or generation error.")

    # 3. Adversarial Round
    with tab_round:
        st.subheader("Red vs Blue Round")
        n_adv = st.number_input("Samples", 10, 500, 100, 10, key="adv_n")
        m_adv = st.selectbox("Blue Model", ["baseline", "roberta"], key="adv_m")
        rid = next_round_id()
        if st.button("Run Round", key="adv_btn"):
            df, dei, red_ = run_round_safe(round_id=rid, n=n_adv, model_type=m_adv, model_path=str(BASELINE_PATH), 
                                         roberta_path=str(ROBERTA_DIR), t5_path=str(T5_DIR), out_dir=str(SYN_DIR))
            st.success(f"Round {rid} complete. DEI: {dei:.2f}%")
            st.dataframe(df.head())

    # 4. Bulk Simulation
    with tab_bulk:
        st.subheader("🤖 Defense Stress Test")
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            v_n = st.number_input("Volume", 50, 2000, 200, 50, key="bulk_n")
            v_m = st.selectbox("Blue Model", ["baseline", "roberta"], key="bulk_m")
        with v_col2:
            v_red = st.selectbox("Red Engine", ["Baseline (Fast)", "T5 (Deep)"], key="bulk_red")

        if st.button("Run Stress Test", key="bulk_btn"):
            prog = st.progress(0); stat = st.empty(); res = []
            t5_b = load_t5_cached() if (v_red == "T5 (Deep)" and T5_AVAILABLE) else None
            
            for i in range(int(v_n)):
                prog.progress((i+1)/v_n)
                stat.text(f"Attacking... {i+1}/{v_n}")
                isp = bool(np.random.choice([True, False]))
                ch = np.random.choice(["email", "sms"])
                
                if t5_b:
                    txt = t5_gen_text(t5_b[0], t5_b[1], t5_build_prompt(ch, topic="Urgent", dept="Admin", is_phishing=isp))
                else:
                    txt = gen_baseline(ch, "template", is_phishing=isp)
                    if isinstance(txt, tuple): txt = txt[0]
                
                pb = float(eval_predict([txt], v_m)[0])
                res.append({"phish": isp, "detected": (pb > 0.5)})
            
            stat.empty(); prog.empty()
            bdf = pd.DataFrame(res)
            
            # Metrics
            p_sent = int(bdf["phish"].sum())
            p_det = int(((bdf["phish"]) & (bdf["detected"])).sum())
            rec = p_det / max(1, p_sent)
            
            m_c1, m_c2, m_c3 = st.columns(3)
            m_c1.metric("Phishing Sent", p_sent)
            m_c1.metric("Detected", p_det)
            m_c2.metric("Recall (DR)", f"{rec:.1%}")
            m_c3.metric("False Positives", int(((~bdf["phish"]) & (bdf["detected"])).sum()))
            
            st.dataframe(bdf.head(20))

    # 5. Train
    with tab_train:
        st.subheader("Training Center")
        tr_csv = st.text_input("CSV Path", str(PROC_DIR / "train.csv"), key="tr_path")
        if st.button("Train Baseline", key="tr_b"):
            subprocess.call(["python", "blue_agent/train_baseline.py", "--input", tr_csv], cwd=str(ROOT))
            st.success("Train baseline triggered.")
        if st.button("Train RoBERTa", key="tr_r"):
            subprocess.call(["python", "blue_agent/train_roberta.py", "--input", tr_csv, "--epochs", "1"], cwd=str(ROOT))
            st.success("Train RoBERTa triggered.")

    # 6. Evaluate
    with tab_eval:
        st.subheader("ML Evaluation")
        e_csv = st.text_input("Eval CSV", str(PROC_DIR / "train.csv"), key="ev_p")
        e_mod = st.selectbox("Model", ["baseline", "roberta"], key="ev_m")
        if st.button("Evaluate", key="ev_b"):
            df_e = pd.read_csv(e_csv)
            if "text" in df_e.columns and "label" in df_e.columns:
                y = df_e["label"].values; ps = eval_predict(df_e["text"].tolist(), e_mod)
                ls = (ps >= 0.5).astype(int)
                st.metric("F1 Score", f"{f1_score(y, ls):.4f}")
                st.code(classification_report(y, ls))
            else:
                st.error("CSV must contain 'text' and 'label' columns.")

    # 7. Rounds & Metrics
    with tab_rounds:
        st.subheader("Experiment History")
        h_files = list_round_files()
        if h_files:
            h_data = []
            for f in h_files:
                d = pd.read_csv(f); id_ = int(f.stem.split("_")[-1]) if "_" in f.stem else 0
                dei_ = 0.0
                if "blue_label" in d.columns:
                    target = "true_label" if "true_label" in d.columns else ("label" if "label" in d.columns else None)
                    if target:
                        dei_ = (d["blue_label"] == d[target]).mean() * 100
                    else:
                        # Fallback: if all rows were phishing (common in early rounds), 
                        # DEI is just the recall of blue_label
                         dei_ = d["blue_label"].mean() * 100
                h_data.append({"Round": id_, "DEI": dei_})
            hdf = pd.DataFrame(h_data).sort_values("Round")
            st.line_chart(hdf.set_index("Round")["DEI"])
        else: st.info("No data.")

    # 8. Dashboard
    with tab_dash:
        render_dashboard()

if __name__ == "__main__":
    main()
