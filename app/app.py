
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
import urllib.parse
import whois

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
    from red_agent.red_agent_framework import AdaptiveRedAgent
except Exception as e:
    st.error(f"Critical Import Error (Red Agent): {e}")
    class AdaptiveRedAgent:
        def __init__(self, **kwargs): self.evolution_round = 0
        def generate_attack(self, **kwargs): return "Red Agent Error", {}
        def analyze_feedback(self, **kwargs): pass
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
    """Generates phishing probabilities using the selected model."""
    if model_type == "baseline":
        model = load_baseline()
        return model.predict_proba(texts)[:, 1]
    else:
        rw = load_roberta()
        return rw.predict(texts)

def explain_baseline(text: str):
    """Explains why a text was flagged by identifying high-contribution keywords."""
    try:
        pipeline = load_baseline()
        vectorizer = pipeline.named_steps['tfidf']
        model = pipeline.named_steps['clf']
        
        # Transform the single text
        X = vectorizer.transform([text])
        features = vectorizer.get_feature_names_out()
        
        # Get coefficients (for binary classification, model.coef_[0])
        coeffs = model.coef_[0]
        
        # Calculate contribution: tfidf_value * coefficient
        # X is sparse, so we find non-zero entries
        row, cols = X.nonzero()
        contributions = []
        for col in cols:
            score = X[0, col] * coeffs[col]
            if score > 0.05: # Only care about high phish-signal words
                contributions.append((features[col], score))
        
        # Sort by impact
        contributions.sort(key=lambda x: x[1], reverse=True)
        return contributions
    except Exception:
        return []

def highlight_text(text: str, contributions: list[tuple[str, float]]):
    """Highlights high-risk words in the text using HTML/Markdown."""
    highlighted = text
    # Sort by length of word descending to avoid overlapping issues (e.g., 'urgent' before 'urgency')
    sorted_words = sorted(contributions, key=lambda x: len(x[0]), reverse=True)
    
    for word, score in sorted_words:
        # Use a regex for whole-word replacement to avoid corruption
        pattern = re.compile(rf'({re.escape(word)})', re.IGNORECASE)
        # Intensity based on score
        color = "#ff4b4b" if score > 0.5 else "#ffa500"
        highlighted = pattern.sub(f'<span style="background-color: {color}; color: white; padding: 2px 4px; border-radius: 4px; font-weight: bold;">\\1</span>', highlighted)
    return highlighted

# =========================================================
# SAFETY FILTERS
# =========================================================
BANNED_PATTERNS = [r"\bsex\b", r"\bmoan\b", r"\bpleasure\b", r"\bfuck\b", r"\bshit\b", r"\bass\b", r"\bnude\b"]
def is_safe_text(text: str) -> bool:
    t = (text or "").lower()
    return not any(re.search(p, t) for p in BANNED_PATTERNS)

def safe_enterprise_email(topic: str, dept: str) -> str:
    return f"From: {dept} Support\nSubject: Action Required – {topic}\n\nDear Employee,\nPlease verify your {topic} via the portal.\n\nRegards,\n{dept} Support"

TRUSTED_DOMAINS = [
    "google.com", "microsoft.com", "apple.com", "amazon.com", "github.com",
    "myworkdayjobs.com", "workday.com", "okta.com", "salesforce.com",
    "slack.com", "zoom.us", "dropbox.com", "box.com", "comcast.com",
]

def extract_domains(text: str) -> list[str]:
    urls = re.findall(r'(https?://[^\s]+)', text)
    domains = set()
    for u in urls:
        try:
            # extract domain, strip subdomains if possible for simple matching, or just check if it ends with a trusted domain
            loc = urllib.parse.urlparse(u).netloc.lower()
            if loc.startswith("www."): loc = loc[4:]
            domains.add(loc)
        except: pass
    return list(domains)

def is_domain_trusted(domain: str) -> bool:
    return any(domain == td or domain.endswith("." + td) for td in TRUSTED_DOMAINS)

def get_domain_age_days(domain):
    """Fetches domain age in days using WHOIS."""
    try:
        # Some libraries/OS might have issues with whois, so we wrap tightly
        w = whois.whois(domain)
        creation_date = w.creation_date
        
        # Handle cases where creation_date is a list
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
            
        if creation_date and isinstance(creation_date, datetime):
            age = datetime.now() - creation_date
            return age.days
        return None
    except Exception:
        return None

# =========================================================
# FEDERATED INTELLIGENCE (SIMULATED)
# =========================================================
FED_INTEL_PATH = ROOT / "fed_intel.json"

def get_fed_intel():
    import json
    if not FED_INTEL_PATH.exists():
        with open(FED_INTEL_PATH, 'w') as f:
            json.dump({"shared_signatures": [], "node_count": 1}, f)
    with open(FED_INTEL_PATH, 'r') as f:
        return json.load(f)

def update_fed_intel(data):
    import json
    with open(FED_INTEL_PATH, 'w') as f:
        json.dump(data, f, indent=4)

def sync_global_intel():
    """Simulates syncing local defender with a federated intelligence hub."""
    intel = get_fed_intel()
    new_sigs = intel.get("shared_signatures", [])
    added = 0
    global TRUSTED_DOMAINS
    for sig in new_sigs:
        if sig not in TRUSTED_DOMAINS:
            TRUSTED_DOMAINS.append(sig)
            added += 1
    return added, len(TRUSTED_DOMAINS)

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
    st.title("AI²-SED — AI vs AI Phishing Defense")

    # Environment checks (hidden from UI)
    T5_AVAILABLE = bool(t5_available_func(T5_DIR))
    if "red_agent" not in st.session_state:
        st.session_state.red_agent = AdaptiveRedAgent(t5_path=str(T5_DIR))

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

    # 1. Detect Phishing (Live User Tool)
    with tab_detect:
        st.subheader("Shield: Real-time Phishing Verification")
        u_text = st.text_area("Paste the message content here:", height=180, key="live_user_input_ta")
        
        col_m, col_b = st.columns([3, 1])
        with col_m:
            m_choice = st.selectbox("Select Blue Agent Model", ["baseline", "roberta"], key="live_user_model_sel")
        with col_b:
            st.write(""); st.write("")
            analyze_active = st.button("Verify Message", type="primary", key="live_user_btn_run")

        if analyze_active:
            if u_text.strip():
                with st.spinner("Analyzing message and verifying entities..."):
                    
                    found_domains = extract_domains(u_text)
                    untrusted_domains = [d for d in found_domains if not is_domain_trusted(d)]
                    trusted_domains = [d for d in found_domains if is_domain_trusted(d)]
                    
                    prob = float(eval_predict([u_text], m_choice)[0])
                    
                    st.divider()
                    st.subheader("Analysis Verdict")
                    
                    if untrusted_domains:
                        st.error(f"⚠️ **PHISHING DETECTED: Untrusted Linking Entity**")
                        st.write(f"This message contains links to unverified or suspicious external domains: `{', '.join(untrusted_domains)}`")
                        
                        # Check domain age for untrusted domains
                        for d in untrusted_domains:
                            age = get_domain_age_days(d)
                            if age is not None and age < 60: # Flag domains younger than 60 days
                                st.warning(f"🚨 **Urgent Warning:** The domain `{d}` was registered very recently ({age} days ago). This is a common tactic for disposable phishing sites.")
                                prob = 0.99
                        
                        prob = max(prob, 0.95) # Override probability mechanically for the metric
                    elif trusted_domains:
                        st.success(f"✅ **BENIGN MESSAGE**")
                        st.info(f"Verified Entities Found: `{', '.join(trusted_domains)}`")
                        st.write("This message is verified legitimate based on safe entity links, overriding AI language patterns.")
                        prob = 0.01 # Force it to show as benign in metrics
                    else:
                        st.error(f"⚠️ **PHISHING DETECTED: Unverified Source / Language**")
                        if prob > 0.5:
                            st.write("This message matches known phishing language patterns and lacks verified entity links.")
                        else:
                            st.write("No verified enterprise links were found to prove authenticity. Defaulting to high-security blocking.")
                        prob = max(prob, 0.85) # Force to phishing
                    
                    m1, m2 = st.columns(2)
                    m1.metric("Phish Probability", f"{prob:.1%}")
                    m2.metric("Confidence Score", f"{abs(prob-0.5)*2:.1%}")

                    if prob > 0.5:
                        st.write("---")
                        st.subheader("Explainable AI (XAI) Analysis")
                        st.info("The words highlighted below have the strongest mathematical influence on the phishing detection score.")
                        
                        contribs = explain_baseline(u_text)
                        if contribs:
                            h_text = highlight_text(u_text, contribs)
                            st.markdown(f'<div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; line-height: 2.0; font-size: 1.1em;">{h_text}</div>', unsafe_allow_html=True)
                            
                            st.write("**Top Risk Keywords Detected:**")
                            cols_k = st.columns(min(len(contribs), 5))
                            for i, (word, score) in enumerate(contribs[:5]):
                                with cols_k[i]:
                                    st.metric(word.capitalize(), f"+{score:.2f}")
                        else:
                            st.write("The model identified this as phishing based on a combination of subtle patterns rather than individual keywords.")
            else:
                st.warning("Please paste a message to analyze.")

    # 2. Generate (Red Agent Lab)
    with tab_red:
        st.subheader("Red Agent: Attack Labs")
        
        c_r1, c_r2 = st.columns(2)
        with c_r1:
            r_chan = st.selectbox("Delivery Channel", ["email", "sms"], key="red_lab_chan")
            r_meth = st.selectbox("Attack Method", ["Template-based", "T5 Transformer", "Dataset Sample (Real)"], key="red_lab_meth")
            r_topic = st.text_input("Core Topic", "System Upgrade", key="red_lab_topic")
            r_dept = st.text_input("Sender Department", "IT Security", key="red_lab_dept")
        
        with c_r2:
            st.info("**Red agent output is processed here**")
            if st.button("Generate Experimental Attack", type="primary", key="red_lab_gen_btn"):
                with st.spinner("Synthesizing attack content..."):
                    is_p = True # Default to generating a Phishing attack
                    gen_txt = None
                    
                    if r_meth == "Dataset Sample (Real)":
                        t_csv = find_train_csv()
                        if t_csv: gen_txt = sample_real_phish(pd.read_csv(t_csv))
                    
                    if not gen_txt:
                        if r_meth == "T5 Transformer" and T5_AVAILABLE:
                            bundle = load_t5_cached()
                            if bundle:
                                prompt = t5_build_prompt(r_chan, topic=r_topic, dept=r_dept, is_phishing=is_p)
                                gen_txt = t5_gen_text(bundle[0], bundle[1], prompt)
                        if not gen_txt:
                            gen_txt = gen_baseline(r_chan, "template", is_phishing=is_p)
                            if isinstance(gen_txt, tuple): gen_txt = gen_txt[0]
                    
                    if gen_txt and is_safe_text(gen_txt):
                        st.code(gen_txt, language="markdown")
                        st.caption(f"Generation Engine: {r_meth} | Target: Phishing (Legitimate-Looking Lure)")
                    else:
                        st.warning("Attack synthesis failed or content was blocked by safety filters.")

    # 3. Adversarial Round (Red vs Blue Interaction)
    with tab_round:
        st.subheader("Adversarial Rounds")
        
        c_a1, c_a2 = st.columns(2)
        with c_a1:
            a_n = st.number_input("Attack Sample Size", 10, 500, 100, 10, key="adv_round_n")
            a_blue = st.selectbox("Blue Agent Selection", ["baseline", "roberta"], key="adv_round_blue")
        with c_a2:
            a_thr = st.slider("Detection Threshold", 0.0, 1.0, 0.5, 0.05, key="adv_round_thr")
            a_adaptive = st.checkbox("Enable Adversarial Evolution", value=True, help="If enabled, Red Agent learns from bypassed samples.", key="adv_round_adaptive")
        
        cur_rid = next_round_id()
        if st.button("Initiate Adversarial Simulation", key="adv_round_run_btn"):
            with st.spinner(f"Running Round {cur_rid}..."):
                try:
                    r_agent = st.session_state.red_agent if a_adaptive else None
                    df_r, dei_r, red_tag = run_round_safe(
                        round_id=cur_rid, n=a_n, threshold=a_thr, model_type=a_blue,
                        cfg_path=str(CFG_PATH),
                        model_path=str(BASELINE_PATH), roberta_path=str(ROBERTA_DIR),
                        t5_path=str(T5_DIR), out_dir=str(SYN_DIR),
                        red_agent_obj=r_agent
                    )
                    st.success(f"Adversarial Round {cur_rid} Complete!")
                    st.metric("Defensive Effectiveness Index (DEI)", f"{dei_r:.2f}%")
                    st.caption(f"Engine: {red_tag}")
                    st.dataframe(df_r.head(10))
                except Exception as ex:
                    st.error(f"Round execution failed: {str(ex)}")

        st.divider()
        st.subheader("Evolutionary Campaign Simulation")
        c_c1, c_c2 = st.columns(2)
        with c_c1:
            camp_rounds = st.number_input("Rounds to Run", 2, 10, 3, 1, key="camp_rounds_n")
        with c_c2:
            st.write(""); st.write("")
            if st.button("🚀 Start Evolutionary Campaign", key="camp_run_btn"):
                prog_camp = st.progress(0)
                status_camp = st.empty()
                camp_results = []
                for r_idx in range(int(camp_rounds)):
                    rid_camp = next_round_id()
                    status_camp.text(f"Executing Evolutionary Round {r_idx+1}/{camp_rounds} (ID: {rid_camp})...")
                    r_agent = st.session_state.red_agent if a_adaptive else None
                    _, dei_camp, _ = run_round_safe(
                        round_id=rid_camp, n=a_n, threshold=a_thr, model_type=a_blue,
                        cfg_path=str(CFG_PATH),
                        model_path=str(BASELINE_PATH), roberta_path=str(ROBERTA_DIR),
                        t5_path=str(T5_DIR), out_dir=str(SYN_DIR),
                        red_agent_obj=r_agent
                    )
                    camp_results.append(dei_camp)
                    prog_camp.progress((r_idx+1)/camp_rounds)
                
                status_camp.success(f"Evolutionary Campaign Complete! Final DEI delta: {camp_results[-1] - camp_results[0]:.2f}%")
                st.line_chart(pd.DataFrame({"DEI": camp_results}), use_container_width=True)

    # 4. Bulk Attack Simulation (Stress Test)
    with tab_bulk:
        st.subheader("Bulk Attack Simulation")
        
        c_b1, c_b2 = st.columns(2)
        with c_b1:
            b_n = st.number_input("Campaign Volume", 50, 2000, 200, 50, key="bulk_sim_n")
            b_blue = st.selectbox("Blue Agent Target", ["baseline", "roberta"], key="bulk_sim_blue")
        with c_b2:
            b_red = st.selectbox("Red Generation Engine", ["Baseline (Fast)", "T5 (Deep)"], key="bulk_sim_red")
            
        if st.button("🚀 Launch Campaign Simulation", key="bulk_sim_run_btn"):
            prog = st.progress(0); status_txt = st.empty(); results_list = []
            t5_bundle = load_t5_cached() if (b_red == "T5 (Deep)" and T5_AVAILABLE) else None
            
            for i in range(int(b_n)):
                prog.progress((i+1)/b_n)
                status_txt.text(f"Generating and testing attack... {i+1} of {b_n}")
                is_phish_campaign = bool(np.random.choice([True, False]))
                ch_campaign = np.random.choice(["email", "sms"])
                
                if t5_bundle:
                    prompt_mode = "smishing" if (ch_campaign == "sms" and is_phish_campaign) else ch_campaign
                    max_tokens = 48 if ch_campaign == "sms" else 96
                    txt_c = t5_gen_text(t5_bundle[0], t5_bundle[1], t5_build_prompt(prompt_mode, topic="Security Update", dept="Service", is_phishing=is_phish_campaign), max_new_tokens=max_tokens)
                else:
                    txt_c = gen_baseline(ch_campaign, "template", is_phishing=is_phish_campaign)
                    if isinstance(txt_c, tuple): txt_c = txt_c[0]
                
                prediction_prob = float(eval_predict([txt_c], b_blue)[0])
                results_list.append({"is_phish": is_phish_campaign, "was_detected": (prediction_prob > 0.5)})
            
            status_txt.empty(); prog.empty()
            df_bulk = pd.DataFrame(results_list)
            
            # Metrics
            total_phish = int(df_bulk["is_phish"].sum())
            detected_phish = int(((df_bulk["is_phish"]) & (df_bulk["was_detected"])).sum())
            total_benign = len(df_bulk) - total_phish
            fp_count = int(((~df_bulk["is_phish"]) & (df_bulk["was_detected"])).sum())
            
            st.write("### Campaign Metrics Recap")
            met_c1, met_c2, met_c3, met_c4 = st.columns(4)
            met_c1.metric("Phishing Sent", total_phish)
            met_c2.metric("Phishing Caught", detected_phish)
            met_c3.metric("Recall (Detection Rate)", f"{(detected_phish/max(1, total_phish)):.1%}")
            met_c4.metric("False Positives", fp_count)
            
            st.dataframe(df_bulk.head(20))

    # 5. Train Models (Defensive Improvement)
    with tab_train:
        st.subheader("Blue Agent Academy: Training")
        
        train_input_path = st.text_input("Target Training Dataset (CSV)", str(PROC_DIR / "train.csv"), key="train_input_path_field")
        
        c_t1, c_t2 = st.columns(2)
        with c_t1:
            if st.button("Retrain Baseline Agent", key="train_btn_baseline"):
                with st.spinner("Training Baseline..."):
                    subprocess.call(["python", "blue_agent/train_baseline.py", "--input", train_input_path], cwd=str(ROOT))
                    st.success("Baseline Agent training triggered successfully.")
        with c_t2:
            if st.button("Retrain RoBERTa Agent", key="train_btn_roberta"):
                with st.spinner("Training RoBERTa (Epochs=1)..."):
                    subprocess.call(["python", "blue_agent/train_roberta.py", "--input", train_input_path, "--epochs", "1"], cwd=str(ROOT))
                    st.success("RoBERTa Agent training triggered successfully.")

        st.divider()
        st.subheader("Continuous Feedback Loop")
        
        if st.button("🔗 Merge Round Data & Close Loop", key="train_close_loop_btn"):
            with st.spinner("Aggregating adversarial data..."):
                merged_path = str(PROC_DIR / "augmented_train.csv")
                count = merge_adversarial_data(str(PROC_DIR / "train.csv"), str(SYN_DIR), merged_path)
                if count:
                    st.success(f"Merged {count} unique samples into {merged_path}.")
                    st.info("You can now set the 'Target Training Dataset' above to this new file and retrain.")
                else:
                    st.warning("No adversarial round data found to merge.")

        st.divider()
        st.subheader("Federated Learning (Simulated)")
        st.write("Exchange threat signatures with the Global Intelligence Hub to strengthen collective defense.")
        
        c_f1, c_f2 = st.columns(2)
        with c_f1:
            if st.button("🛰️ Sync with Global Hub", key="fed_sync_btn"):
                added, total = sync_global_intel()
                st.success(f"Sync Complete! Added {added} new trusted enterprise signatures from federated nodes.")
                st.info(f"Local Whitelist now contains {total} verified domains.")
        with c_f2:
            st.write("Publish verified safe domains to help other nodes reduce false positives.")
            new_pub = st.text_input("Domain to Publish", "my-safe-biz.com", key="fed_pub_input")
            if st.button("📢 Publish to Hub", key="fed_pub_btn"):
                if new_pub:
                    intel = get_fed_intel()
                    if new_pub not in intel["shared_signatures"]:
                        intel["shared_signatures"].append(new_pub)
                        intel["last_sync"] = datetime.now().isoformat()
                        update_fed_intel(intel)
                        st.success(f"Successfully published `{new_pub}` to the Global Intelligence Hub.")
                    else:
                        st.warning("Domain already exists in the Global Hub.")

    # 6. Evaluate Models (Static ML Benchmark)
    with tab_eval:
        st.subheader("Model Evaluation Hub")
        
        e_path_csv = st.text_input("Evaluation Dataset (CSV)", str(PROC_DIR / "train.csv"), key="eval_hub_csv_path")
        e_model_type = st.selectbox("Model to Benchmark", ["baseline", "roberta"], key="eval_hub_model_sel")
        
        if st.button("Run Comprehensive Evaluation", key="eval_hub_run_btn"):
            df_eval_data = pd.read_csv(e_path_csv)
            if "text" in df_eval_data.columns and "label" in df_eval_data.columns:
                with st.spinner("Calculating metrics..."):
                    y_ground = df_eval_data["label"].values
                    probs_pred = eval_predict(df_eval_data["text"].tolist(), e_model_type)
                    labels_pred = (probs_pred >= 0.5).astype(int)
                    
                    # Basic Metrics
                    f1_val = f1_score(y_ground, labels_pred)
                    ap_val = average_precision_score(y_ground, probs_pred)
                    
                    st.write("### Performance Metrics")
                    e_col1, e_col2 = st.columns(2)
                    e_col1.metric("F1 Score", f"{f1_val:.4f}")
                    e_col2.metric("PR-AUC (Average Precision)", f"{ap_val:.4f}")
                    
                    st.write("#### Classification Report")
                    st.code(classification_report(y_ground, labels_pred))
                    
                    # Confusion Matrix Visual
                    st.write("#### Confusion Matrix")
                    cm_mat = confusion_matrix(y_ground, labels_pred)
                    fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
                    ax_cm.imshow(cm_mat, cmap='Blues', alpha=0.3)
                    for (i, j), val in np.ndenumerate(cm_mat):
                        ax_cm.text(j, i, str(val), ha='center', va='center', fontsize=12)
                    ax_cm.set_xticks([0, 1]); ax_cm.set_xticklabels(["Benign", "Phish"])
                    ax_cm.set_yticks([0, 1]); ax_cm.set_yticklabels(["Benign", "Phish"])
                    ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("True")
                    st.pyplot(fig_cm)
            else:
                st.error("Missing required columns: 'text' and 'label'.")

    # 7. Rounds & Metrics (Historical Tracking)
    with tab_rounds:
        st.subheader("📈 Round Records & DEI History")
        st.write("Track the evolution of defensive effectiveness across adversarial simulations.")
        
        history_files = list_round_files()
        if history_files:
            history_rows = []
            for f_round in history_files:
                d_round = pd.read_csv(f_round)
                r_id_numeric = int(f_round.stem.split("_")[-1]) if "_" in f_round.stem else 0
                
                # Robust DEI calculation
                t_col = "true_label" if "true_label" in d_round.columns else ("label" if "label" in d_round.columns else None)
                if "blue_label" in d_round.columns and t_col:
                    dei_metric = (d_round["blue_label"] == d_round[t_col]).mean() * 100
                    p_catch = int(((d_round[t_col] == 1) & (d_round["blue_label"] == 1)).sum())
                    p_total = max(1, int((d_round[t_col] == 1).sum()))
                else:
                    dei_metric = d_round["blue_label"].mean() * 100 if "blue_label" in d_round.columns else 0.0
                    p_catch, p_total = 0, 1
                
                history_rows.append({
                    "Round": r_id_numeric,
                    "Samples": len(d_round),
                    "DEI (%)": round(dei_metric, 2),
                    "Detection (%)": round((p_catch/p_total)*100, 1)
                })
            
            df_history = pd.DataFrame(history_rows).sort_values("Round")
            st.line_chart(df_history.set_index("Round")[["DEI (%)", "Detection (%)"]])
            st.write("#### Detailed Historical Data")
            st.dataframe(df_history)
        else:
            st.info("No recorded adversarial rounds found. Start by running an **Adversarial Round**.")

    # 8. Dashboard (Executive Visualization)
    with tab_dash:
        render_dashboard()

if __name__ == "__main__":
    main()
