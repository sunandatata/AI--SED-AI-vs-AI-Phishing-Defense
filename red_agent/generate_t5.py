# red_agent/generate_t5.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Lightweight prompt templates (kept generic / non-operational)
PROMPTS = {
    "email_phishing": "Write a short, formal phishing email about {topic}. It should sound {tone} and appear to be from {dept}. Make it sound urgent.",
    "email_legit": "Write a short, formal legitimate email about {topic}. It should sound {tone} and appear to be from {dept}.",
    "sms_phishing": "Write a short mobile text about {topic}. Keep it concise and {tone}. Include a placeholder short link.",
    "sms_legit": "Write a short legitimate mobile text about {topic}. Keep it concise and {tone}.",
    "paraphrase": "Paraphrase the following message while preserving intent and tone: {text}",
    "obfuscation": "Rewrite the message with subtle obfuscations (zero-width/homoglyph hints) but keep it readable: {text}",
}

DEFAULT_MODEL_DIR = "red_agent/models/t5_phish"


def model_available(path: str | Path = DEFAULT_MODEL_DIR) -> bool:
    """Return True if a saved model/ tokenizer folder exists."""
    p = Path(path)
    return p.exists() and (p / "config.json").exists() and (p / "tokenizer_config.json").exists()


def load_model(path: str | Path = DEFAULT_MODEL_DIR) -> Tuple:
    """Load tokenizer and model from a local directory."""
    path = str(path)
    tok = AutoTokenizer.from_pretrained(path)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(path)
    mdl.eval()
    return tok, mdl


def _build_prompt(
    mode: str,
    topic: str = "account verification",
    dept: str = "IT",
    text: str = "Please review your account settings at the portal.",
    tone: str = "professional",
    is_phishing: bool = True,
) -> str:
    if mode == "email":
        return PROMPTS["email_phishing" if is_phishing else "email_legit"].format(topic=topic, dept=dept, tone=tone)
    if mode == "sms":
        return PROMPTS["sms_phishing" if is_phishing else "sms_legit"].format(topic=topic, tone=tone)
    if mode == "paraphrase":
        return PROMPTS["paraphrase"].format(text=text)
    if mode == "obfuscation":
        return PROMPTS["obfuscation"].format(text=text)
    # default fallback
    return PROMPTS["email_phishing" if is_phishing else "email_legit"].format(topic=topic, dept=dept, tone=tone)


def generate_text(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int = 96,
    temperature: float = 0.9,
    top_p: float = 0.95,
    do_sample: bool = True,
) -> str:
    """Run T5 generation on CPU."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        top_p=top_p,
        temperature=temperature,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def generate_with_local_model(
    mode: str,
    model_path: str | Path = DEFAULT_MODEL_DIR,
    topic: str = "account verification",
    dept: str = "IT",
    text: str = "Please review your account settings at the portal.",
    tone: str = "professional",
    max_new_tokens: int = 96,
    is_phishing: bool = True,
) -> str:
    """
    Convenience wrapper: builds a prompt and generates text using a local T5 folder.
    """
    tok, mdl = load_model(model_path)
    prompt = _build_prompt(mode, topic=topic, dept=dept, text=text, tone=tone, is_phishing=is_phishing)
    return generate_text(tok, mdl, prompt, max_new_tokens=max_new_tokens)


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Generate text with tiny local T5.")
    ap.add_argument("--model_path", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--mode", choices=["email", "sms", "paraphrase", "obfuscation"], default="email")
    ap.add_argument("--topic", default="account verification")
    ap.add_argument("--dept", default="IT")
    ap.add_argument("--text", default="Please review your account settings at the portal.")
    ap.add_argument("--tone", default="professional")
    ap.add_argument("--max_new_tokens", type=int, default=96)
    args = ap.parse_args()

    if not model_available(args.model_path):
        print(f"[ERROR] Model not found at: {args.model_path}. Train it first with red_agent/train_t5.py")
        return

    prompt = _build_prompt(args.mode, topic=args.topic, dept=args.dept, text=args.text, tone=args.tone)
    tok, mdl = load_model(args.model_path)
    out = generate_text(tok, mdl, prompt, max_new_tokens=args.max_new_tokens)
    print(out)


if __name__ == "__main__":
    _cli()
