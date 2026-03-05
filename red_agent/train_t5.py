# red_agent/train_t5.py
from __future__ import annotations

import argparse
import os
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

def build_tiny_parallel() -> Dataset:
    """
    Tiny, sandboxed parallel set for quick CPU training.
    Keeps outputs generic (non-operational) to avoid misuse.
    """
    pairs = [
        ("Write a phishing-style email about payroll verification",
         "Notice: Your payroll account needs verification to remain active."),
        ("Create a short SMS about a generic bank alert",
         "Bank notice: unusual activity detected. Check status at short.link/xyz."),
        ("Draft a password-reset style email from IT",
         "IT: A password reset is queued. Review your settings to continue."),
        ("Delivery issue notification (SMS length)",
         "Delivery update: address confirmation required. Visit short.link/abc."),
        ("Generic account security reminder",
         "Security reminder: review your account preferences at the portal."),
        ("Subscription billing confirmation message",
         "Billing confirmation pending. Visit your profile to confirm details."),
    ]
    df = pd.DataFrame(pairs, columns=["src", "tgt"])
    return Dataset.from_pandas(df)

def tokenize_pair(tokenizer, batch, max_src=96, max_tgt=128):
    enc = tokenizer(batch["src"], max_length=max_src, truncation=True)
    lab = tokenizer(text_target=batch["tgt"], max_length=max_tgt, truncation=True)
    enc["labels"] = lab["input_ids"]
    return enc

def train_small_t5(model_name: str, out_dir: str, epochs: int, batch_size: int) -> None:
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    ds = build_tiny_parallel()
    ds_tok = ds.map(
        lambda b: tokenize_pair(tokenizer, b),
        batched=True,
        remove_columns=ds.column_names,
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    args_tr = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=5e-4,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        no_cuda=True,          # Force CPU (Windows-friendly)
        dataloader_num_workers=0,
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=args_tr,
        train_dataset=ds_tok,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[OK] Saved tiny T5 to {os.path.abspath(out_dir)}")

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a tiny T5 generator (CPU-friendly).")
    p.add_argument("--model_name", default="t5-small")
    p.add_argument("--model_out",  default="red_agent/models/t5_phish")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    return p

def main() -> None:
    args = _build_parser().parse_args()
    train_small_t5(args.model_name, args.model_out, args.epochs, args.batch_size)

if __name__ == "__main__":
    main()
