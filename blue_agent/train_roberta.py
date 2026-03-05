import argparse
from pathlib import Path
import pandas as pd
from datasets import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer


def load_subset(csv_path: Path, max_train=2000, max_val=500):
    df = pd.read_csv(csv_path)
    df = df[["text", "label"]].dropna()
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_df = df.iloc[:max_train]
    val_df = df.iloc[max_train:max_train + max_val]

    print(f"Loaded {len(train_df)} train, {len(val_df)} val samples.")
    return Dataset.from_pandas(train_df), Dataset.from_pandas(val_df)


def tokenize(batch, tokenizer):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )


def train_roberta_model(input_csv, output_dir, epochs, batch_size,
                        max_train_samples, max_val_samples):

    print("📥 Loading data subset...")
    train_ds, val_ds = load_subset(
        input_csv,
        max_train=max_train_samples,
        max_val=max_val_samples
    )

    print("📚 Loading tokenizer + model...")
    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilroberta-base",
        num_labels=2
    )

    print("🔄 Tokenizing...")
    train_ds = train_ds.map(lambda x: tokenize(x, tokenizer), batched=True)
    val_ds = val_ds.map(lambda x: tokenize(x, tokenizer), batched=True)

    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    print("⚙️ Training starting... (No evaluation strategy)")

    # --- UNIVERSALLY SUPPORTED TrainingArguments ---
    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_total_limit=1,      # works on all versions
        logging_steps=100,       # safe on all versions
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,     # allowed even without evaluation_strategy
        tokenizer=tokenizer
    )

    trainer.train()

    print("💾 Saving model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--model_out", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_train_samples", type=int, default=2000)
    parser.add_argument("--max_val_samples", type=int, default=500)

    args = parser.parse_args()

    train_roberta_model(
        input_csv=Path(args.input),
        output_dir=Path(args.model_out),
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples
    )
