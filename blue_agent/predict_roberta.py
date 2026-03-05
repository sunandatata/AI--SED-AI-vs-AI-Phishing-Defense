# blue_agent/predict_roberta.py
import os
from typing import List, Tuple
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABELS = ["ham", "phish"]  # 0=ham, 1=phish

class RobertaBlue:
    def __init__(self, model_dir: str):
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(
                f"RoBERTa model directory not found: {model_dir}\n"
                "Train it with train_roberta.py or fix the path."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        self.device = torch.device("cpu")
        self.model.to(self.device)

    @torch.no_grad()
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Returns probability for class 1 (phish) for each text. Shape: [N]
        """
        enc = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.model(**enc).logits  # [N,2]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()  # [N,2]
        return probs[:, 1]  # prob of 'phish'

    @torch.no_grad()
    def predict(self, texts: List[str]) -> List[Tuple[str, float]]:
        """
        Returns list of (label, phish_prob)
        """
        p = self.predict_proba(texts)
        labels = np.where(p >= 0.5, "phish", "ham")
        return list(zip(labels.tolist(), p.tolist()))


# Convenience top-level functions (keeps app code simple)

_cached_model = None
_cached_path = None

def load_model(model_dir: str) -> RobertaBlue:
    global _cached_model, _cached_path
    if _cached_model is None or _cached_path != model_dir:
        _cached_model = RobertaBlue(model_dir)
        _cached_path = model_dir
    return _cached_model

def predict(text: str, model_dir: str) -> Tuple[str, float]:
    mdl = load_model(model_dir)
    return mdl.predict([text])[0]
