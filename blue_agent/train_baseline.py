import argparse, joblib, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from pathlib import Path

def train_baseline(data_path: str, model_out: str | None = None):
    if model_out is None:
        model_out = "blue_agent/models/baseline.joblib"
    
    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path)
    X = df["text"].astype(str); y = df["label"].astype(int)

    pipe = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1,2))), ("clf", LogisticRegression(max_iter=300))])
    pipe.fit(X, y)
    print(classification_report(y, pipe.predict(X)))
    joblib.dump(pipe, model_out)
    print("Saved", model_out)

def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="CSV or folder with train data")
    p.add_argument("--model_out", default="blue_agent/models/baseline.joblib")
    return p

def main():
    args = build_parser().parse_args()
    train_baseline(args.input, args.model_out)

if __name__ == "__main__":
    main()