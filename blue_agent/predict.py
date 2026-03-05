import joblib, re, json, sys

CUES = {
 "urgency": ["urgent","immediately","now","expires","today","act fast","limited"],
 "authority": ["it","admin","security","hr","management","compliance","ceo"],
 "scarcity": ["limited","only","last chance","final notice"],
 "reciprocity": ["reward","gift","voucher","refund"],
 "fear": ["suspend","locked","breach","compromised","warning"],
 "curiosity": ["secret","confidential","exclusive","prize"]
}

def explain(text):
    cues = {}
    low = text.lower()
    for k, toks in CUES.items():
        hits = [t for t in toks if re.search(r"\b"+re.escape(t)+r"\b", low)]
        if hits: cues[k]=hits
    return cues

def load_model(path="blue_agent/models/baseline.joblib"):
    return joblib.load(path)

def predict(model, text, threshold=0.5):
    proba = float(model.predict_proba([text])[0,1])
    return {"proba": proba, "label": int(proba>=threshold), "tactics": explain(text)}

if __name__ == "__main__":
    model = load_model()
    text = sys.stdin.read().strip()
    print(json.dumps(predict(model, text), indent=2))