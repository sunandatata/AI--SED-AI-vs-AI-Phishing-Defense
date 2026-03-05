CUE_WEIGHTS = {"urgency":0.30,"authority":0.20,"scarcity":0.15,"reciprocity":0.10,"fear":0.20,"curiosity":0.05}
CUES = {
 "urgency": ["urgent","immediately","expires","today","now"],
 "authority": ["it","admin","security","hr","management","compliance","ceo"],
 "scarcity": ["limited","only","final","last"],
 "reciprocity": ["reward","gift","voucher","refund"],
 "fear": ["suspend","locked","breach","compromised","warning"],
 "curiosity": ["secret","exclusive","confidential","prize"]
}
def persuasion_index(text:str)->float:
    low=text.lower(); score=0.0
    for k,toks in CUES.items():
        if any(t in low for t in toks): score+=CUE_WEIGHTS[k]
    return min(1.0, score)