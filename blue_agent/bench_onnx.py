# blue_agent/bench_onnx.py
import time, numpy as np, onnxruntime as ort

def bench(session, texts, runs=200):
    name = session.get_inputs()[0].name
    t0 = time.time()
    for _ in range(runs):
        inp = np.array([[texts[_ % len(texts)]]], dtype=object)
        session.run(None, {name: inp})
    dt = time.time()-t0
    print(f"Runs: {runs}, Total: {dt:.3f}s, Avg: {dt/runs*1000:.2f} ms/inference")

if __name__ == "__main__":
    sess = ort.InferenceSession("blue_agent/models/baseline.onnx", providers=["CPUExecutionProvider"])
    samples = [
        "URGENT: Your account will be suspended today unless you verify now.",
        "Lunch at 1pm? I sent the notes yesterday.",
        "IT: Password reset pending. Complete immediately."
    ]
    bench(sess, samples, runs=300)
