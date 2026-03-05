# blue_agent/export_onnx.py
from __future__ import annotations

import argparse
from pathlib import Path
import sys

def export_to_onnx(model_in: str, onnx_out: str) -> str:
    """
    Convert a scikit-learn Pipeline (TF-IDF + LogisticRegression) saved with joblib
    to ONNX and verify it with onnxruntime. Returns output path.
    """
    try:
        import joblib
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import StringTensorType
    except Exception as e:
        print("[ERROR] Missing deps:", e)
        print("       Run: pip install onnx onnxruntime skl2onnx")
        sys.exit(1)

    in_path = Path(model_in)
    out_path = Path(onnx_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        print(f"[ERROR] Model not found: {in_path.resolve()}")
        sys.exit(1)

    print(f"[INFO] Loading baseline model: {in_path.resolve()}")
    model = joblib.load(in_path)

    print("[INFO] Converting to ONNX…")
    # ONNX expects a 2D string tensor for text input: [batch, 1]
    initial_types = [("text", StringTensorType([None, 1]))]
    onx = convert_sklearn(model, initial_types=initial_types)

    with open(out_path, "wb") as f:
        f.write(onx.SerializeToString())

    print(f"[OK] Saved ONNX to: {out_path.resolve()}")

    # ---- quick verification
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        sample = np.array([["Action needed: confirm your payroll account now"]], dtype=object)
        _ = sess.run(None, {input_name: sample})
        print("[OK] ONNX verified with onnxruntime.")
    except Exception as e:
        print("[WARN] onnxruntime verification failed:", e)

    return str(out_path)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export baseline (sklearn) to ONNX.")
    p.add_argument("--model_in",  default="blue_agent/models/baseline.joblib")
    p.add_argument("--onnx_out",  default="blue_agent/models/baseline.onnx")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    export_to_onnx(args.model_in, args.onnx_out)


if __name__ == "__main__":
    main()
