import os, sys, joblib, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
_model = None

def _load():
    global _model
    if _model is None:
        _model = joblib.load(os.path.join(ROOT, "models", "crop_model.joblib"))
    return _model

def predict_crop(values, top_k=3):
    """values: 7 numbers in FEATURES order. Returns [(crop, probability), ...]."""
    model = _load()
    probs = model.predict_proba(pd.DataFrame([values], columns=FEATURES))[0]
    ranked = sorted(zip(model.classes_, probs), key=lambda t: -t[1])[:top_k]
    return [(c, round(float(p), 3)) for c, p in ranked]

if __name__ == "__main__":
    if len(sys.argv) != 8:
        sys.exit("Usage: python src/predict.py N P K temperature humidity ph rainfall")
    for crop, p in predict_crop([float(a) for a in sys.argv[1:]]):
        print(f"{crop:10s} {p*100:5.1f}%")
