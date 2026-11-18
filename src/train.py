import os, joblib, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

def main():
    df = pd.read_csv(os.path.join(ROOT, "data", "crop_data.csv"))
    X, y = df[FEATURES], df["label"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    print(f"Accuracy: {accuracy_score(yte, pred):.3f}\n")
    print(classification_report(yte, pred))
    imp = sorted(zip(FEATURES, model.feature_importances_), key=lambda t: -t[1])
    print("Feature importance:", ", ".join(f"{f}={v:.2f}" for f, v in imp))
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    joblib.dump(model, os.path.join(ROOT, "models", "crop_model.joblib"))
    print("Model saved -> models/crop_model.joblib")

if __name__ == "__main__":
    main()
