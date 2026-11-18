"""Generate a synthetic crop dataset (replace with real data when available)."""
import numpy as np, pandas as pd, os

# crop: (N, P, K, temp, humidity, ph, rainfall) typical means
PROFILES = {
    "rice":      (80, 48, 40, 24, 82, 6.4, 230),
    "wheat":     (100, 45, 40, 20, 60, 6.8, 80),
    "maize":     (78, 48, 20, 23, 65, 6.2, 85),
    "cotton":    (120, 45, 20, 24, 80, 6.9, 80),
    "chickpea":  (40, 68, 80, 19, 17, 7.3, 80),
    "banana":    (100, 80, 50, 27, 80, 6.0, 105),
    "mango":     (20, 28, 30, 31, 50, 5.8, 95),
    "coffee":    (100, 28, 30, 25, 58, 6.8, 158),
    "jute":      (78, 46, 40, 25, 80, 6.7, 175),
    "lentil":    (19, 68, 20, 25, 65, 6.9, 46),
}
SPREAD = (8, 6, 6, 2.5, 6, 0.35, 25)

def main(n_per_class=200, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for crop, mean in PROFILES.items():
        x = rng.normal(mean, SPREAD, size=(n_per_class, 7))
        x = np.clip(x, [0,5,5,8,14,4,20], [140,145,205,44,100,9.5,300])
        df = pd.DataFrame(x, columns=["N","P","K","temperature","humidity","ph","rainfall"])
        df["label"] = crop
        rows.append(df)
    out = pd.concat(rows).sample(frac=1, random_state=seed).round(2)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crop_data.csv")
    out.to_csv(path, index=False)
    print(f"Saved {len(out)} rows -> {path}")

if __name__ == "__main__":
    main()
