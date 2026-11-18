# AgriSmart – AI/ML Crop Recommendation System

Recommends the best crop to grow from soil nutrients (N, P, K), pH, temperature,
humidity and rainfall, using a Random Forest classifier, with a Flask web UI.

## Structure
```
data/generate_dataset.py   Creates a synthetic dataset (data/crop_data.csv)
src/train.py               Trains + evaluates the model -> models/crop_model.joblib
src/predict.py             CLI / importable prediction helper
app/app.py                 Flask web app (+ /api/predict JSON endpoint)
app/templates/index.html   Frontend form
```

## Run
```bash
pip install -r requirements.txt
python data/generate_dataset.py
python src/train.py
python app/app.py          # open http://127.0.0.1:5000
```

CLI: `python src/predict.py 90 42 43 21 82 6.5 200`
(N P K temperature humidity ph rainfall)

## Using real data
Replace `data/crop_data.csv` with the Kaggle "Crop Recommendation Dataset"
(columns: N,P,K,temperature,humidity,ph,rainfall,label) and re-run `src/train.py`.
