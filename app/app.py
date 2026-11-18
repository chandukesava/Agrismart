import os, sys
from flask import Flask, render_template, request, jsonify
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from predict import predict_crop, FEATURES

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html", features=FEATURES)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True, silent=True) or {}
    try:
        values = [float(data[f]) for f in FEATURES]
    except (KeyError, TypeError, ValueError):
        return jsonify(error=f"Provide numeric values for: {', '.join(FEATURES)}"), 400
    results = predict_crop(values)
    return jsonify(best=results[0][0], top=[{"crop": c, "probability": p} for c, p in results])

if __name__ == "__main__":
    app.run(debug=True)
