from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Charger le modèle
model_path = os.path.join(os.path.dirname(__file__), "../models/random_forest_rul.pkl")
model = joblib.load(model_path)

@app.route("/", methods=["GET"])
def formulaire():
    return render_template("form.html")

@app.route("/predict_web", methods=["POST"])
def predict_web():
    try:
        features = [float(request.form[f"f{i}"]) for i in range(20)]
        prediction = model.predict([features])[0]
        return render_template("result.html", prediction=round(float(prediction), 2))
    except Exception as e:
        return f"Erreur : {e}", 500

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(features)[0]
        return jsonify({"RUL prédite": round(float(prediction), 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
