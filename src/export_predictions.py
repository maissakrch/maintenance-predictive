import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split

def load_data():
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(BASE_DIR, "data", "cleaned", "train_FD001_cleaned.csv")
    df = pd.read_csv(path)
    X = df.drop(columns=["RUL", "unit", "cycle", "max_cycle"])
    y = df["RUL"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    print("[🚀] Chargement des données...")
    X_train, X_test, y_train, y_test = load_data()

    print("[📦] Chargement du modèle...")
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(BASE_DIR, "models", "random_forest_rul.pkl")
    model = joblib.load(model_path)

    print("[🔮] Prédictions en cours...")
    y_pred = model.predict(X_test)

    print("[📤] Export vers CSV...")
    results_df = pd.DataFrame({
        "Valeur réelle": y_test.values,
        "Valeur prédite": y_pred
    })
    output_path = os.path.join(BASE_DIR, "rapport", "predictions_vs_reality.csv")
    results_df.to_csv(output_path, index=False)
    print(f"[✅] Fichier exporté : {output_path}")
