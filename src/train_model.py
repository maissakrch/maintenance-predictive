import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

def load_data(_):
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(BASE_DIR, "data", "cleaned", "train_FD001_cleaned.csv")
    df = pd.read_csv(data_path)
    
    X = df.drop(columns=["RUL", "unit", "cycle", "max_cycle"])
    y = df["RUL"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"[✅] MAE : {mae:.2f}")
    print(f"[✅] RMSE : {rmse:.2f}")

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"[✅] Modèle sauvegardé dans : {path}")

if __name__ == "__main__":
    print("[🚀] Chargement des données...")
    X_train, X_test, y_train, y_test = load_data(None)

    print("[🧠] Entraînement du modèle...")
    model = train_model(X_train, y_train)

    print("[📊] Évaluation du modèle...")
    evaluate_model(model, X_test, y_test)

    print("[💾] Sauvegarde du modèle...")
    save_model(model, "../models/random_forest_rul.pkl")

    # 🔍 Génération du graphe réel vs prédit
    y_pred = model.predict(X_test)

    plt.figure(figsize=(10, 6))
    plt.plot(y_test[:100].values, label='Valeurs réelles')
    plt.plot(y_pred[:100], label='Valeurs prédites')
    plt.xlabel('Échantillons')
    plt.ylabel('RUL (nombre de cycles restants)')
    plt.title('Comparaison Réel vs Prédit')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("rapport/graph_prediction_vs_reality.png")
    plt.show()
