import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt

# Chargement des données
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(BASE_DIR, "data", "cleaned", "train_FD001_cleaned.csv")
df = pd.read_csv(data_path)

X = df.drop(columns=["RUL", "unit", "cycle", "max_cycle"])
y = df["RUL"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Dictionnaire des modèles
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
}

results = {}

# Entraînement et évaluation de chaque modèle
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results[name] = {"MAE": mae, "RMSE": rmse}
    print(f"\n📊 {name} — MAE: {mae:.2f} | RMSE: {rmse:.2f}")

# Graphique comparatif
maes = [results[model]["MAE"] for model in models]
rmses = [results[model]["RMSE"] for model in models]

plt.figure(figsize=(10, 6))
x_labels = list(models.keys())
plt.bar(x_labels, maes, alpha=0.6, label='MAE')
plt.bar(x_labels, rmses, alpha=0.6, label='RMSE')
plt.ylabel("Erreur")
plt.title("Comparaison des performances des modèles")
plt.legend()
plt.tight_layout()

# Sauvegarde du graphique
graph_path = os.path.join(BASE_DIR, "rapport", "comparaison_modeles.png")
plt.savefig(graph_path)
plt.show()

print(f"[✅] Graphique sauvegardé dans : {graph_path}")
