import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Chargement du modèle
model = joblib.load("models/random_forest_rul.pkl")

# Chargement des données
df = pd.read_csv("data/cleaned/train_FD001_cleaned.csv")
X = df.drop(columns=["unit", "cycle", "RUL", "max_cycle"])

# Sélection de 100 exemples aléatoires
X_sample = X.sample(100, random_state=42)

# Création de l'explainer SHAP
explainer = shap.Explainer(model.predict, X_sample)

# Calcul des valeurs SHAP
shap_values = explainer(X_sample)

# Résumé global
shap.summary_plot(shap_values, X_sample)
