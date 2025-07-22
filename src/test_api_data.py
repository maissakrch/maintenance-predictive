import pandas as pd

# 1. Charger le fichier des données nettoyées
df = pd.read_csv("data/cleaned/train_FD001_cleaned.csv")

# 2. Supprimer les colonnes inutiles
df = df.drop(columns=["RUL", "unit", "cycle", "max_cycle"])

# 3. Prendre une ligne au hasard
ligne = df.sample(1).values.tolist()[0]

# 4. Afficher les valeurs pour tester l'API
print(ligne)
print("Nombre de valeurs :", len(ligne))
