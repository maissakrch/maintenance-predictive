# 🔧 Maintenance Prédictive – Prédiction de défaillance moteur (CMAPSS)

## 🎯 Objectif du projet

Ce projet a pour objectif de concevoir une solution complète de **maintenance prédictive** pour des moteurs d’avion, en se basant sur des données de capteurs simulées issues du dataset **CMAPSS (NASA)**.  
Le but est de **prédire le RUL (Remaining Useful Life)** de chaque moteur afin d'anticiper les pannes et optimiser les opérations de maintenance.

---

## 🧱 Architecture du projet

- `data/` : données brutes et données nettoyées
- `src/` : scripts de traitement, entraînement, prédiction, explicabilité
- `api/` : API Flask permettant de faire des prédictions
- `dashboard/` : dashboard interactif avec Dash
- `Dockerfile` : containerisation de l'application
- `requirements.txt` : dépendances Python

---

## 🛠️ Fonctionnalités

- Nettoyage et structuration du jeu de données CMAPSS
- Entraînement d’un modèle RandomForestRegressor
- API Flask pour effectuer des prédictions
- Dashboard interactif avec Dash pour visualisation
- Analyse d’explicabilité avec SHAP
- Organisation modulaire du projet + versionnage Git

---

## 🧪 Stack technique

- Python
- Pandas, Scikit-learn, SHAP
- Flask, Dash
- Docker
- Git, VS Code, Jupyter

---

## 👩‍💻 Réalisé par

Maïssa Kerchaoui – Projet de certification RNCP Développeuse IA (Simplon)
