# Airflow x MLflow

Ce projet présente un pipeline complet de machine learning qui intègre :
- Ingestion et prétraitement des données
- Entraînement et validation de modèles (régression logistique et random forest)
- Orchestration avec airflow
- Suivi des expérimentations et gestion des modèles avec MLflow
- Déploiement du modèle via une api rest
- Conteneurisation de l'environnement avec docker

## Structure du projet

La structure du projet :
```
.
├── airflow_dags
│   └── pipeline_ml_dag.py
├── data
│   ├── X_train.npy, X_test.npy, X.npy
│   ├── y_train.npy, y_test.npy, y.npy
├── Dockerfile
├── mlruns
├── models
│   ├── logistic_regression.onnx
│   └── random_forest.onnx
├── requirements.txt
├── src
│   ├── data_ingestion.py
│   ├── preprocessing.py
│   ├── project_paths.py
│   ├── train_lr.py
│   ├── train_rf.py
│   └── validate.py
└── test_server.py
```

## Prérequis

- python 3.10
- docker
- airflow
- mlflow
- git

## Installation et configuration

1. Cloner le dépôt et se placer dans le répertoire :
   `git clone https://github.com/lukalafaye/airflow-mlflow-tutorial`
   `cd projet_ml`
2. Créer et activer l'environnement virtuel :
   `python3 -m venv .venv`
   `source .venv/bin/activate`
3. Installer les dépendances :
   `pip install -r requirements.txt`

## Utilisation

### Airflow

Initialiser la base de données et le compte admin :
```
airflow db init
airflow users create \
  --username admin \
  --firstname FIRST_NAME \
  --lastname LAST_NAME \
  --role Admin \
  --email admin@example.com
```
      
Modifier les paramètres de `airflow/airflow.cfg`:
```
dags_folder = /chemin/vers/le/dossier/airflow_dags
load_examples = False
```

Lancer airflow: `airflow standalone` et accéder à http://localhost:8080 pour déclencher manuellement le dag.

### MLflow

Lancer MLflow avec `mlflow ui` et consulter http://localhost:5000 pour suivre les runs, paramètres, métriques et modèles.

### Docker

1. Construire l'image :
   `docker build -t airflow .`
2. Lancer un conteneur :
   `docker run -it --name projet_ml_container airflow`
   puis dans le conteneur, exécuter les scripts Python de `src/`

### Déploiement du modèle

Pour servir le modèle via MLflow, exécutez :
   `export MLFLOW_TRACKING_URI=http://localhost:5000`
   `mlflow models serve -m "models:/iris_best_model/1" --port 1234 --no-conda`
Cela lance une api rest sur le port 1234.

### Test de l'api

Exemple avec python (test_server.py) :
```py
import requests, json
data = {"inputs": [[5.1, 3.5, 1.4, 0.2], [6.2, 2.8, 4.8, 1.8]]}
response = requests.post("http://localhost:1234/invocations",
                        headers={"Content-Type": "application/json"},
                        data=json.dumps(data))
print(response.json())
```

ou avec curl :
```sh
curl -X POST http://localhost:1234/invocations \
    -H "Content-Type: application/json" \
    -d '{"inputs": [[5.1, 3.5, 1.4, 0.2], [6.2, 2.8, 4.8, 1.8]]}'
```

## Notes

- Le modèle est logué et enregistré dans MLflow ; consultez l'interface web pour vérifier que paramètres, métriques et modèles sont corrects.
- La gestion des états dans Mlflow est remplacée par des alias et tags.