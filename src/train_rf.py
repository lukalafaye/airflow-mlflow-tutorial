from sklearn.ensemble import RandomForestClassifier

import numpy as np 
import os 
from skl2onnx import to_onnx

import mlflow
from mlflow import MlflowClient

from project_paths import DATA_DIR, MODELS_DIR

def run():
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    print("Train data loaded.")

    clf = RandomForestClassifier(n_estimators=10, random_state=42)

    with mlflow.start_run() as run:
        clf = clf.fit(X_train, y_train)
    print("Random forest finished training.")

    onx = to_onnx(clf, X_train[:1].astype(np.float32), target_opset=12)

    onnx_path = os.path.join(MODELS_DIR, "random_forest.onnx")

    with open(onnx_path, "wb") as f:
        f.write(onx.SerializeToString())
    print("Saved onnx model.")

if __name__ == "__main__":
    run()