from sklearn import linear_model
import numpy as np 
from skl2onnx import to_onnx
import os 
import mlflow
from mlflow import MlflowClient

from project_paths import DATA_DIR, MODELS_DIR

def run():
    mlflow.sklearn.autolog()

    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    print("Train data loaded.")

    reg = linear_model.LogisticRegression(random_state=42)

    with mlflow.start_run() as run:
        reg = reg.fit(X_train, y_train)

    print("Logistic finished training.")

    onx = to_onnx(reg, X_train[:1].astype(np.float32), target_opset=12)

    onnx_path = os.path.join(MODELS_DIR, "logistic_regression.onnx")

    with open(onnx_path, "wb") as f:
        f.write(onx.SerializeToString()) # want to save in MODELS_DIR
    print("Saved onnx model.")


if __name__ == "__main__":
    run()