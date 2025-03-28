import os
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score
from onnxruntime import InferenceSession
from project_paths import DATA_DIR, MODELS_DIR
import onnx 

def run():
    model_files = {
        "logistic_regression": os.path.join(MODELS_DIR, "logistic_regression.onnx"),
        "random_forest": os.path.join(MODELS_DIR, "random_forest.onnx")
    }

    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy")).astype(np.float32)
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    best_accuracy = -1
    best_model_name = None
    best_model_path = None

    mlflow.set_tracking_uri("http://localhost:5000")

    with mlflow.start_run(run_name="model_comparison") as mlflow_run:

        for model_name, model_path in model_files.items():
            sess = InferenceSession(model_path, providers=["CPUExecutionProvider"])
            y_pred = sess.run(None, {"X": X_test})[0]

            acc = accuracy_score(y_test, y_pred)

            mlflow.log_metric(f"{model_name}_accuracy", acc)
            print(f"Model: {model_name}, Accuracy: {acc:.4f}")

            if acc > best_accuracy:
                best_accuracy = acc
                best_model_name = model_name
                best_model_path = model_path

        mlflow.log_param("best_model_name", best_model_name)
        mlflow.log_metric("best_model_accuracy", best_accuracy)

        onnx_model = onnx.load(best_model_path)
        mlflow.onnx.log_model(onnx_model, artifact_path="best_model")

        model_uri = f"runs:/{mlflow_run.info.run_id}/best_model"
        model_name = "iris_best_model"

        mlflow.register_model(model_uri, model_name)
        print(f"✅ Registered best model '{model_name}' (accuracy={best_accuracy:.4f}) to MLflow Registry.")

if __name__ == "__main__":
    run()
