from sklearn.model_selection import train_test_split
import numpy as np 
from project_paths import DATA_DIR
import os

def run():
    X = np.load(os.path.join(DATA_DIR, "X.npy"))
    y = np.load(os.path.join(DATA_DIR, "y.npy"))
    print("X/y iris data loaded from disk.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    np.save(os.path.join(DATA_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(DATA_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(DATA_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(DATA_DIR, "y_test.npy"), y_test)
    print("Saved iris dataset train/test to disk.")

if __name__ == "__main__":
    run()