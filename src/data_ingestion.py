from sklearn import datasets
import numpy as np 
from project_paths import DATA_DIR
import os

def run():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    print("Downloaded iris dataset.")

    np.save(os.path.join(DATA_DIR, "X.npy"), X)
    np.save(os.path.join(DATA_DIR, "y.npy"), y)
    print("Saved iris dataset X/y to disk.") 

if __name__ == "__main__":
    run()