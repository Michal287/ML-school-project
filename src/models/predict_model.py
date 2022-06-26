import numpy as np
import pandas as pd
import joblib


def load_data():
    """
        Load data
    """
    return pd.read_hdf('../../data/processed/test_data.h5')


def min_max_scaler(X: pd.DataFrame, path: str) -> np.array:
    """
        1. Load scaler model
        2. Transform data

        :X: pd.DataFrame:
        :path: str:
    """
    scaler = joblib.load(path)
    return scaler.transform(X)


def predict(X: np.array, path: str) -> np.array:
    """
        1. Load model
        2. Predicting value

        :X: pd.DataFrame:
        :path: str:
    """
    knn = joblib.load(path)
    return knn.predict(X)


def export(labels: np.array, path: str):
    """
    Export labels to csv
    """
    pd.DataFrame(labels).to_csv(path, header=None, index=None)


def main():
    """
        1. Load data
        2. Scaling data
        3. Predicting by loaded model
        4. Exporting to csv
    """
    X_test = load_data()
    X_test = min_max_scaler(X_test, '../../models/min_max_scaler.pkl')

    export(predict(X_test, '../../models/knn.pkl'), "../../data/export/predicted_labels.csv")


if __name__ == '__main__':
    main()