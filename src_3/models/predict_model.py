import pandas as pd
import joblib
import numpy as np


def load_data(path: str) -> pd.DataFrame:
    """
        Load data
    """
    return pd.read_csv(path)


def min_max_scaler(X: np.array) -> np.array:
    """
        1. Load scaler model
        2. Transform data

        :X: np.array:
    """
    scaler = joblib.load('../../models/min_max_scaler_3.pkl')
    return scaler.transform(X)


def dimension_reduction(X: np.array) -> np.array:
    """
        1. Load dimension_reduction model
        2. Transform data

        :X: np.array:
    """
    kpca = joblib.load('../../models/kpca_dimension_reduction.pkl')
    return kpca.transform(X)


def predict(X: np.array, path: str) -> np.array:
    """
        1. Load model
        2. Predicting value

        :X: np.array:
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
        2. Dimention reduction
        3. Min max scalling
        3. Predicting by loaded model
        4. Exporting to csv
    """
    X_test = load_data('../../data/raw/test_data.csv')
    #X_test = dimension_reduction(X_test)
    X_test = min_max_scaler(X_test)

    export(predict(X_test, '../../models/knn_3.pkl'), "../../data/export/predicted_labels_3.csv")


if __name__ == '__main__':
    main()
