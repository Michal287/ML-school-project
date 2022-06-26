import pandas as pd
import joblib
import numpy as np


def load_data():
    """
    Load data
    """

    return pd.read_csv(r'../../data/raw/test_data.csv', header=None)


def dimension_reduction(X: pd.DataFrame) -> np.arrray:
    """
    Load model to reduction dimension and than transform data

    :X: pd.DataFrame: - df which we want transform
    """

    kpca = joblib.load('../../models/kpca_dimension_reduction.pkl')
    return kpca.transform(X)


def to_hdf(path: str, dataset_name: str, data: np.arrray):
    """
        Saving data to hdf

        :path: str: - path where to save data
        :dataset_name: str: - key to save data
        :data: np.arrray: - values to save
    """

    df = pd.DataFrame(data=data)
    df.to_hdf(path, key=dataset_name)


def main():
    """
        1. Load data
        3. Dimension reduction
        4. Save data as hdf
    """
    X = dimension_reduction(load_data())
    to_hdf('../../data/processed/test_data.h5', 'X', X)


if __name__ == '__main__':
    main()
