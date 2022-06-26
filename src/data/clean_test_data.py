import pandas as pd
import joblib


def load_data():
    return pd.read_csv(r'../../data/raw/test_data.csv', header=None)


def dimension_reduction(X):
    kpca = joblib.load('../../models/kpca_dimension_reduction.pkl')
    return kpca.transform(X)


def to_hdf(path, dataset_name, data):
    df = pd.DataFrame(data=data)
    df.to_hdf(path, key=dataset_name)


def main():
    X = dimension_reduction(load_data())
    to_hdf('../../data/processed/test_data.h5', 'X', X)


if __name__ == '__main__':
    main()

