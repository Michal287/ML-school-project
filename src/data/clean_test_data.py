import pandas as pd
import joblib
import h5py


def load_data():
    return pd.read_csv(r'../../data/raw/test_data.csv', header=None)


def dimension_reduction(X):
    kpca = joblib.load('../../models/kpca_dimension_reduction.pkl')
    return kpca.transform(X)


def to_hdf(path, dataset_name, data):
    with h5py.File(path, 'w') as hf:
        hf.create_dataset(dataset_name, data=data)


def main():
    X = dimension_reduction(load_data())
    to_hdf('../../data/processed/test_data.h5', 'X', X)


if __name__ == '__main__':
    main()

