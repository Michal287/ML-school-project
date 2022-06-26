import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import KernelPCA
import joblib


def load_data():
    train_data = pd.read_csv(r'../../data/raw/train_data.csv', header=None)
    train_labels = pd.read_csv(r'../../data/raw/train_labels.csv', names=['target'], header=None)

    return train_data, train_labels


def add_sample(X: pd.DataFrame, y: pd.DataFrame):
    sm = SMOTE()
    return sm.fit_resample(X, y)


def dimension_reduction(X, y):
    kpca = KernelPCA(n_components=90, gamma=None, kernel="linear")
    X = kpca.fit_transform(X, y)
    joblib.dump(kpca, '../../models/kpca_dimension_reduction.pkl')
    return X


def to_hdf(path, dataset_name, data):
    df = pd.DataFrame(data=data)
    df.to_hdf(path, key=dataset_name)


def main():
    X, y = load_data()
    X, y = add_sample(X, y)
    X = dimension_reduction(X, y)
    to_hdf('../../data/processed/train_data.h5', 'X', X)
    to_hdf('../../data/processed/train_labels.h5', 'y', y)


if __name__ == '__main__':
    main()
