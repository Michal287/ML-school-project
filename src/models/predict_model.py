import pandas as pd
import joblib


def load_data(path):
    return pd.read_hdf(path)


def min_max_scaler(X, path):
    scaler = joblib.load(path)
    return scaler.transform(X)


def predict(X, path):
    knn = joblib.load(path)
    return knn.predict(X)


def export(labels, path):
    pd.DataFrame(labels).to_csv(path, header=None, index=None)


def main():
    X_test = load_data('../../data/processed/test_data.h5')
    X_test = min_max_scaler(X_test, '../../models/min_max_scaler.pkl')

    export(predict(X_test, '../../models/knn.pkl'), "../../data/export/predicted_labels.csv")


if __name__ == '__main__':
    main()
