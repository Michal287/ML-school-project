import pandas as pd
import joblib


def load_data(path):
    return pd.read_csv(path)


def min_max_scaler(X):
    scaler = joblib.load('../../models/min_max_scaler.pkl')
    return scaler.transform(X)


def dimension_reduction(X):
    kpca = joblib.load('../../models/kpca_dimension_reduction.pkl')
    return kpca.transform(X)


def predict(X, path):
    knn = joblib.load(path)
    return knn.predict(X)


def export(labels, path):
    pd.DataFrame(labels).to_csv(path, header=None, index=None)


def main():
    X_test = load_data('../../data/raw/test_data.csv')
    X_test = dimension_reduction(X_test)
    X_test = min_max_scaler(X_test)

    export(predict(X_test, '../../models/knn_2.pkl'), "../../data/export/predicted_labels_2.csv")


if __name__ == '__main__':
    main()
