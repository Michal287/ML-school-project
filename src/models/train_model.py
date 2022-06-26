import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import joblib


def load_data():
    """
        Load data
    """
    X = pd.read_hdf('../../data/processed/train_data.h5')
    y = pd.read_hdf('../../data/processed/train_labels.h5')
    return X, y


def min_max_scaler(X_train: np.array, X_test: np.array) -> tuple:
    """
        1. Init scaler model
        2. Fit model
        3. Save to file
        4. transform

        :X_train: np.array:
        :X_test: np.array:
    """
    scaler = MinMaxScaler(clip=True, feature_range=(-1.0, 1.0))
    scaler.fit(X_train)
    joblib.dump(scaler, '../../models/min_max_scaler.pkl')
    return scaler.transform(X_train), scaler.transform(X_test)


def main():
    """
    1. Load data
    2. Init Kfold
    3. Init model
    4. Split to train and test
    5. Scale data
    6. Fitting model
    7. Preditction and save to list
    8. Saving model
    """

    X, y = load_data()

    cv = KFold(n_splits=10, shuffle=True)

    f1_scores = []
    knn = KNeighborsClassifier(algorithm='ball_tree', leaf_size=31, metric='manhattan',
                               n_jobs=1, n_neighbors=1, p=1.4528804003307858,
                               weights='distance')

    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        X_train, X_test = min_max_scaler(X_train, X_test)

        knn.fit(X_train, np.ravel(y_train))
        y_pred = knn.predict(X_test)
        f1_scores.append(f1_score(np.ravel(y_test), y_pred))

    joblib.dump(knn, '../../models/knn.pkl')

    print(f1_scores)
    print(np.mean(f1_scores))


if __name__ == '__main__':
    main()