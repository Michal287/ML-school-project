import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import joblib


def load_data():
    X = pd.read_hdf('../../data/processed/train_data.h5')
    y = pd.read_hdf('../../data/processed/train_labels.h5')
    return X, y


def min_max_scaler(X_train, X_test):
    scaler = MinMaxScaler(clip=True, feature_range=(-1.0, 1.0))
    scaler.fit(X_train)
    joblib.dump(scaler, '../../models/min_max_scaler.pkl')
    return scaler.transform(X_train), scaler.transform(X_test)


def main():
    X, y = load_data()

    #X_train, X_test, y_train, y_test = train_test_split(X, np.ravel(y), test_size=0.2, shuffle=True)

    #X_train, X_test = min_max_scaler(X_train, X_test)

    cv = KFold(n_splits=10, shuffle=True)

    f1_scores = []
    knn = KNeighborsClassifier(algorithm='ball_tree', leaf_size=31, metric='manhattan',
                               n_jobs=1, n_neighbors=1, p=1.4528804003307858,
                               weights='distance')

    #knn.fit(X_train, y_train)
    #y_pred = knn.predict(X_test)

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
    #print(f1_score(y_test, y_pred))


if __name__ == '__main__':
    main()