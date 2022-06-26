import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import f1_score
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import KernelPCA


def load_data():
    X = pd.read_csv('../../data/raw/train_data.csv')
    y = pd.read_csv('../../data/raw/train_labels.csv')
    return X, y


def min_max_scaler(X_train, X_test):
    scaler = MinMaxScaler(clip=True, feature_range=(-1.0, 1.0))
    scaler.fit(X_train)
    joblib.dump(scaler, '../../models/min_max_scaler.pkl')
    return scaler.transform(X_train), scaler.transform(X_test)


def add_sample(X_train, y_train):
    sm = SMOTE()
    return sm.fit_resample(X_train, y_train)


def dimension_reduction(X_train, y_train, X_test):
    kpca = KernelPCA(n_components=90, gamma=None, kernel="linear")
    kpca.fit(X_train, y_train)
    joblib.dump(kpca, '../../models/kpca_dimension_reduction.pkl')
    return kpca.transform(X_train), kpca.transform(X_test)


def main():
    X, y = load_data()

    cv = KFold(n_splits=5, shuffle=True)

    f1_scores = []
    knn = KNeighborsClassifier(algorithm='brute', leaf_size=39, metric='manhattan',
                      n_jobs=1, n_neighbors=7, p=3.198659538429792,
                      weights='distance')

    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        print('Sampling ...')
        X_train, y_train = add_sample(X_train, y_train)

        print('Dimension reduction ...')
        X_train, X_test = dimension_reduction(X_train, y_train, X_test)

        print('Min Max Scaler ...')
        X_train, X_test = min_max_scaler(X_train, X_test)

        print('Train model ...')
        knn.fit(X_train, np.ravel(y_train))
        y_pred = knn.predict(X_test)
        f1_scores.append(f1_score(np.ravel(y_test), y_pred))

    joblib.dump(knn, '../../models/knn_2.pkl')

    print(f1_scores)
    print(np.mean(f1_scores))


if __name__ == '__main__':
    main()
