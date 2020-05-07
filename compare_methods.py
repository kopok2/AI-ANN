# coding=utf-8
"""
Porównanie skuteczności metod uczenia maszynowego.

Klasyfikacja - czy klient banku spłaci pożyczkę.
Źródło danych: http://archive.ics.uci.edu/ml/machine-learning-databases/00350/
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder

VERBOSE = True


def load_dataset(path="bank.csv", verbose=True):
    print("Loading data...")
    dataset = pd.read_csv(path, dtype=str)
    if verbose:
        print(dataset.head())
        print(dataset.describe())
    return dataset


def split_dataset(dataset):
    print("Spliting dataset...")
    X = dataset.drop(['deposit', 'age', 'balance'], axis=1)
    y = dataset['deposit']
    z = dataset[['age', 'balance']]
    return X, y, z


if __name__ == "__main__":
    df = load_dataset(verbose=VERBOSE)
    # print(df)

    X, y, z = split_dataset(df)
    # print(X)
    # print(z)
    #print(y)
    z = z.replace(" ", "0.0", regex=True)
    z = z.apply(pd.to_numeric)

    # print(z)
    X = pd.DataFrame(OneHotEncoder().fit_transform(X).toarray())
    y = pd.DataFrame(OneHotEncoder().fit_transform(np.array(y).reshape(-1, 1)).toarray()).loc[:, 1]
    X = pd.concat([X, z], axis=1)
    out = pd.concat([X, z, y], axis=1)
    out.to_csv('bank_clean.csv')

    x_train, x_test, y_train, y_test = train_test_split(X, y)

    print("\n\n\n")
    print("printing x train")
    print(x_train)
    print("pritning test")
    print(y_train)
    print("Training models...")
    # for model in [DecisionTreeClassifier, GaussianNB, KNeighborsClassifier, SVC]:
    #     print("Training " + model.__name__)
    #     model_ = model()
    #     model_.fit(x_train, y_train)
    #     print("Evaluating model:")
    #     print("Training data:")
    #     print(classification_report(y_train, model_.predict(x_train)))
    #     print("Confusion matrix:")
    #     print(confusion_matrix(y_train, model_.predict(x_train)))
    #     print("Test data:")
    #     print(classification_report(y_test, model_.predict(x_test)))
    #     print("Confusion matrix:")
    #     print(confusion_matrix(y_test, model_.predict(x_test)))
