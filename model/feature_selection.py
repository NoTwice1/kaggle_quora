# import xgboost as xgb
from __future__ import print_function
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import utils

RF_feature = "./RF_features.txt"
LR_feature = "./LR_features.txt"


def logistic_Select(filename):
    X_train, y_train = utils.get_train_data()
    X_train = X_train[:50000]
    y_train = y_train[:50000]
    X_train, y_train = utils.resample(X_train, y_train)

    print("select from logistic regression...")
    lr = LogisticRegression(penalty='l1', C=2)
    selector = SelectFromModel(lr, threshold=0.01)
    selector.fit(X_train, y_train)
    selected_features = selector.get_support()

    print("write results...")
    f = open(filename, "w")
    for i, name in enumerate(X_train.columns):
        if selected_features[i]:
            f.write(name + '\n')
    print("Done.")


def logistic_RFE(filename):
    X_train, y_train = utils.get_train_data()
    X_train = X_train[:50000]
    y_train = y_train[:50000]
    X_train, y_train = utils.resample(X_train, y_train)

    print("eliminating useless features...")
    estimator = LogisticRegression(penalty='l1', C=2)
    selector = RFE(estimator, 30, verbose=True)
    selector = selector.fit(X_train, y_train)

    print("writing results...")
    f = open(filename, "w")
    for i, feature_name in enumerate(X_train.columns):
        if selector.support_[i]:
            f.write(feature_name + '\n')
    print("Done.")


def RF_feature_importance(filename):
    X_train, y_train = utils.get_train_data()
    X_train = X_train[:50000]
    y_train = y_train[:50000]
    X_train, y_train = utils.resample(X_train, y_train)

    rf = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=8)
    rf.fit(X_train, y_train)

    print("writing results...")
    i = 0
    importance = []
    for col in enumerate(X_train.columns):
        importance.append([col, rf.feature_importances_[i]])
        i += 1
    importance.sort(key=lambda x: x[1], reverse=True)

    f = open(filename, "w")
    for col in importance:
        f.write(col[0][1] + "\t" + str(col[1]) + "\n")
    f.close()


def get_LR_feature(filename=LR_feature):
    features = []
    f = open(filename)
    for line in f:
        features.append(line.strip())
    return features


def get_RF_feature(n=30, filename=RF_feature):
    features = []
    f = open(filename)
    cnt = 0
    for line in f:
        cnt += 1
        if cnt <= n:
            features.append(line.split('\t')[0])
    return features


if __name__ == '__main__':
    # RF_feature_importance(RF_feature)
    # logistic_RFE(LR_feature)
    logistic_Select(LR_feature)
