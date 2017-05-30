from __future__ import print_function
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd
import numpy as np
import xgboost as xgb

import utils
from cross_validation import oversample_cv_data
import feature_selection


def get_data():
    print("read train and test data...")
    # features = feature_selection.get_feature_names()
    features = None
    X_train, y_train = utils.get_train_data(features)
    X_test, X_test_id = utils.get_test_data(features)

    return X_train, y_train, X_test, X_test_id


def LR_model_tfidf():
    lr = LogisticRegression(
        penalty='l1',
        C=0.1,
        verbose=1,
        n_jobs=16
    )

    X_train, y_train = utils.get_tfidf_train_data()

    train_x, valid_x, train_y, valid_y = train_test_split(X_train, y_train, test_size=0.2, random_state=1024)
    train_x, train_y = utils.resample_for_sparse(train_x, train_y)
    valid_x, valid_y = utils.resample_for_sparse(valid_x, valid_y)

    lr.fit(train_x, train_y)
    print("training log_loss: ")
    pred = lr.predict_proba(train_x)
    print(log_loss(train_y, pred))

    print("validation log_loss: ")
    pred = lr.predict_proba(valid_x)
    print(log_loss(valid_y, pred))

    pred_train = lr.predict_proba(X_train)[:, 1]
    pd.DataFrame(pred_train).to_csv("../../data/feature/lr_tfidf_train.csv", index=False, header=["lr_tfidf"])

    X_test, X_test_id = utils.get_tfidf_test_data()
    print("predicting...")
    pred_test = lr.predict_proba(X_test)[:, 1]
    pd.DataFrame(pred_test).to_csv("../../data/feature/lr_tfidf_test.csv", index=False, header=["lr_tfidf"])

    res = pd.DataFrame({"test_id": X_test_id, "is_duplicate": pred_test})
    write_result(res, 'lr_tfidf')


def LR_model_bow():
    lr = LogisticRegression(
        penalty='l1',
        C=0.1,
        verbose=1,
        n_jobs=16
    )

    X_train, y_train = utils.get_bow_train_data()

    train_x, valid_x, train_y, valid_y = train_test_split(X_train, y_train, test_size=0.2, random_state=1024)
    train_x, train_y = utils.resample_for_sparse(train_x, train_y)
    valid_x, valid_y = utils.resample_for_sparse(valid_x, valid_y)

    lr.fit(train_x, train_y)
    print("training log_loss: ")
    pred = lr.predict_proba(train_x)
    print(log_loss(train_y, pred))

    print("validation log_loss: ")
    pred = lr.predict_proba(valid_x)
    print(log_loss(valid_y, pred))

    pred_train = lr.predict_proba(X_train)[:, 1]
    pd.DataFrame(pred_train).to_csv("../../data/feature/lr_bow_train.csv", index=False, header=["lr_bow"])

    X_test, X_test_id = utils.get_bow_test_data()
    print("predicting...")
    pred_test = lr.predict_proba(X_test)[:, 1]
    pd.DataFrame(pred_test).to_csv("../../data/feature/lr_bow_test.csv", index=False, header=["lr_bow"])

    res = pd.DataFrame({"test_id": X_test_id, "is_duplicate": pred_test})
    write_result(res, 'lr_bow')


def LR_model_svd():
    lr = LogisticRegression(
        # penalty='l1',
        # C=0.1,
        verbose=1,
        n_jobs=16
    )

    X_train, y_train = utils.get_svd_train_data()
    X_train = pd.DataFrame(X_train)

    train_x, valid_x, train_y, valid_y = train_test_split(X_train, y_train, test_size=0.2, random_state=1024)
    train_x, train_y = utils.resample(train_x, train_y)
    valid_x, valid_y = utils.resample(valid_x, valid_y)

    lr.fit(train_x, train_y)
    print("training log_loss: ")
    pred = lr.predict_proba(train_x)
    print(log_loss(train_y, pred))

    print("validation log_loss: ")
    pred = lr.predict_proba(valid_x)
    print(log_loss(valid_y, pred))

    pred_train = lr.predict_proba(X_train)[:, 1]
    pd.DataFrame(pred_train).to_csv("../../data/feature/lr_svd_train.csv", index=False, header=["lr_svd"])

    # X_test, X_test_id = utils.get_svd_test_data()
    # print("predicting...")
    # pred_test = lr.predict_proba(X_test)[:, 1]
    # pd.DataFrame(pred_test).to_csv("../../data/feature/lr_svd_test.csv",index=False,header=['lr_svd'])
    #
    # res = pd.DataFrame({"test_id": X_test_id, "is_duplicate": pred_test})
    # write_result(res, 'lr_svd')


def RF_model():
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=15,
        min_samples_split=16,
        n_jobs=16,
    )

    X_train, y_train, X_test, X_test_id = get_data()
    RX_train, Ry_train = utils.resample(X_train, y_train)

    print("training model...")
    rf.fit(RX_train, Ry_train)

    print("predicting train data...")
    y_pred = rf.predict_proba(X_train)
    res = pd.DataFrame({'rf_predict': y_pred[:, 1]})
    res.to_csv("../../data/feature/train_rf.csv", index=False)

    print("predicting test data...")
    y_pred = rf.predict_proba(X_test)
    res = pd.DataFrame({'rf_predict': y_pred[:, 1]})
    res.to_csv("../../data/feature/test_rf.csv", index=False)

    res = pd.DataFrame({"test_id": X_test_id, "is_duplicate": y_pred[:, 1]})
    write_result(res, 'rf')


def calibrated_xgb():
    param = {
        "learning_rate": 0.015,
        "n_estimators": 500,
        "max_depth": 8,
        # "min_child_weight": 1,
        # "gamma": 0,
        "subsample": 0.7,
        "colsample_bytree": 0.6,
        "objective": "binary:logistic",
        "nthread": 16,
    }

    X_train, y_train, X_test, X_test_id = get_data()
    cv = oversample_cv_data(X_train, y_train)

    xgb = XGBClassifier()
    xgb.set_params(**param)
    clf_isotonic = CalibratedClassifierCV(xgb, cv=cv, method='isotonic')
    clf_isotonic.fit(X_train, y_train)
    y_pred = clf_isotonic.predict_proba(X_test)[:, 1]

    res = pd.DataFrame({"test_id": X_test_id, "is_duplicate": y_pred})
    write_result(res, 'cali')

def xgb_model():
    param = {
        "booster": 'gbtree',
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 6,
        "subsample": 0.7,
        "colsample_bytree": 0.6,
        "eta": 0.015,
        "lambda": 50,
        "nthread": 16,
        # "seed": 1024,
        "silent": 1,
    }

    X_train, y_train, X_test, X_test_id = get_data()
    train_x, valid_x, train_y, valid_y = train_test_split(X_train, y_train, test_size=0.2, random_state=1024)
    train_x, train_y = utils.resample(train_x, train_y)
    valid_x, valid_y = utils.resample(valid_x, valid_y)

    dtrain = xgb.DMatrix(train_x, label=train_y)
    dvalid = xgb.DMatrix(valid_x, label=valid_y)
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

    dtest = xgb.DMatrix(X_test)

    print("train xgb model...")
    bst = xgb.train(param, dtrain, 500, watchlist, early_stopping_rounds=50, verbose_eval=50)

    print("predicting...")
    y_pred = bst.predict(dtest)

    res = pd.DataFrame({"test_id": X_test_id, "is_duplicate": y_pred})
    write_result(res, 'xgb')


def write_result(res, name):
    print("writing result...")

    res.set_index('test_id', inplace=True)
    res.to_csv("../../results/submission/" + name + "_submission.csv")


if __name__ == '__main__':
    # LR_model_tfidf()
    # LR_model_bow()
    # LR_model_svd()
    RF_model()
    # xgb_model()
