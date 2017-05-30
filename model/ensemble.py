from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
import pandas as pd
import numpy as np

import utils
import predict

def voting():
    base_models = [
        RandomForestClassifier(n_estimators=500, max_depth=20, min_samples_split=16, n_jobs=16, random_state=1),
        RandomForestClassifier(n_estimators=500, max_depth=20, min_samples_split=16, n_jobs=16, random_state=100),
        XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.1,
                      objective='binary:logistic', nthread=16, seed=1),
        XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.1,
                      objective='binary:logistic', nthread=16, seed=100)
    ]

    vote_clf = VotingClassifier(estimators=zip(['RF1','RF2','XGB1','XGB2'],base_models),voting='soft')
    X_train, y_train = utils.get_train_data()
    X_train, y_train = utils.resample(X_train, y_train)

    X_test, X_test_id = utils.get_test_data()
    print("training voting model...")
    vote_clf.fit(X_train,y_train)
    print("predicting...")
    pred = vote_clf.predict_proba(X_test)

    res = pd.DataFrame({"test_id": X_test_id, "is_duplicate": pred[:, 1]})
    return res

def stacking(cv=5):
    base_models = [
        RandomForestClassifier(n_estimators=500, max_depth=20, min_samples_split=16, n_jobs=16,random_state=1),
        RandomForestClassifier(n_estimators=500, max_depth=20, min_samples_split=16, n_jobs=16,random_state=100),
        ExtraTreesClassifier(n_estimators=500, max_depth=20, min_samples_split=16, n_jobs=16),
        XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.1,
                      objective='binary:logistic', nthread=16)
    ]
    # stack_model = XGBClassifier(n_estimators=400, max_depth=4,
    #                             objective='binary:logistic', learning_rate=0.05, nthread=16)
    stack_model = LogisticRegression()

    X_train, y_train = utils.get_train_data()
    X_test, X_test_id = utils.get_test_data()

    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    n_estimators = len(base_models)

    middle_train = np.zeros((n_train, n_estimators))
    middle_test = np.zeros((n_test, cv * n_estimators))

    print("training base models, {} folds ...".format(cv))
    rs = KFold(n_splits=cv, shuffle=True, random_state=1024)
    i = 0
    for train_index, remain_index in rs.split(X_train):
        print("fold {}...".format(i+1))
        X_train_folds = X_train.ix[train_index]
        y_train_folds = y_train.ix[train_index]
        X_train_folds, y_train_folds = utils.resample(X_train_folds, y_train_folds)
        print("train fold shape", X_train_folds.shape)

        X_remain_folds = X_train.ix[remain_index]
        y_remain_folds = y_train.ix[remain_index]

        for j, model in enumerate(base_models):
            print("base model {}...".format(j+1))
            print("    training...")
            model.fit(X_train_folds, y_train_folds)

            print("    predict train ramain...")
            pred_train = model.predict_proba(X_remain_folds)[:, 1]
            middle_train[remain_index, j] = pred_train

            print("    predict test...")
            pred_test = model.predict_proba(X_test)[:, 1]
            middle_test[:, j * cv + i] = pred_test
        i += 1

    print("generat middle_test data...")
    for j in range(n_estimators):
        cols = range(j * cv, (j + 1) * cv)
        middle_test[:, i] = np.mean(middle_test[:, cols], axis=1)
    middle_test = middle_test[:, range(n_estimators)]

    print("training stack model and predecting...")
    stack_model.fit(middle_train, y_train)
    pred = stack_model.predict_proba(middle_test)

    res = pd.DataFrame({"test_id": X_test_id, "is_duplicate": pred[:, 1]})
    return res

def weighting():
    xgb_pred = pd.read_csv("../../results/submission/xgb_submission.csv")
    lstm_pred = pd.read_csv("../../data/feature/LSTM_test.csv")

    xgb_pred.is_duplicate = xgb_pred.is_duplicate * 0.9 + lstm_pred.is_duplicated * 0.1
    return xgb_pred

if __name__ == '__main__':
    # predict.write_result(stacking(), 'stacking')
    predict.write_result(voting(),'voting')
    # predict.write_result(weighting(),'weighting')