from __future__ import print_function
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import GridSearchCV, train_test_split, ShuffleSplit
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import log_loss
from functools import partial
import json
import warnings

from utils import get_train_data, resample
import feature_selection

warnings.filterwarnings("ignore", category=DeprecationWarning)


params_init = {
    'LR': {"C": 2, "penalty": 'l2'},
    'SVC':{
        "C":1
    },
    'RFC': {
        'n_estimators': 1000,
        'max_depth': 15,
        'min_samples_split':16,
        'n_jobs': 16,
    },
    'ETC': {
        'n_estimators': 1000,
        'min_samples_leaf': 2,
        'max_depth': 15,
        'n_jobs': 16,
    },
    'XGB': {
        "booster": 'gbtree',
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 8,
        # "subsample": 0.7,
        # "colsample_bytree": 0.6,
        "eta": 0.02,
        # "lambda": 50,
        "nthread": 16,
        # "seed": 1024,
        "silent": 1,
    },
    "XGBClassifier": {
        "learning_rate": 0.04,
        "n_estimators": 500,
        "max_depth": 8,
        # "min_child_weight": 1,
        # "gamma": 0,
        # "subsample": 0.8,
        # "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "nthread": 16,
        # "scale_pos_weight": 1,
    }
}

params_grid = {
    'LR': {
        "C": [1, 4, 8, 10, 50, 100],
        "penalty": ["l1", "l2"]
    },
    'RFC': {
        # 'n_estimators': [100,300,500,1000],
        'max_depth': [15, 20, 25, 30],
        'min_samples_split': [2, 8, 16, 24]
    },
    "ETC": {
        'max_depth': [15, 20, 25, 30],
        'min_samples_split': [2, 8, 16, 24]
    },
    "XGBClassifier": {
        "max_depth": range(2, 8, 2),
        "min_child_weight": range(3, 7, 2),

    }
}


def oversample_cv_data(X_train, y_train, cv=3):
    train_fold = []
    valid_fold = []

    rs = ShuffleSplit(n_splits=cv, test_size=0.2, random_state=1024)
    for train_index, valid_index in rs.split(X_train):
        _, _, train_index = resample(X_train.ix[train_index, 0], y_train.ix[train_index], train_index)
        _, _, valid_index = resample(X_train.ix[valid_index, 0], y_train.ix[valid_index], valid_index)
        train_fold.append(train_index)
        valid_fold.append(valid_index)

    return zip(train_fold, valid_fold)


def cross_validation(model, model_name):
    print("cross validation for " + model_name + " ...")
    features = None
    X_train, y_train = get_train_data(features)
    X_train = X_train[:50000]
    y_train = y_train[:50000]

    # model.set_params(**params_init[model_name])vim
    cv = oversample_cv_data(X_train, y_train)

    print(cross_val_score(model, X_train, y_train, cv=cv, scoring="log_loss"))


def grid_search(model, model_name):
    features = None
    X_train, y_train = get_train_data(features)
    X_train = X_train[:50000]
    y_train = y_train[:50000]

    model.set_params(**params_init[model_name])
    cv = oversample_cv_data(X_train, y_train)

    print("grid search for " + model_name + " ...")
    clf = GridSearchCV(model, params_grid[model_name], cv=cv, scoring="log_loss", verbose=2)
    clf.fit(X_train, y_train)

    print("best score: ", clf.best_score_)
    best_params = clf.best_params_
    print("best params: ")
    print(best_params)
    model.set_params(**best_params)

    print("write best params to json...")
    f = open("../../results/params/" + model_name + ".json", "w")
    json.dump(best_params, f)


def xgb_cv():
    # features = feature_selection.get_LR_feature()
    features = None
    X_train, y_train = get_train_data(features)

    X_train = X_train[:100000]
    y_train = y_train[:100000]

    X_train, y_train = resample(X_train, y_train)

    print("xgb cv...")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    cv_log = xgb.cv(params_init['XGB'], dtrain, num_boost_round=500, nfold=3, metrics='logloss',
                    early_stopping_rounds=50, seed=1024, verbose_eval=50)

    print('test_logloss: ', cv_log['test-logloss-mean'].min())
    print('train_loglss: ', cv_log['train-logloss-mean'].min())


if __name__ == '__main__':
    rf = RandomForestClassifier()
    # grid_search(rf, 'RFC')
    cross_validation(rf, 'RFC')
    # hyper_search(rf, 'RFC')

    # et = ExtraTreesClassifier()
    # grid_search(et, 'ETC')

    # lr = LogisticRegression()
    # cross_validation(lr, 'LR')
    # grid_search(lr, 'LR')
    # hyper_search(lr, "LR")

    n_es = 10
    clf = BaggingClassifier(SVC(probability=True), max_samples=1.0/n_es, n_estimators=n_es, n_jobs=n_es)
    cross_validation(clf,'SVC')


    # xgb = XGBClassifier()
    # grid_search(xgb, "XGBClassifier")
    # cross_validation(xgb, "XGBClassifier")


    # xgb_cv()
