from __future__ import division
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import sparse
import cPickle

scaler = None


def log(V):
    return np.log(V + 1)


def get_svd_train_data():
    print("loading svd...")
    X_train = cPickle.load(open("../../data/feature/svd_tfidf_train.pkl", "rb"))

    original_train = pd.read_csv("../../data/preprocessed_data/train.csv")
    y_train = original_train['is_duplicate']

    print("train data shape: ", X_train.shape)
    print("train label shape: ", y_train.shape)

    return X_train, y_train


def get_bow_train_data():
    print("loading tfidf...")

    X_train = sparse.hstack(
        [
            cPickle.load(open("../../data/feature/train_q1_bow.pkl", "rb")),
            cPickle.load(open("../../data/feature/train_q2_bow.pkl", "rb")),
            # pd.read_csv("../../data/feature/ngram_feature_train.csv"),
            # pd.read_csv("../../data/feature/diff_feature_train.csv"),
        ]
    )

    X_train = X_train.tocsr()

    original_train = pd.read_csv("../../data/preprocessed_data/train.csv")
    y_train = original_train['is_duplicate']

    print("train data shape: ", X_train.shape)
    print("train label shape: ", y_train.shape)

    return X_train, y_train


def get_bow_test_data():
    print("loading tfidf...")
    other_features = pd.concat(
        [
            pd.read_csv("../../data/feature/ngram_feature_test.csv"),
            pd.read_csv("../../data/feature/diff_feature_test.csv"),
        ]
    )

    original_test = pd.read_csv("../../data/preprocessed_data/test.csv")
    test_id = original_test['test_id']
    print("test id shape: ", test_id.shape)

    X_test = []
    begin = 0
    for i in range(10):
        X_test_part = cPickle.load(open("../../data/feature/test_bow" + str(i+1) + ".pkl", "rb"))
        rows = X_test_part.shape[0]
        end = begin + rows
        X_test_part = sparse.hstack([X_test_part, other_features[begin:end]])
        begin = end

        yield X_test_part, test_id[begin:end]


def get_tfidf_train_data():
    print("loading tfidf...")

    X_train = sparse.hstack(
        [
            cPickle.load(open("../../data/feature/train_q1_tfidf.pkl", "rb")),
            cPickle.load(open("../../data/feature/train_q2_tfidf.pkl", "rb")),
            pd.read_csv("../../data/feature/ngram_feature_train.csv"),
            pd.read_csv("../../data/feature/diff_feature_train.csv"),
        ]
    )

    X_train = X_train.tocsr()

    original_train = pd.read_csv("../../data/preprocessed_data/train.csv")
    y_train = original_train['is_duplicate']

    print("train data shape: ", X_train.shape)
    print("train label shape: ", y_train.shape)

    return X_train, y_train


def get_tfidf_test_data():
    print("loading tfidf...")
    X_test = sparse.hstack(
        [
            cPickle.load(open("../../data/feature/test_q1_tfidf.pkl", "rb")),
            cPickle.load(open("../../data/feature/test_q2_tfidf.pkl", "rb")),
            pd.read_csv("../../data/feature/ngram_feature_test.csv"),
            pd.read_csv("../../data/feature/diff_feature_test.csv"),
        ]
    )

    original_test = pd.read_csv("../../data/preprocessed_data/test.csv")
    test_id = original_test['test_id']

    print("test data shape: ", X_test.shape)
    print("test id shape: ", test_id.shape)

    return X_test, test_id


def get_train_data(feature_names=None, log_trans=False, stand_trans=False):
    global scaler

    print("reading train data features...")
    X_train = pd.concat(
        [
            pd.read_csv("../../data/feature/ngram_feature_train.csv"),
            pd.read_csv("../../data/feature/tfidf_feature_train.csv"),
            pd.read_csv("../../data/feature/diff_feature_train.csv"),
            pd.read_csv("../../data/feature/noun_feature_train.csv"),
            pd.read_csv("../../data/feature/share_idf_train.csv"),
            pd.read_csv("../../data/feature/train_lr.csv"),
            pd.read_csv("../../data/feature/qid_train.csv"),
            # pd.read_csv("../../data/feature/word2vec_feature_train.csv"),
        ], axis=1
    )

    X_word2vec = pd.read_csv("../../data/feature/word2vec_feature_train.csv")
    X_word2vec.fillna(0, inplace=True)
    X_word2vec.vec_wmd[X_word2vec.vec_wmd == np.inf] = 100
    X_word2vec.vec_norm_wmd[X_word2vec.vec_norm_wmd == np.inf] = 100
    X_word2vec.vec_cosine[X_word2vec.vec_cosine < 0.0001] = 0.0001
    X_train = pd.concat([X_train, X_word2vec], axis=1)

    if feature_names is not None:
        X_train = X_train[feature_names]

    if log_trans:
        print("log transform data...")
        X_train = X_train.apply(log, axis=0)
    elif stand_trans:
        print("standardize data...")
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)

    original_train = pd.read_csv("../../data/preprocessed_data/train.csv")
    y_train = original_train['is_duplicate']

    print("train data shape: ", X_train.shape)
    print("train label shape: ", y_train.shape)

    return X_train, y_train


def get_test_data(feature_names=None, log_trans=False, stand_trans=False):
    global scaler

    print("reading test data features...")
    X_test = pd.concat(
        [
            pd.read_csv("../../data/feature/ngram_feature_test.csv"),
            pd.read_csv("../../data/feature/tfidf_feature_test.csv"),
            pd.read_csv("../../data/feature/diff_feature_test.csv"),
            pd.read_csv("../../data/feature/noun_feature_test.csv"),
            pd.read_csv("../../data/feature/share_idf_test.csv"),
            pd.read_csv("../../data/feature/test_lr.csv"),
            pd.read_csv("../../data/feature/qid_test.csv"),
            # pd.read_csv("../../data/feature/word2vec_feature_test.csv"),
        ], axis=1
    )

    X_word2vec = pd.read_csv("../../data/feature/word2vec_feature_test.csv")
    X_word2vec.fillna(0, inplace=True)
    X_word2vec.vec_wmd[X_word2vec.vec_wmd == np.inf] = 100
    X_word2vec.vec_norm_wmd[X_word2vec.vec_norm_wmd == np.inf] = 100
    X_word2vec.vec_cosine[X_word2vec.vec_cosine < 0.0001] = 0.0001
    X_test = pd.concat([X_test, X_word2vec], axis=1)

    if feature_names is not None:
        X_test = X_test[feature_names]

    if log_trans:
        print("log transform data...")
        X_test = X_test.apply(log, axis=0)
    elif stand_trans:
        if scaler is None:
            print("error! Did not standardize in train data!")
        else:
            print("standardize data...")
            X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

    original_test = pd.read_csv("../../data/preprocessed_data/test.csv")
    test_id = original_test['test_id']

    print("test data shape: ", X_test.shape)
    print("test id shape: ", test_id.shape)

    return X_test, test_id


def resample_for_sparse(X, y):
    print("resample...")
    y = y.values

    pos_index = (y == 1).nonzero()[0]
    neg_index = (y == 0).nonzero()[0]

    pos_X = X[pos_index]
    neg_X = X[neg_index]
    pos_y = y[pos_index]
    neg_y = y[neg_index]

    len_pos, len_neg = pos_X.shape[0], neg_X.shape[0]
    new_neg_X, new_neg_y = neg_X, neg_y

    p = 0.165
    # scale = ((len(pos_X) * 1.0 / (len(pos_X) + len(neg_X))) / p) - 1
    scale = len_pos / len_neg * (1 - p) / p - 1
    while scale > 1:
        new_neg_X = sparse.vstack([new_neg_X, neg_X])
        new_neg_y = np.hstack([new_neg_y, neg_y])
        scale -= 1

    remain = int(scale * len_neg)
    new_neg_X = sparse.vstack([new_neg_X, neg_X[:remain]])
    new_neg_y = np.hstack([new_neg_y, neg_y[:remain]])

    X = sparse.vstack([pos_X, new_neg_X])
    y = np.hstack([pos_y, new_neg_y])

    if None:
        print("X shape after resample: ", X.shape)
        print("y shape after resample: ", y.shape)
        print("positive sample: ", pos_X.shape[0])
        print("negative sample: ", new_neg_X.shape[0])
        print("pos ratio: ", pos_X.shape[0] * 1.0 / X.shape[0])

    return X, y


def resample(X, y, index=None):
    print("resample...")

    pos_X = X[y == 1]
    neg_X = X[y == 0]
    pos_y = y[y == 1]
    neg_y = y[y == 0]
    if not index is None:
        pos_index = index[y == 1]
        neg_index = index[y == 0]

    len_pos, len_neg = pos_X.shape[0], neg_X.shape[0]
    new_neg_X, new_neg_y = neg_X, neg_y
    if not index is None:
        new_neg_index = neg_index

    p = 0.165
    # scale = ((len(pos_X) * 1.0 / (len(pos_X) + len(neg_X))) / p) - 1
    scale = len_pos / len_neg * (1 - p) / p - 1
    while scale > 1:
        new_neg_X = pd.concat([new_neg_X, neg_X])
        new_neg_y = pd.concat([new_neg_y, neg_y])
        if not index is None:
            new_neg_index = np.hstack([new_neg_index, neg_index])
        scale -= 1

    remain = int(scale * len_neg)
    new_neg_X = pd.concat([new_neg_X, neg_X[:remain]])
    new_neg_y = pd.concat([new_neg_y, neg_y[:remain]])
    if not index is None:
        new_neg_index = np.hstack([new_neg_index, neg_index[:remain]])

    X = pd.concat([pos_X, new_neg_X])
    y = pd.concat([pos_y, new_neg_y])
    if not index is None:
        index = np.hstack([pos_index, new_neg_index])

    if None:
        print("X shape after resample: ", X.shape)
        print("y shape after resample: ", y.shape)
        print("positive sample: ", pos_X.shape[0])
        print("negative sample: ", new_neg_X.shape[0])
        print("pos ratio: ", pos_X.shape[0] * 1.0 / X.shape[0])

    if not index is None:
        return X, y, index
    return X, y
