# coding:utf-8
from __future__ import print_function
from __future__ import division
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
# from tqdm import tqdm
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis

stop_words = stopwords.words('english')
print("loading googlenew word2vec...")
model = KeyedVectors.load_word2vec_format('../../data/GoogleNews-vectors-negative300.bin.gz', binary=True)
norm_model = KeyedVectors.load_word2vec_format('../../data/GoogleNews-vectors-negative300.bin.gz', binary=True)
norm_model.init_sims(replace=True)


def wmd(s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return model.wmdistance(s1, s2)


def norm_wmd(s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return norm_model.wmdistance(s1, s2)


def sent2vec(s):
    words = str(s).lower().decode('utf-8')
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())


def create_word2vec_feature(X, name):
    print("generating word2vec features...")
    X['vec_wmd'] = X.apply(lambda x: wmd(x['question1'], x['question2']), axis=1)
    X['vec_norm_wmd'] = X.apply(lambda x: norm_wmd(x['question1'], x['question2']), axis=1)

    question1_vectors = np.zeros((X.shape[0], 300))
    question2_vectors = np.zeros((X.shape[0], 300))

    for i, q in enumerate(X.question1.values):
        question1_vectors[i, :] = sent2vec(q)
    for i, q in enumerate(X.question2.values):
        question2_vectors[i, :] = sent2vec(q)

    X['vec_cosine'] = \
        [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),np.nan_to_num(question2_vectors))]
    X['vec_cityblock'] \
        = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),np.nan_to_num(question2_vectors))]
    X['vec_jaccard'] = \
        [jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),np.nan_to_num(question2_vectors))]
    X['vec_canberra'] = \
        [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),np.nan_to_num(question2_vectors))]
    X['vec_euclidean'] = \
        [euclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),np.nan_to_num(question2_vectors))]
    X['vec_minkowski'] = \
        [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors),np.nan_to_num(question2_vectors))]
    X['vec_braycurtis'] = \
        [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),np.nan_to_num(question2_vectors))]

    X['skew_q1vec'] = [skew(x) for x in np.nan_to_num(question1_vectors)]
    X['skew_q2vec'] = [skew(x) for x in np.nan_to_num(question2_vectors)]
    X['kur_q1vec'] = [kurtosis(x) for x in np.nan_to_num(question1_vectors)]
    X['kur_q2vec'] = [kurtosis(x) for x in np.nan_to_num(question2_vectors)]

    print("writing word2vec feature to csv...")
    new_features = [col for col in X.columns if 'vec' in col]
    X[new_features].to_csv("../../data/feature/word2vec_feature_" + name + ".csv", index=False)

    print("Done.")


if __name__ == "__main__":
    X_train = pd.read_csv("../../data/preprocessed_data/train.csv")
    X_test = pd.read_csv("../../data/preprocessed_data/test.csv")
    X_train.fillna("", inplace=True)
    X_test.fillna("", inplace=True)

    print("processing train data...")
    create_word2vec_feature(X_train, "train")
    print("processing test data...")
    create_word2vec_feature(X_test, "test")
