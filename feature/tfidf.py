from __future__ import print_function
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import cPickle
from scipy import sparse


def cos(vector1, vector2):
    return cosine_similarity(vector1, vector2)[0][0]


def initial_tfidf_bow(X_train, X_test):
    print('initializing tfidf, bag of words object...')

    tfidf = TfidfVectorizer(max_features=None, ngram_range=(1, 3), min_df=3,max_df=0.99)
    bow = CountVectorizer(max_df=0.999, min_df=60, max_features=30000, ngram_range=(1, 7),
                          binary=True, analyzer='char')

    docs = X_train.question1.tolist() + X_train.question2.tolist() + \
           X_test.question1.tolist() + X_test.question2.tolist()

    print("fitting tfidf...")
    tfidf.fit(docs)
    cPickle.dump(tfidf, open("../model/trained_models/tfidf_object.pkl", "wb"))

    print("fitting bow...")
    bow.fit(docs)
    cPickle.dump(bow, open("../model/trained_models/bow_object.pkl", "wb"))

    print("done.")

    return tfidf, bow


def read_tfidf_bow():
    print("loading tfidf and bow object...")
    tfidf = cPickle.load(open("../model/trained_models/tfidf_object.pkl", "rb"))
    bow = cPickle.load(open("../model/trained_models/bow_object.pkl", "rb"))

    return tfidf, bow


def create_bow_feature(X, bow, name):
    print('generating bag of words features...')
    q1_bow = bow.transform(X['question1'])
    q2_bow = bow.transform(X['question2'])
    if name == "train":
        cPickle.dump(q1_bow, open("../../data/feature/train_q1_bow.pkl", "wb"))
        cPickle.dump(q2_bow, open("../../data/feature/train_q2_bow.pkl", "wb"))
    elif name == "test":  #run out of memory, split to 10 parts
        rows = q1_bow.shape[0]
        parts = 10
        part_len = int(rows / parts)
        for i in range(parts):
            begin = part_len * i
            if i < parts - 1:
                end = part_len * (i + 1)
            else:
                end = rows
            cPickle.dump(sparse.hstack([q1_bow[begin:end],q2_bow[begin:end]]),
                         open("../../data/feature/test_bow_" + str(i+1) + ".pkl", "wb"))

    # X['bow_consine'] = map(cos, q1_bow, q2_bow)
    # X['q1_bow_sum'] = map(np.sum, q1_bow)
    # X['q1_bow_mean'] = map(np.mean, q1_bow)
    # X['q2_bow_sum'] = map(np.sum, q2_bow)
    # X['q2_bow_mean'] = map(np.mean, q2_bow)

    print('done.')


def create_tfidf_feature(X, tfidf, name):
    print("generating tfidf features...")
    q1_tfidf = tfidf.transform(X['question1'])
    q2_tfidf = tfidf.transform(X['question2'])
    cPickle.dump(q1_tfidf, open("../../data/feature/" + name + "_q1_tfidf.pkl", "wb"))
    cPickle.dump(q2_tfidf, open("../../data/feature/" + name + "_q2_tfidf.pkl", "wb"))

    # X['tfidf_consine'] = map(cos, q1_tfidf, q2_tfidf)
    # X['q1_tfidf_sum'] = map(np.sum, q1_tfidf)
    # X['q1_tfidf_mean'] = map(np.mean, q1_tfidf)
    # X['q2_tfidf_sum'] = map(np.sum, q2_tfidf)
    # X['q2_tfidf_mean'] = map(np.mean, q2_tfidf)

    print("done.")


def create_tfidf_bow_feature(X, tfidf, bow, name):
    create_tfidf_feature(X, tfidf, name)
    create_bow_feature(X, bow, name)

    print('write bag of words and tfidf features to csv...')

    new_features = [col for col in X.columns if 'bow' in col or 'tfidf' in col]
    X[new_features].to_csv("../../data/feature/tfidf_feature_" + name + ".csv", index=False)

    print("Done.")


if __name__ == '__main__':
    X_train = pd.read_csv("../../data/preprocessed_data/train.csv")
    X_test = pd.read_csv("../../data/preprocessed_data/test.csv")
    X_train.fillna("", inplace=True)
    X_test.fillna("", inplace=True)

    tfidf, bow = initial_tfidf_bow(X_train, X_test)
    # tfidf, bow = read_tfidf_bow()

    print("processing train data...")
    create_tfidf_bow_feature(X_train, tfidf, bow, "train")
    print("processing test data...")
    create_tfidf_bow_feature(X_test, tfidf, bow, "test")
