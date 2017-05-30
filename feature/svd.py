from sklearn.decomposition import TruncatedSVD
import cPickle
from scipy import sparse
import numpy as np


def create_svd_feature(Q1, Q2, name):
    svd = TruncatedSVD(n_components=100)
    X = sparse.vstack([Q1, Q2])
    svd.fit(X)
    X_svd_q1 = svd.transform(Q1)
    X_svd_q2 = svd.transform(Q2)
    X_svd = np.hstack([X_svd_q1, X_svd_q2])
    cPickle.dump(X_svd, open("../../data/feature/svd_tfidf_" + name + ".pkl", "wb"))


if __name__ == '__main__':
    print("Processing train data...")
    q1_tfidf = cPickle.load(open("../../data/feature/train_q1_tfidf.pkl", "rb"))
    q2_tfidf = cPickle.load(open("../../data/feature/train_q2_tfidf.pkl", "rb"))
    create_svd_feature(q1_tfidf, q2_tfidf, "train")
    print("processing test data...")
    q1_tfidf = cPickle.load(open("../../data/feature/test_q1_tfidf.pkl", "rb"))
    q2_tfidf = cPickle.load(open("../../data/feature/test_q2_tfidf.pkl", "rb"))
    create_svd_feature(q1_tfidf, q2_tfidf, "test")
