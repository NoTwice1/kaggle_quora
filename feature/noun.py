from __future__ import print_function
from __future__ import division
import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords

from n_gram import jaccard, dice_dist


def get_nouns(question):
    words = nltk.word_tokenize(question)
    nouns = [w for w, t in nltk.pos_tag(words) if t[:1] == 'N']
    return nouns


def common_noun(q1, q2):
    ans = 0
    for w in q1:
        if w in q2:
            ans += 1
    return ans

def create_noun_feature(X, name):
    print('generating noun feature...')

    q1_nouns = map(get_nouns, X.question1)
    q2_nouns = map(get_nouns, X.question2)

    X['q1_noun_num'] = map(len, q1_nouns)
    X['q2_noun_num'] = map(len, q2_nouns)
    X['common_noun_num'] =map(common_noun, q1_nouns, q2_nouns)

    X['jaccard_noun'] = map(jaccard, q1_nouns, q2_nouns)
    X['dice_dist_noun'] = map(dice_dist, q1_nouns, q2_nouns)

    print("done. Write noun features to csv...")
    new_features = [col for col in X.columns if 'noun' in col]
    X[new_features].to_csv('../../data/feature/noun_feature_' + name + '.csv', index=False)

    print('Done.')


if __name__ == '__main__':
    print("processing train data...")
    X_train = pd.read_csv("../../data/preprocessed_data/train.csv")
    X_train.fillna("", inplace=True)
    create_noun_feature(X_train, 'train')

    print("processing test data...")
    X_test = pd.read_csv("../../data/preprocessed_data/test.csv")
    X_test.fillna("", inplace=True)
    create_noun_feature(X_test, 'test')
