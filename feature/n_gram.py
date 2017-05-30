from __future__ import print_function
from __future__ import division
import pandas as pd
import numpy as np
from nltk.corpus import stopwords

stops = set(stopwords.words('english'))

def get_n_gram(text, n):
    text = text.split(" ")
    lens = len(text)
    grams = []
    for i in range(lens - n + 1):
        words = []
        for j in range(i, i + n):
            words.append(text[j])
        grams.append('_'.join(words))
    return grams


def word_match(q1_gram1, q2_gram1):
    q1_words = {}
    q2_words = {}
    for word in q1_gram1:
        if word not in stops:
            q1_words[word] = 1
    for word in q2_gram1:
        if word not in stops:
            q2_words[word] = 1
    if len(q1_words) == 0 or len(q2_words) == 0:
        return 0
    shared_words = [w for w in q1_words.keys() if w in q2_words]
    R = len(shared_words) * 2.0 / (len(q1_words) + len(q2_words))
    return R

def generate_grams(X):
    print('generate unigram, bigram')

    grams = {}
    grams['q1_gram1'] = X.apply(lambda row: get_n_gram(row['question1'], 1), axis=1)
    grams['q1_gram2'] = X.apply(lambda row: get_n_gram(row['question1'], 2), axis=1)
    grams['q2_gram1'] = X.apply(lambda row: get_n_gram(row['question2'], 1), axis=1)
    grams['q2_gram2'] = X.apply(lambda row: get_n_gram(row['question2'], 2), axis=1)

    return grams


def jaccard(gram1, gram2):
    g1, g2 = set(gram1), set(gram2)
    return try_divide(len(g1.intersection(g2)), len(g1.union(g2)))


def dice_dist(gram1, gram2):
    g1, g2 = set(gram1), set(gram2)
    return 2 * try_divide(len(g1.intersection(g2)), len(g1) + len(g2))


def common_gram_num(row1, row2):
    num = 0
    for g1 in row1:
        if g1 in row2:
            num += 1.
    return num


def try_divide(x, y):
    return x * 1.0 / y if y else 0


def get_position_list(row1, row2):
    pos = []
    for i, g1 in enumerate(row1):
        if g1 in row2:
            pos.append(i)
    if not pos:
        pos = [0]
    lens = len(row1)
    pos = map(try_divide, pos, [lens] * len(pos))
    return pos


def create_gram_feature(X, name):
    grams = generate_grams(X)

    print('generate count feature...')
    X['q1_len'] = map(len, X.question1)
    X['q2_len'] = map(len, X.question2)
    X['q1_q2_len_diff'] = X['q1_len'] - X['q2_len']
    X['q1_char_len'] = X.question1.apply(lambda x: len(''.join(set(x.replace(' ','')))))
    X['q2_char_len'] = X.question2.apply(lambda x: len(''.join(set(x.replace(' ', '')))))

    X['q1_gram1_cnt'] = map(len, grams['q1_gram1'])
    X['q2_gram1_cnt'] = map(len, grams['q2_gram1'])

    print('generate intersect count feature...')
    X['q1_q2_gram1_cnt'] = map(common_gram_num, grams['q1_gram1'], grams['q2_gram1'])
    X['q1_q2_gram2_cnt'] = map(common_gram_num, grams['q1_gram2'], grams['q2_gram2'])

    X['q1_q2_gram1_ratio_q1'] = map(try_divide, X['q1_q2_gram1_cnt'], X['q1_gram1_cnt'])
    X['q1_q2_gram1_ratio_q2'] = map(try_divide, X['q1_q2_gram1_cnt'], X['q2_gram1_cnt'])
    X['q1_q2_gram2_ratio_q1'] = map(try_divide, X['q1_q2_gram2_cnt'], X['q1_gram1_cnt'] - 1)
    X['q1_q2_gram2_ratio_q2'] = map(try_divide, X['q1_q2_gram2_cnt'], X['q2_gram1_cnt'] - 1)

    print("generating word match feature...")
    X['word_match'] = map(word_match, grams['q1_gram1'], grams['q2_gram1'])

    print("generate distance feature...")
    for g in ['gram1', 'gram2']:
        g1, g2 = 'q1_' + g, 'q2_' + g

        target = g + "_jaccard"
        X[target] = map(jaccard, grams[g1], grams[g2])

        target = g + "_dice_dist"
        X[target] = map(dice_dist, grams[g1], grams[g2])


    # print('generate intersect position features...')
    # for g in ['gram1', 'gram2']:
    #     for q1, q2 in zip(['q1', 'q2'], ['q2', 'q1']):
    #         g1, g2 = q1 + '_' + g, q2 + '_' + g
    #         pos = map(get_position_list, grams[g1], grams[g2])
    #
    #         target = 'q1_q2_' + g + '_pos_' + q1
    #         X[target + '_min'] = map(min, pos)
    #         X[target + '_max'] = map(max, pos)
    #         X[target + '_mean'] = map(np.mean, pos)
    #         X[target + '_median'] = map(np.median, pos)
    #         X[target + '_std'] = map(np.std, pos)

    print("done. Write n_gram features to csv...")
    new_features = [col for col in X.columns if 'gram' in col or 'len' in col] + ['word_match']
    X[new_features].to_csv('../../data/feature/ngram_feature_' + name + '.csv', index=False)

    print('Done.')


if __name__ == '__main__':
    print("processing train data...")
    X_train = pd.read_csv("../../data/preprocessed_data/train.csv")
    X_train.fillna("", inplace=True)
    create_gram_feature(X_train, "train")

    # print("preprocessing test data...")
    # X_test = pd.read_csv("../../data/preprocessed_data/test.csv")
    # X_test.fillna("", inplace=True)
    # create_gram_feature(X_test, "test")
