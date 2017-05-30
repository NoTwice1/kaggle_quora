import re
import pandas as pd
import numpy as np
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from string import punctuation

stop_words = ['the', 'a', 'an', 'and', 'but', 'if', 'or', 'as', 'this', 'that',
              'these', 'those', 'then', 'just', 'so', 'such', 'about', 'for',
              'is', 'of', 'to']


def replace_text(question, no_stopwords=False, no_punctuation=True, stem=False):
    question = question.lower()

    puncs = dict.fromkeys(punctuation, 1)
    if no_punctuation:
        question = ''.join(c for c in question if c not in puncs)

    if no_stopwords:
        question = word_tokenize(question)
        question = [w for w in question if not w in stop_words]
        question = " ".join(question)

    if stem:
        question = question.split(" ")
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in question]
        question = " ".join(stemmed_words)

    return question


def clear_data(X):
    X.question1.fillna("", inplace=True)
    X.question2.fillna("", inplace=True)

    X.question1 = X.question1.apply(lambda q: re.sub(r'[^\x00-\x7f]', r'', str(q)))
    X.question2 = X.question2.apply(lambda q: re.sub(r'[^\x00-\x7f]', r'', str(q)))

    X.question1 = X.question1.apply(replace_text)
    X.question2 = X.question2.apply(replace_text)

    return X


if __name__ == '__main__':
    print("processing train data...")
    X_train = pd.read_csv("../data/input/train.csv")
    processed_X_train = clear_data(X_train)
    processed_X_train.to_csv("../data/preprocessed_data/train.csv", index=False)

    print("processing test data...")
    X_test = pd.read_csv("../data/input/test.csv")
    processed_X_test = clear_data(X_test)
    processed_X_test.to_csv("../data/preprocessed_data/test.csv", index=False)

    print("Done.")
