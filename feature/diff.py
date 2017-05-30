from __future__ import print_function
import difflib
from fuzzywuzzy import fuzz
import pandas as pd


def diff_ratio(str1, str2):
    seq = difflib.SequenceMatcher()
    seq.set_seqs(str1.lower(), str2.lower())
    return seq.ratio()


def create_diff_feature(X, name):
    print("generate diff feature...")
    X['diff_ratio'] = X.apply(lambda row: diff_ratio(row['question1'], row['question2']), axis=1)

    print("generate fuzz feature...")
    X['fuzz_qratio'] = X.apply(lambda row: fuzz.QRatio(row['question1'], row['question2']), axis=1)
    X['fuzz_wratio'] = X.apply(lambda row: fuzz.WRatio(row['question1'], row['question2']), axis=1)
    X['fuzz_partial_ratio'] = X.apply(lambda row: fuzz.partial_ratio(row['question1'], row['question2']), axis=1)
    X['fuzz_partial_token_set_ratio'] = X.apply(
        lambda row: fuzz.partial_token_set_ratio(row['question1'], row['question2']), axis=1)
    X['fuzz_partial_token_sort_ratio'] = X.apply(
        lambda row: fuzz.partial_token_sort_ratio(row['question1'], row['question2']), axis=1)
    X['fuzz_token_set_ratio'] = X.apply(lambda row: fuzz.token_set_ratio(row['question1'], row['question2']), axis=1)
    X['fuzz_token_sort_ratio'] = X.apply(lambda row: fuzz.token_sort_ratio(row['question1'], row['question2']), axis=1)

    print("done. writing diff feature to csv...")
    new_features = [col for col in X.columns if 'fuzz' in col] + ['diff_ratio']
    X[new_features].to_csv("../../data/feature/diff_feature_" + name + ".csv", index=False)
    print("Done.")


if __name__ == '__main__':
    print("processing train data...")
    X_train = pd.read_csv("../../data/preprocessed_data/train.csv")
    X_train.fillna("", inplace=True)
    create_diff_feature(X_train, "train")

    # print("processing test data...")
    # X_test = pd.read_csv("../../data/preprocessed_data/test.csv")
    # X_test.fillna("", inplace=True)
    # create_diff_feature(X_test, "test")
