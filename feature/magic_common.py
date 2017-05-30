'''
https://www.kaggle.com/c/quora-question-pairs/discussion/33287
'''

import pandas as pd
from collections import defaultdict

question_neighbor = defaultdict(set)

print("reading train questions...")
train_question = pd.read_csv("../../data/preprocessed_data/train.csv")[['question1', 'question2']]
for i in range(train_question.shape[0]):
    row = train_question.iloc[i,:]
    question_neighbor[row['question1']].add(row['question2'])
    question_neighbor[row['question2']].add(row['question1'])

print("reading test questions...")
test_question = pd.read_csv("../../data/preprocessed_data/test.csv")[['question1', 'question2']]
for i in range(test_question.shape[0]):
    row = test_question.iloc[i,:]
    question_neighbor[row['question1']].add(row['question2'])
    question_neighbor[row['question2']].add(row['question1'])

print("count intersection number...")
all_question = pd.concat([train_question, test_question], axis=0)
all_question['intersection_number'] = \
    all_question.apply(
        lambda x: len(question_neighbor[x['question1']].intersection(question_neighbor[x['question2']]))
                ,axis=1)

print("writing features...")
X_train = all_question['intersection_number'].iloc[:train_question.shape[0]]
X_test = all_question['intersection_number'].iloc[train_question.shape[0]:]

X_train.to_csv("../../data/feature/magic_common_train.csv", index=False, header=['magic_intersection'])
X_test.to_csv("../../data/feature/magic_common_test.csv", index=False, header=['magic_intersection'])
