import pandas as pd

question_to_id = {}

print("reading train questions...")
train_question = pd.read_csv("../../data/preprocessed_data/train.csv")[['qid1', 'qid2', 'question1', 'question2']]
max_id = 0
for i in range(train_question.shape[0]):
    row = train_question.iloc[i, :]
    if row['question1'] not in question_to_id:
        question_to_id[row['question1']] = row['qid1']
        max_id = max(max_id, row['qid1'])
    if row['question2'] not in question_to_id:
        question_to_id[row['question2']] = row['qid2']
        max_id = max(max_id, row['qid2'])

train_max_id = max_id

print("reading test questions...")
test_question = pd.read_csv("../../data/preprocessed_data/test.csv")[['question1', 'question2']]
for i in range(test_question.shape[0]):
    row = test_question.iloc[i, :]
    if row['question1'] not in question_to_id:
        question_to_id[row['question1']] = max_id + 1
        max_id += 1
    if row['question2'] not in question_to_id:
        question_to_id[row['question2']] = max_id + 1
        max_id += 1

for q in question_to_id:
    if question_to_id[q] > train_max_id:
        question_to_id[q] = (question_to_id[q] - train_max_id) / 6 + train_max_id

print("setting id to X...")
train_q1_id = train_question.apply(lambda x: question_to_id[x['question1']], axis=1)
train_q2_id = train_question.apply(lambda x: question_to_id[x['question2']], axis=1)
train_qmin = map(min, train_q1_id, train_q2_id)
train_qmax = map(max, train_q1_id, train_q2_id)

test_q1_id = test_question.apply(lambda x: question_to_id[x['question1']], axis=1)
test_q2_id = test_question.apply(lambda x: question_to_id[x['question2']], axis=1)
test_qmin = map(min, test_q1_id, test_q2_id)
test_qmax = map(max, test_q1_id, test_q2_id)

print("writing...")
X_train = pd.DataFrame({'q1_id': train_q1_id, 'q2_id': train_q2_id, 'qmin': train_qmin, 'qmax': train_qmax})
X_train.to_csv("../../data/feature/qid_train.csv", index=False)

X_test = pd.DataFrame({'q1_id': test_q1_id, 'q2_id': test_q2_id, 'qmin': test_qmin, 'qmax': test_qmax})
X_test.to_csv("../../data/feature/qid_test.csv", index=False)
