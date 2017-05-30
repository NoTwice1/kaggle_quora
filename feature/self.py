import pandas as pd
from collections import defaultdict
import numpy as np

print("reading train data...")
train_question = pd.read_csv("../../data/preprocessed_data/train.csv")[['question1','question2']]
train_predict = pd.read_csv("../../results/submission/xgb_train.csv")
train = pd.concat([train_question, train_predict], axis=1)

print("reading test data...")
test_question = pd.read_csv("../../data/preprocessed_data/test.csv")[['question1','question2']]
test_predict = pd.read_csv("../../results/submission/xgb_submission.csv")
test = pd.concat([test_question, test_predict], axis=1)

all = pd.concat([train, test], axis=0)

print("counting...")
hist = defaultdict(list)
for i in range(all.shape[0]):
    row = all.iloc[i,:]
    hist[row['question1']].append(row['is_duplicate'])
    hist[row['question2']].append(row['is_duplicate'])

print("calculating features...")
pred_mean = map(lambda x: np.mean(hist[x]), hist)
pred_mean = dict((k,v) for k,v in zip(hist.keys(), pred_mean))

pred_min = map(lambda x: min(hist[x]), hist)
pred_min = dict((k,v) for k,v in zip(hist.keys(), pred_min))

pred_max = map(lambda x: max(hist[x]), hist)
pred_max = dict((k,v) for k,v in zip(hist.keys(), pred_max))

pred_std = map(lambda x: np.std(hist[x]), hist)
pred_std = dict((k,v) for k,v in zip(hist.keys(), pred_std))

print("setting to X...")
q1_pred_mean = all.apply(lambda x: pred_mean[x['question1']], axis=1)
q2_pred_mean = all.apply(lambda x: pred_mean[x['question2']], axis=1)
q1_pred_min = all.apply(lambda x: pred_min[x['question1']], axis=1)
q2_pred_min = all.apply(lambda x: pred_min[x['question2']], axis=1)
q1_pred_max = all.apply(lambda x: pred_max[x['question1']], axis=1)
q2_pred_max = all.apply(lambda x: pred_max[x['question2']], axis=1)
q1_pred_std = all.apply(lambda x: pred_std[x['question1']], axis=1)
q2_pred_std = all.apply(lambda x: pred_std[x['question2']], axis=1)

print("writing...")
X = pd.DataFrame({'q1_pred_mean':q1_pred_mean,'q1_pred_min':q1_pred_min, 'q1_pred_max':q1_pred_max,'q1_pred_std':q1_pred_std,
                  'q2_pred_mean':q2_pred_mean,'q2_pred_min':q2_pred_min,'q2_pred_max':q2_pred_max,'q2_pred_std': q2_pred_std})
X_train = X.iloc[:train.shape[0]]
X_test = X.iloc[train.shape[0]:]

X_train.to_csv("../../data/feature/self_train.csv",index=False)
X_test.to_csv("../../data/feature/self_test.csv",index=False)



