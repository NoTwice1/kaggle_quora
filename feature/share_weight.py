#coding: utf-8
# copy from yalin zhou
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import multiprocessing as mp
from collections import Counter
import pdb

filename_list = ['train','test'] ####
parallel = True

stops = set(stopwords.words("english"))

def get_weights_func(counts, eps=10000,min_count=2):
    if(counts<min_count):
        return 0.0
    else:
        return 1.0/(counts + eps)

def get_weigths():
    data = pd.read_csv('../../data/preprocessed_data/train.csv')
    train_qs = pd.Series(data['question1'].tolist() + data['question2'].tolist()).astype(str)
    words = " ".join(train_qs).split(" ")
    counts = Counter(words)
    weights = {w: get_weights_func(c) for w,c in counts.items()} ###
    result = pd.DataFrame(weights.items(), columns=['word','weight'])
    result.to_csv('../model/trained_models/word_weights.csv',index=False)

def read_weights():
    weights_df = pd.read_csv('../model/trained_models/word_weights.csv')
    weights = weights_df.set_index('word')['weight'].to_dict()
    return weights

def extract_feature_func(df):
    #pdb.set_trace()
    q1_words = set(str(df['question1']).lower().split())
    q2_words = set(str(df['question2']).lower().split())

    q1_words = q1_words-stops
    q2_words = q2_words-stops

    #share_words feature
    df['shared_words'] = float(len(q1_words&q2_words))/(len(q1_words)+len(q2_words)+1)

    #tf_idf feature
    shared_weights = [weights_dict.get(w,0) for w in (q1_words&q2_words)]
    total_weights = [weights_dict.get(w,0) for w in q1_words] + [weights_dict.get(w,0) for w in q2_words]
    sum_total_weights = np.sum(total_weights)
    if(sum_total_weights==0):
        df['share_idf'] = 0
    else:
        df['share_idf'] = np.sum(shared_weights)/sum_total_weights

    #continue

    return df['share_idf']

def extract_feature(df):
    res = df.apply(extract_feature_func,axis=1,raw=True)
    return res

def get_feature(filename):
    data = pd.read_csv('../../data/preprocessed_data/'+filename+'.csv')
    if parallel:
        p = mp.Pool(processes=16)
        pool_results = p.map(extract_feature,np.array_split(data,16))
        p.close()
        p.join()

        results = pd.concat(pool_results,axis=0)
    else:
        results = extract_feature(data)
    results.to_csv('../../data/feature/'+'share_idf_' + filename + '.csv', index=False, header=['share_idf'])

if __name__=='__main__':
    #get_weigths()
    weights_dict = read_weights() ###
    for f in filename_list:
        print f
        get_feature(f)