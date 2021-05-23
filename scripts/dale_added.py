from os import replace
import pandas as pd
import numpy as np
# from nltk.stem import Porterstemmer

# data = pd.read_csv('assets/train_with_features.csv.gz',compression='gzip')
test_data = pd.read_csv('assets/test_aoa_updated.csv.gz',compression='gzip',index_col='id')
print(test_data.head())

f = open('assets/dale_chall.txt','r')
dale = f.readlines()
dale = [x.replace('\n','') for x in dale]
# print(dale)
def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

def ratio_calculator(str,lst):
    strlst = str.split()
    itersect = intersection(strlst,lst)
    if len(itersect)==0:
        return 0
    else:
        return (len(itersect)/len(strlst))

# data['dale_ratio'] = data['original_text'].apply(lambda x: ratio_calculator(x,dale))
test_data['dale_ratio']=test_data['original_text'].apply(lambda x: ratio_calculator(x,dale))

# data.to_csv('assets/train_with_features.csv.gz',index=False,compression='gzip')
test_data.to_csv('assets/test_aoa_updated.csv.gz',compression='gzip')
# print(data.head(20))