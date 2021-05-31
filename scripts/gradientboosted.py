from numpy import testing
from sklearn.metrics import f1_score
from urllib.request import urlopen
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier


data = pd.read_csv('assets/train.csv.gz',compression='gzip')
print(data.columns)

# feature_list = ['sentence_len', 'freq_score', 'aoa_score', 'syllable_count',
#        'Flesch_Kincaid', 'Flesch_Kincaid_binary', 'dale_ratio', 'char_len', 
#        'avg_word_len', 'stopwords', 'non_stopwords', 'sim_aoa_ratio',       
#        'dif_aoa_ratio', 'phonemes', 'conc_score']
feature_list = ['aoa_score','char_len','syllable_count']

data_f = data.fillna(0)
X_train = data_f[feature_list]
X_train = X_train.astype('float')
y_train = data_f.label

clf = GradientBoostingClassifier()
clf.fit(X_train,y_train)



testdata = pd.read_csv('assets/test.csv.gz',compression='gzip')

testdata_f = testdata[feature_list]

testdata_f = testdata_f.fillna(0)
testdata_f = testdata_f.astype('float')
print(testdata_f.head())
feature_importance = list(zip(feature_list,  clf.feature_importances_))
feature_importance = sorted(feature_importance,key= lambda x: x[1],reverse=True)
print(feature_importance)
# print(testdata.columns)
predict = clf.predict(testdata_f)

prediction = pd.DataFrame(predict, columns=['label'])
prediction.rename({"index":'id'},axis=0,inplace=True)
prediction.to_csv('prediction/gradient_prediction_with_3_features.csv')