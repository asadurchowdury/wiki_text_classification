from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
# data = pd.read_csv('assets/test.csv')
data = pd.read_csv('assets/train_with_features.csv')
data_f = data[['label', 'sentence_len', 'freq_score', 'aoa_score', 'syllable_count', 'Flesch_Kincaid', 'Flesch_Kincaid_binary']]
data_f = data_f.dropna()
X_train = data_f[['sentence_len', 'freq_score', 'syllable_count', 'Flesch_Kincaid', 'Flesch_Kincaid_binary']]
X_train = X_train.astype('float')
y_train = data_f['label']

clf = RandomForestClassifier(n_estimators=100,n_jobs=-1)
clf.fit(X_train,y_train)



testdata = pd.read_csv('assets/test_with_features.csv')

# testdata = testdata.dropna()
print(len(testdata))
testdata_f = testdata[['sentence_len', 'freq_score', 'syllable_count', 'Flesch_Kincaid', 'Flesch_Kincaid_binary']]
print(len(testdata_f))
testdata_f = testdata_f.fillna(0)
testdata_f = testdata_f.astype('float')
print(len(testdata_f))
# print(testdata.columns)
predict = clf.predict(testdata_f)

prediction = pd.DataFrame(predict, columns=['label'])
prediction.rename({"Unnamed: 0":'id'},axis=1,inplace=True)
prediction.to_csv('prediction_with_features.csv')
