# create svm model
from numpy import testing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from zipfile import ZipFile
import io
from urllib.request import urlopen
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# print(ZipFile('assets/train_with_features.csv.gz'))

# file = ZipFile(io.BytesIO('assets/train_with_features.csv.zip'))
data = pd.read_csv('assets/train_with_features.csv.gz',compression='gzip')
print(data.columns)

data_f = data[['label', 'sentence_len','freq_score', 'aoa_score', 'syllable_count', 'Flesch_Kincaid', 'Flesch_Kincaid_binary','dale_ratio']]
data_f = data_f.dropna()
X_train = data_f[['sentence_len', 'freq_score', 'aoa_score','syllable_count', 'Flesch_Kincaid','dale_ratio']]
X_train = X_train.astype('float')
y_train = data_f['label']

print('training SVM model...')
clf = make_pipeline(StandardScaler(),SVC(kernel='rbf',gamma='scale',cache_size=7000))
# clf = SVC(kernel='rbf',gamma='scale')
clf.fit(X_train,y_train)


print('loading test data and making prediction...')
testdata = pd.read_csv('assets/test_aoa_updated.csv.gz',compression='gzip')
print(testdata.head())
# testdata = testdata.dropna()

print(len(testdata))
testdata_f = testdata[['sentence_len', 'freq_score','aoa_score', 'syllable_count', 'Flesch_Kincaid','dale_ratio']]
print(len(testdata_f))
testdata_f = testdata_f.fillna(0)
testdata_f = testdata_f.astype('float')
print(len(testdata_f))
# print(testdata.columns)
predict = clf.predict(testdata_f)

prediction = pd.DataFrame(predict, columns=['label'])
prediction.rename({"Unnamed: 0":'id'},axis=1,inplace=True)
prediction.to_csv('prediction/prediction_with_features_SVM.csv')
