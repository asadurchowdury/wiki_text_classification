import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from extract_feature import preprocessing

traindf = pd.read_csv(r'assets/WikiLarge_Train.csv')

traindf = preprocessing(traindf, clean = True)
X_train = traindf.dropna().drop('label', axis = 1)[['sentence_len','freq_score','aoa_score','syllable_count','Flesch_Kincaid_binary','char_len','stopwords','dif_aoa_ratio','phonemes']]
y_train = traindf.dropna().label

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

X_train = ss.fit_transform(X_train)

testdf = pd.read_csv(r'assets/WikiLarge_Test.csv')

testdf = preprocessing(testdf, clean = True)
X_test = testdf.drop('label', axis = 1).dropna()[['sentence_len','freq_score','aoa_score','syllable_count','Flesch_Kincaid_binary','char_len','stopwords','dif_aoa_ratio','phonemes']]


knn_f = KNeighborsClassifier(n_neighbors = 19, metric = 'manhattan', weights = 'distance')
knn_f.fit(X_train, y_train)

y_pred = knn_f.predict(X_test)
prediction = pd.DataFrame(y_pred, columns=['label'])
prediction.rename({"Unnamed: 0":'id'},axis=1,inplace=True)
prediction.to_csv('KNN_prediction.csv')

