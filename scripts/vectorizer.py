from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
import pandas as pd
import numpy as np
data = pd.read_csv('assets/test.csv')

# other features can be added to this vectorizer, checkout sklearn for np.hstack
vectorizer = TfidfVectorizer(min_df=100,ngram_range=(1,2),stop_words='english')

print('Vectorizing the train data ...')
X_train = vectorizer.fit_transform(data.original_text)
y_train = data.label

print('Training SVM model ...')
clf = svm.SVC(kernel='rbf')
clf.fit(X_train,y_train)

testdata = pd.read_csv('assets/WikiLarge_Train.csv')

X_test = vectorizer.transform(testdata.original_text)
y_test = testdata.label

y_pred = clf.predict(X_test)

from sklearn.metrics import f1_score, accuracy_score, auc

score = f1_score(y_test,y_pred)

print("F1 score for SVM is ", score)





