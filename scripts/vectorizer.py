from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
data = pd.read_csv('assets/test.csv')

# other features can be added to this vectorizer, checkout sklearn for np.hstack
vectorizer = TfidfVectorizer(min_df=100,ngram_range=(1,2),stop_words='english')

print('Vectorizing the training data ...')
X_train = vectorizer.fit_transform(data.original_text)
y_train = data.label

from sklearn.preprocessing import MaxAbsScaler, RobustScaler
scaling = RobustScaler(with_centering=False).fit(X_train)
X_train = scaling.transform(X_train)


print('Training SVM model ...')
clf = svm.SVC(kernel='linear')
clf.fit(X_train,y_train)

testdata = pd.read_csv('assets/WikiLarge_Train.csv')

print('Fitting test data to vectorizer ...')
X_test = vectorizer.transform(testdata.original_text)
X_test = scaling.transform(X_test)
y_test = testdata.label

y_pred = clf.predict(X_test)


from sklearn.metrics import f1_score, accuracy_score, auc

score = f1_score(y_test,y_pred)

print("F1 score for SVM is ", score)





