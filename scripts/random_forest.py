from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
<<<<<<< HEAD
from functools import cache, lru_cache

# @ cache

data = pd.read_csv('assets/test.csv')
data = pd.read_csv('assets/train_with_features.csv')
=======
data = pd.read_csv('assets/test.csv')
>>>>>>> d5868e35d85adb1119c6b687750c6acca526f392

# other features can be added to this vectorizer, checkout sklearn for np.hstack
vectorizer = TfidfVectorizer(min_df=100,ngram_range=(1,2),stop_words='english')

print('Vectorizing the training data ...')
X_train = vectorizer.fit_transform(data.original_text)
y_train = data.label
print(X_train.shape)
# from sklearn.preprocessing import MaxAbsScaler, RobustScaler
# scaling = RobustScaler(with_centering=False).fit(X_train)
# X_train = scaling.transform(X_train)


print('Training Random forest model ...')
clf = RandomForestClassifier(n_estimators=10,n_jobs=-1)
clf.fit(X_train,y_train)

testdata = pd.read_csv('assets/WikiLarge_Test.csv')

print('Fitting test data to vectorizer ...')
X_test = vectorizer.transform(testdata.original_text)
# X_test = scaling.transform(X_test)
y_test = testdata.label

y_pred = clf.predict(X_test)
<<<<<<< HEAD
prediction = pd.DataFrame(y_pred, columns=['label'])
prediction.rename({"Unnamed: 0":'id'},axis=1,inplace=True)
prediction.to_csv('prediction_random.csv')

=======
prediction = pd.DataFrame(y_pred, columns=['label']).to_csv('prediction.csv')
>>>>>>> d5868e35d85adb1119c6b687750c6acca526f392

from sklearn.metrics import f1_score, accuracy_score, auc

score = f1_score(y_test,y_pred)

print("F1 score for random forest model is ", score)




<<<<<<< HEAD
if __name__ == '__main__':
    main()
=======

>>>>>>> d5868e35d85adb1119c6b687750c6acca526f392
