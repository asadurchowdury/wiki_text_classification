from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
# from functools import cache, lru_cache
from sklearn.pipeline import Pipeline, FeatureUnion
from scipy.sparse import hstack
# @ cache


data = pd.read_csv('assets/WikiLarge_Train.csv')

# other features can be added to this vectorizer, checkout sklearn for np.hstack

# vectorizer = TfidfVectorizer(ngram_range=(1,2))
# testing countvectorizer instead of tfidf

vectorizer = CountVectorizer(ngram_range=(1,2),max_df=1000)
print('Vectorizing the training data ...')
X_train = vectorizer.fit_transform(data.original_text)
y_train = data.label
print(X_train.shape)

# adding other features to vectorizer
# from sklearn.preprocessing import MaxAbsScaler, RobustScaler
# scaling = RobustScaler(with_centering=False).fit(X_train)
# X_train = scaling.transform(X_train)


print('Training Random forest model ...')
clf = RandomForestClassifier(n_estimators=100,n_jobs=-1,verbose=1)
clf.fit(X_train,y_train)

testdata = pd.read_csv('assets/WikiLarge_Test.csv')
print(testdata.head())

print('Fitting test data to vectorizer ...')
X_test = vectorizer.transform(testdata.original_text)
# X_test = scaling.transform(X_test)
y_test = testdata.label

y_pred = clf.predict(X_test)
prediction = pd.DataFrame(y_pred, columns=['label'])
prediction.rename({"Unnamed: 0":'id'},axis=1,inplace=True)
prediction.to_csv('prediction/prediction_count_vec.csv')


# from sklearn.metrics import f1_score, accuracy_score, auc

# score = f1_score(y_test,y_pred)

# print("F1 score for random forest model is ", score)




# if __name__ == '__main__':
#     main()
