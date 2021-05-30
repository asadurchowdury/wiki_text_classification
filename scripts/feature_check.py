from numpy import testing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from zipfile import ZipFile
import io
from urllib.request import urlopen
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


# print(ZipFile('assets/train_with_features.csv.gz'))

# file = ZipFile(io.BytesIO('assets/train_with_features.csv.zip'))
# data = pd.read_csv('./assets/train.csv.gz',compression='gzip')

data = pd.read_csv('./assets/AoA_51715_words.csv')
data = data.iloc[:,:10]
print(data.columns)
print(data.describe())
# from nltk.corpus import stopwords

# eng_stopwords = stopwords.words('english')
# print(eng_stopwords)

