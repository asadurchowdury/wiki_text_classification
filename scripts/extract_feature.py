# this script will extract different features for each observations
# potentially use mrjob/spark in future

# feature 1 : simple word count
import pandas as pd
import numpy as np
testdf = pd.read_csv('assets/test.csv')
def text_counter(sentence):
    count = len(sentence.split())
    return count

def word_count(df):
    df['word_count']=df['original_text'].apply(text_counter)
    return df

import nltk
import spacy

from sklearn.linear_model import LogisticRegression as LR
# feature 2 : word count without stopwords and punctuations

