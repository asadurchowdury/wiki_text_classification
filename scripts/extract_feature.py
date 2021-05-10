# this script will extract different features for each observation
# potentially use mrjob/spark in future

import pandas as pd
import re
import numpy as np
import syllapy

testdf = pd.read_csv('assets/test.csv')
aoa = pd.read_csv(r'C:\Users\socce\Downloads\AoA_51715_words.csv', encoding= 'unicode_escape')

#Freq_pm: Freq of the Word in general English (larger -> more common)
dict_lookup = dict(zip(aoa["Word"], aoa["Freq_pm"]))

#AoA_Kup_lem: Estimated AoA based on Kuperman et al. study lemmatized words.
other_dict_lookup  = dict(zip(aoa["Word"], aoa["AoA_Kup_lem"]))

def syllable_counter(tokenized_list):
    return sum([syllapy.count(token) for token in tokenized_list])

def preprocessing(df):
    df['words'] = df['original_text'].apply(lambda x: re.findall(r"\w+", x))
    df['sentence_len'] = df.words.apply(lambda x: len(x))
    df['freq_score'] = df.words.apply(lambda x: np.mean([dict_lookup.get(i) if i in dict_lookup else 0 for i in x]))
    df['aoa_score'] = df.words.apply(lambda x: np.mean([other_dict_lookup.get(i) for i in x if i in other_dict_lookup]))
    df['syllable_count'] = df['words'].progress_apply(lambda x: syllable_counter(x))
    df['Flesch_Kincaid'] = (206.835 - (1.015 * df.sentence_len) - (84.6 * (df.syllable_count / df.sentence_len)))
    df['Flesch_Kincaid_binary'] = np.where(df['Flesch_Kincaid'] > df['Flesch_Kincaid'].mean(), 0, 1)
    return df

import nltk
<<<<<<< HEAD
import spacy

=======
from sklearn.linear_model import LogisticRegression as LR
>>>>>>> a91194304c250b91fd810e750588810ccc801fd9
# feature 2 : word count without stopwords and punctuations

