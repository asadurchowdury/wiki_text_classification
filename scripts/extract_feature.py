# this script will extract different features for each observation
# potentially use mrjob/spark in future



from numpy.lib.function_base import extract

import pandas as pd
import re
import numpy as np
import syllapy




aoa = pd.read_csv('assets/AoA_51715_words.csv', encoding= 'unicode_escape')

#Freq_pm: Freq of the Word in general English (larger -> more common)
dict_lookup = dict(zip(aoa["Word"], aoa["Freq_pm"]))

#AoA_Kup_lem: Estimated AoA based on Kuperman et al. study lemmatized words.
other_dict_lookup  = dict(zip(aoa["Word"], aoa["AoA_Kup_lem"]))

import nltk

import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk_data_path = "assets/nltk_data"
if nltk_data_path not in nltk.data.path:
    nltk.data.path.append(nltk_data_path)
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
eng_stopwords = stopwords.words('english')


# please update the link for aoa word list
aoa = pd.read_csv(r'assets/AoA_51715_words.csv', encoding= 'unicode_escape')
concrete = pd.read_csv(r'assets/Concreteness_ratings_Brysbaert_et_al_BRM.txt',sep='\t')

#Freq_pm: Freq of the Word in general English (larger -> more common)
dict_lookup = dict(zip(aoa["Word"], aoa["Freq_pm"]))

#AoA_Kup_lem: Estimated AoA based on Kuperman et al. study lemmatized words.
other_dict_lookup  = dict(zip(aoa["Word"], aoa["AoA_Kup_lem"]))

phoneme_dict = dict(zip(aoa["Word"], aoa["Nphon"]))

concrete_dict = dict(zip(concrete["Word"], concrete["Conc.M"]))



# Dale chall ratio
f = open('assets/dale_chall.txt','r')
dale = f.readlines()
dale = [x.replace('\n','') for x in dale]

def intersection(lst1, lst2):
    '''return intersection between two lists'''
    return list(set(lst1) & set(lst2))

def ratio_calculator(str,lst):
    '''given a string and list of words, it calculates how many words of that string are in the list'''
    strlst = str.split()
    itersect = intersection(strlst,lst)
    if len(itersect)==0:
        return 0
    else:
        return (len(itersect)/len(strlst))



def syllable_counter(tokenized_list):
    return sum([syllapy.count(token) for token in tokenized_list])


def avg_word(sent):
    words = sent.split()
    return (sum(len(word) for word in words)/len(words))

def lemma_list(lst):
    final = [lemmatizer.lemmatize(word) for word in lst]
    return final


def sim_ratio(l):
    sim = [x for x in l if x < 11]
    dif = [x for x in l if x > 11]
    if (len(sim) + len(dif)) == 0:
        return np.nan
    else:
        return len(sim) / (len(sim) + len(dif))


def dif_ratio(l):
    sim = [x for x in l if x < 11]
    dif = [x for x in l if x > 11]
    if (len(sim) + len(dif)) == 0:
        return np.nan
    else:
        return len(dif) / (len(sim) + len(dif))

def preprocessing(df,clean=True):
    df['words'] = df['original_text'].apply(lambda x: re.findall(r"\w+", x))
    if clean:
        df['words'] = df.words.apply(lambda x: [word for word in x if len(word)>1])
        df['words'] = df.words.apply(lambda x: [word for word in x if word not in ['LRB','RRB']])

    df['sentence_len'] = df.words.apply(lambda x: len(x))
    df['lem_words'] = df.words.apply(lambda x: lemma_list(x))
    df['freq_score'] = df.lem_words.apply(lambda x: np.mean([dict_lookup.get(i) if i in dict_lookup else 0 for i in x]))
    df['ya_score'] = df.words.apply(lambda x: [other_dict_lookup.get(i) for i in x if i in other_dict_lookup])
    df['aoa_score'] = df.lem_words.apply(lambda x: np.mean([other_dict_lookup.get(i) for i in x if i in other_dict_lookup]))
    df['syllable_count'] = df['words'].apply(lambda x: syllable_counter(x))
    df['Flesch_Kincaid'] = (206.835 - (1.015 * df.sentence_len) - (84.6 * (df.syllable_count / df.sentence_len)))
    df['Flesch_Kincaid_binary'] = np.where(df['Flesch_Kincaid'] > df['Flesch_Kincaid'].mean(), 0, 1)
    df['dale_ratio']=df['original_text'].apply(lambda x: ratio_calculator(x,dale))
    df['char_len'] = df['original_text'].str.len()
    df['avg_word_len'] = df['original_text'].apply(lambda x: avg_word(x))
    df['stopwords'] = df['words'].apply(lambda x: len([x for x in x if x in eng_stopwords]))
    df['non_stopwords'] = df['sentence_len']-df['stopwords']

    df['sim_aoa_ratio'] = df.ya_score.apply(lambda x: sim_ratio(x))  # higher is better
    df['dif_aoa_ratio'] = df.ya_score.apply(lambda x: dif_ratio(x))  # lower is better
    df['phonemes'] = df.words.apply(lambda x: np.mean([phoneme_dict.get(i) for i in x if i in phoneme_dict]))
    df['conc_score'] = df.words.apply(lambda x: np.mean([concrete_dict.get(i) for i in x if i in concrete_dict]))
    df.drop(['original_text','lem_words','words', 'ya_score'],axis=1,inplace = True)

    df.drop(['original_text','lem_words','words'],axis=1,inplace = True)


    return df



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='combined data file (CSV)')
    parser.add_argument('output_file', help='cleaned data file (CSV)')
    args = parser.parse_args()
    df = pd.read_csv(args.input_file)
    extract_features = preprocessing(df,True)
    extract_features.to_csv(args.output_file, index=False,compression = 'gzip')
    
    