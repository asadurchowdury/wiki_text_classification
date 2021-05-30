# this script will extract different features for each observation
# potentially use mrjob/spark in future

from numpy.lib.function_base import extract
import pandas as pd
import re
import numpy as np
import syllapy
import nltk
import spacy



# please update the link for aoa word list
aoa = pd.read_csv(r'C:\Users\Lannister\Desktop\wiki_text_classification\assets\AoA_51715_words.csv', encoding= 'unicode_escape')

#Freq_pm: Freq of the Word in general English (larger -> more common)
dict_lookup = dict(zip(aoa["Word"], aoa["Freq_pm"]))

#AoA_Kup_lem: Estimated AoA based on Kuperman et al. study lemmatized words.
other_dict_lookup  = dict(zip(aoa["Word"], aoa["AoA_Kup_lem"]))

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

def preprocessing(df):
    df['words'] = df['original_text'].apply(lambda x: re.findall(r"\w+", x))
    df['sentence_len'] = df.words.apply(lambda x: len(x))
    df['freq_score'] = df.words.apply(lambda x: np.mean([dict_lookup.get(i) if i in dict_lookup else 0 for i in x]))
    df['aoa_score'] = df.words.apply(lambda x: np.mean([other_dict_lookup.get(i) for i in x if i in other_dict_lookup]))
    df['syllable_count'] = df['words'].apply(lambda x: syllable_counter(x))
    df['Flesch_Kincaid'] = (206.835 - (1.015 * df.sentence_len) - (84.6 * (df.syllable_count / df.sentence_len)))
    df['Flesch_Kincaid_binary'] = np.where(df['Flesch_Kincaid'] > df['Flesch_Kincaid'].mean(), 0, 1)
    df['dale_ratio']=df['original_text'].apply(lambda x: ratio_calculator(x,dale))
    return df



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='combined data file (CSV)')
    parser.add_argument('output_file', help='cleaned data file (CSV)')
    args = parser.parse_args()
    df = pd.read_csv(args.input_file)
    extract_features = preprocessing(df)
    extract_features.to_csv(args.output_file, index=False,compression = 'gzip')

