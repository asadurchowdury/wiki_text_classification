# call option python preprocessing.py "filename"
# return a pandas dataframe 
# print option will save the dataframe to file
import pandas as pd
import numpy as np

def preprocess(file, toprint=False):
    df = pd.read_csv(file)
    print(df.head())
    return df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='the original file')
    parser.add_argument('output', help='input the name of output file')
    args = parser.parse_args()

    dataframe = preprocess(args.input)
    dataframe.to_csv(args.output, index = False)



