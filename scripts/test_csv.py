#testfile
import pandas as pd
import numpy as np

data = pd.read_csv('prediction.csv')
data = data.rename({'Unnamed: 0':'id'},axis=1)
print(data.shape)
data.to_csv('prediction_final.csv',index = False)
print(data.head())