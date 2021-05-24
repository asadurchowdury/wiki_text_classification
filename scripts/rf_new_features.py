from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

data = pd.read_csv('assets/train_with_features.csv')
data = data[['original_text', 'label', 'words', 'sentence_len',
       'freq_score', 'aoa_score', 'syllable_count', 'Flesch_Kincaid',
       'Flesch_Kincaid_binary']
X_train = data[['words', 'sentence_len',
       'freq_score', 'aoa_score', 'syllable_count', 'Flesch_Kincaid',
       'Flesch_Kincaid_binary']]

y_train = data['label']

clf = RandomForestClassifier(n_estimators=10, n_jobs=-1)

clf.fit(X_train,y_train)