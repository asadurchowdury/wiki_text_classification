# create svm model
# potentially add gridsearch in future

from sklearn import svm

clf = svm.SVC(kernel='rbf')