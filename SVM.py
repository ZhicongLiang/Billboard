import pandas as pd
import numpy as np
from sklearn import svm

feature_selected = ['Danceability', 
                 'Energy', 
                 'Speechiness', 
                 'Acousticness', 
                 'Instrumentalness', 
                 'Liveness',
                 'Valence',
                 'Loudness',
                 'Tempo',
                 'Artist_Score']


train_set = pd.read_excel('./train_set/train.xlsx')
Xtrain = np.array(train_set[feature_selected])
Ytrain = np.array(train_set['label'], dtype=float)
test_set = pd.read_excel('./test_set/test.xlsx')
Xtest = np.array(test_set[feature_selected])
Ytest = np.array(test_set['label'], dtype=float)

clf = svm.SVC(kernel='linear')

clf.fit(Xtrain, Ytrain)

train_predict = clf.predict(Xtrain)
train_accuracy = (train_predict==Ytrain).mean()
print("Train accuracy:", train_accuracy)

test_predict = clf.predict(Xtest)
test_accuracy = (test_predict==Ytest).mean()
print("Test accuracy:", test_accuracy)

