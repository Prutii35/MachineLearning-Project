import numpy as np
import csv
import re
import pandas as pd
import matplotlib.pyplot as plt
import glob
import statistics

from numpy import genfromtxt

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.svm import SVR

from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix


def normalize(trainX,testX,type=None): #functie pentru normalizarea datelor
    scaler = None
    if type == 'standard':
        scaler = preprocessing.StandardScaler()
    elif type =='min_max':
        scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    elif type =='l1' or type =='l2':
        scaler = preprocessing.Normalizer(norm = type)
    elif type  == 'l2_v2':
        trainX = trainX/np.expand_dims(np.sqrt(np.sum(trainX**2,axis = 1)),axis=1)
        testX = testX/np.expand_dims(np.sqrt(np.sum(testX ** 2, axis=1)), axis=1)
    if scaler is not None:
        scaler.fit(trainX)
        trainX = scaler.transform(trainX)
        testX = scaler.transform(testX)
    return trainX,testX


def split(arr, size): #functie care imparte un vector in size parti
    arrs = []
    while len(arr) > size:
        pice = arr[:size]
        arrs.append(pice)
        arr = arr[size:]
    arrs.append(arr)
    return arrs

def make_features(X_train): # functie care face un nou vector cu features (min max mean mediana),scopul acesteia este de a avea mai putine date
    X_features = []
    for elem in X_train:
        elem = split(elem,10)
        X = []
        for i in elem:
            X.append([max(i),min(i),statistics.mean(i),statistics.median(i)]) #adaug cele 4 feature-uri obtinute de pe cele 45 de date
        X_features.append(X)
    return X_features


# citim etichetele de antrenare
path2 = "data/train_labels.csv"
data = pd.read_csv(path2, index_col=None, header=0)
data = np.array(data).astype(float)
list_of_labels = [] #in aceasta lista tin id urile
list_of_classes = [] #in aceasta lista tin clasele

for i in data:
    list_of_labels.append(i[0])
    list_of_classes.append(i[1])

list_of_classes = np.array(list_of_classes)
list_of_labels = np.array(list_of_labels)

#citim datele din train
path = "data/train/train"
all_files = glob.glob(path + "/*.csv")

X_train = [] #vector in care tin datele de antrenare
for filename in all_files:
    data = pd.read_csv(filename, index_col=None,header = None)

    data = data.values.tolist()
    if len(data) < 150: #in caz ca sunt mai putin de 150 de date in csv, adaug 0
        while len(data) != 150:
            data.append([0.0,0.0,0.0])

    if len(data) > 150 : #in caz ca sunt mai multe, reajustez lungimea la 150
        data = data[0:150]

    data = np.array(data).astype(float)
    X_train.append(data)

X_train = np.array(X_train)
X_train = X_train.reshape(9000,450)

X_features = make_features(X_train) #calculez features pentru datele de train
X_features = np.array(X_features)
X_features = X_features.reshape(9000,180)#deoarece vectorul rezultat o sa fie de tipul (9000,45,4) acesta trebuia sa fie reformat
print(X_features.shape)

#citesc datele pentru test care este asemanatoare cu cea pentru datele de antrenare
path = "data/test/test"
all_files = glob.glob(path + "/*.csv")

Test_Data = []
Test_id = []
for filename in all_files:
    data = pd.read_csv(filename, index_col=None,header = None)

    temp = re.findall(r'\d+', filename)
    res = list(map(int, temp))
    Test_id.extend(res)

    data = data.values.tolist()
    if len(data) < 150:
        while len(data) != 150:
            data.append([0.0,0.0,0.0])

    if len(data) > 150 :
        data = data[0:150]

    data = np.array(data).astype(float)
    Test_Data.append(data)

Test_Data = np.array(Test_Data)
Test_Data = Test_Data.reshape(5000,450)

Test_features = make_features(Test_Data)#calculam features pentru datele de antrenare pe care o sa facem predictul
Test_features = np.array(Test_features)
Test_features = Test_features.reshape(5000,180) #deoarece vectorul rezultat o sa fie de tipul (5000,45,4) acesta trebuia sa fie reformat
print(Test_features.shape)

#X_train,Test_Data = normalize(X_train,Test_Data,'standard')
X_features,Test_features = normalize(X_features,Test_features,'standard') #standardizez datele

#X_train, X_test, y_train, y_test = train_test_split(X_train, list_of_classes, train_size=0.8,test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(X_features, list_of_classes, train_size=0.8,test_size=0.2) #impart datele in 80% pentru train si 20 pentru test


svm = SVC(gamma='scale',C = 5) #clasificatorul
svm.fit(X_train, y_train) #antrenam datele
print('Accuracy of SVM classifier on training set: {:.2f}'
     .format(svm.score(X_train, y_train)))#verificam scorul obtinut pe datele de train
print('Accuracy of SVM classifier on test set: {:.2f}'
     .format(svm.score(X_test, y_test)))#verificam scorul obtinut pe datele de test

'''
predicted_val = svm.predict(Test_features)
#predicted_val = svm.predict(Test_Data)


with open('data/predicted_values.csv', 'w',newline='') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(['id','class'])
    for i in range(len(predicted_val)):
        writer.writerow([int(Test_id[i]),int(predicted_val[i])])

csvFile.close()
'''
#cross validation + confusion matrix
y_pred = svm.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = svm, X = X_train, y = y_train, cv = 3)
print(accuracies.mean())
print(accuracies.std())

scores = []
cv = KFold(n_splits=3, random_state=42, shuffle=False)
best_svr = SVR(kernel='rbf')
for train_index, test_index in cv.split(X_features):
    print("Train Index: ", train_index, "\n")
    print("Test Index: ", test_index)

    X_train, X_test, y_train, y_test = X_features[train_index], X_features[test_index], list_of_labels[train_index], list_of_labels[test_index]
    best_svr.fit(X_train, y_train)
    scores.append(best_svr.score(X_test, y_test))
'''
    y_pred = best_svr.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
'''

#print(np.mean(scores))

