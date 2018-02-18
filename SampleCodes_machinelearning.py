
 Regression Intro and Data, 'features' and Lables
import pandas as pd
import quandl
import math

df = quandl.get('WIKI/GOOGL')
print(df.head())
columns = list(df.columns.values)
df2 = df[columns[7:12]]
print(df2.head())
df2['HL_PCT'] = (df2['Adj. High'] - df2['Adj. Close'])/ df2['Adj. Close'] * 100.0
df2['PCT_change'] = (df2['Adj. Close'] - df2['Adj. Open'])/ df2['Adj. Open'] * 100.0
print(df2.head())

df3 = df2[['Adj. Close','HL_PCT','PCT_change','Adj. Volume',]]

forecast_col = 'Adj. Close'
df3.fillna(-99999, inplace=True)
#forecast_out = int(math.ceil(0.1*len(df3)))
forecast_out = int(math.ceil(0.01*len(df3)))
df3['label'] = df3[forecast_col].shift(-forecast_out)
df3.dropna(inplace=True)
print(df3.head())

 Regression Training and Testing
import pandas as pd
import quandl, math
import numpy as np  array
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

df = quandl.get('WIKI/GOOGL')
print(df.head())
columns = list(df.columns.values)
df2 = df[columns[7:12]]
#print(df2.head())
df2['HL_PCT'] = (df2['Adj. High'] - df2['Adj. Close'])/ df2['Adj. Close'] * 100.0
df2['PCT_change'] = (df2['Adj. Close'] - df2['Adj. Open'])/ df2['Adj. Open'] * 100.0
#print(df2.head())

df3 = df2[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

forecast_col = 'Adj. Close'
df3.fillna(-99999, inplace=True)
#forecast_out = int(math.ceil(0.1*len(df3)))
forecast_out = int(math.ceil(0.01*len(df3)))
df3['label'] = df3[forecast_col].shift(-forecast_out)
print(df3[forecast_col].shift(-forecast_out))
df3.dropna(inplace=True)

X = np.array(df3.drop(['label'],1))  Features
y = np.array(df3['label'])  label
print(len(X),len(y))
X = preprocessing.scale(X)  scale it along side all the data, pay attention to when adding new data
#x = x[:-forecast_out+1]
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)
clf = LinearRegression(n_jobs=-1)  algorithm 
clf.fit(X_train, y_train)  train
accuracy = clf.score(X_test,y_test)  test
print(accuracy)
clf = svm.SVR()  algorithm 
clf.fit(X_train, y_train)  train
accuracy = clf.score(X_test,y_test)  test
print(accuracy)
clf = svm.SVR(kernel='poly')  algorithm 
clf.fit(X_train, y_train)  train
accuracy = clf.score(X_test,y_test)  test
print(accuracy)

# Regression Forecasting and Predicting
import pandas as pd
import pickle
import quandl, math, datetime
import numpy as np  array
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
api_key = open('quandlapikey.txt','r').read()
df = quandl.get('WIKI/GOOGL',authtoken=api_key)
dset = pd.DataFrame(df)
dset.to_pickle('wikigoogle.pickle')
df = pd.read_pickle('wikigoogle.pickle')

print(df.head())
columns = list(df.columns.values)
df2 = df[columns[7:12]]
#print(df2.head())
df2['HL_PCT'] = (df2['Adj. High'] - df2['Adj. Close'])/ df2['Adj. Close'] * 100.0
df2['PCT_change'] = (df2['Adj. Close'] - df2['Adj. Open'])/ df2['Adj. Open'] * 100.0
#print(df2.head())

df3 = df2[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

forecast_col = 'Adj. Close'
df3.fillna(-99999, inplace=True)
#forecast_out = int(math.ceil(0.1*len(df3)))
forecast_out = int(math.ceil(0.01*len(df3)))
df3['label'] = df3[forecast_col].shift(-forecast_out)
print(df3[forecast_col].shift(-forecast_out))


X = np.array(df3.drop(['label'],1))  Features
X = preprocessing.scale(X)  scale it along side all the data, pay attention to when adding new data
X = X[:-forecast_out]
X_lately = X[-forecast_out:]
df3.dropna(inplace=True)
y = np.array(df3['label'])  label
print(len(X),len(y))
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)
clf = LinearRegression(n_jobs=-1)  algorithm 
clf.fit(X_train, y_train)  train
accuracy = clf.score(X_test,y_test)  test
print(accuracy)
forecast_set = clf.predict(X_lately)
#print(X_lately)
#print(forecast_set, accuracy, forecast_out)

df3['Forecast'] = np.nan # NaN
last_date = df3.iloc[-1].name; #print(last_date)
last_unix = last_date.timestamp(); #print(last_unix)
one_day = 86400
next_unix = last_unix + one_day
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df3.loc[next_date] = [np.nan for _ in range(len(df3.columns)-1)] + [i]
#print([np.nan for _ in range(len(df3.columns)-1)])
df3['Adj. Close'].plot()
df3['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend(loc=4)
plt.show()
#print(df3.tail(40))
clf = svm.SVR()  algorithm 
clf.fit(X_train, y_train)  train
accuracy = clf.score(X_test,y_test)  test
print(accuracy)
clf = svm.SVR(kernel='poly')  algorithm 
clf.fit(X_train, y_train)  train
accuracy = clf.score(X_test,y_test)  test
print(accuracy)


# Pickling the fit results
import pandas as pd
import pickle
import quandl, math, datetime
import numpy as np  array
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
api_key = open('quandlapikey.txt','r').read()
df = quandl.get('WIKI/GOOGL',authtoken=api_key)
dset = pd.DataFrame(df)
dset.to_pickle('wikigoogle.pickle')
df = pd.read_pickle('wikigoogle.pickle')

print(df.head())
columns = list(df.columns.values)
df2 = df[columns[7:12]]
#print(df2.head())
df2['HL_PCT'] = (df2['Adj. High'] - df2['Adj. Close'])/ df2['Adj. Close'] * 100.0
df2['PCT_change'] = (df2['Adj. Close'] - df2['Adj. Open'])/ df2['Adj. Open'] * 100.0
#print(df2.head())

df3 = df2[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

forecast_col = 'Adj. Close'
df3.fillna(-99999, inplace=True)
#forecast_out = int(math.ceil(0.1*len(df3)))
forecast_out = int(math.ceil(0.01*len(df3)))
df3['label'] = df3[forecast_col].shift(-forecast_out)
print(df3[forecast_col].shift(-forecast_out))


X = np.array(df3.drop(['label'],1))  Features
X = preprocessing.scale(X)  scale it along side all the data, pay attention to when adding new data
X = X[:-forecast_out]
X_lately = X[-forecast_out:]
df3.dropna(inplace=True)
y = np.array(df3['label'])  label
print(len(X),len(y))
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)
clf = LinearRegression(n_jobs=-1)  algorithm 
clf.fit(X_train, y_train)  train
with open('lm.pickle','wb') as f:
    pickle.dump(clf,f)
pickle_in = open('lm.pickle','rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test,y_test)  test
print(accuracy)
forecast_set = clf.predict(X_lately)
#print(X_lately)
#print(forecast_set, accuracy, forecast_out)

df3['Forecast'] = np.nan # NaN
last_date = df3.iloc[-1].name; #print(last_date)
last_unix = last_date.timestamp(); #print(last_unix)
one_day = 86400
next_unix = last_unix + one_day
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df3.loc[next_date] = [np.nan for _ in range(len(df3.columns)-1)] + [i]
#print([np.nan for _ in range(len(df3.columns)-1)])
df3['Adj. Close'].plot()
df3['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend(loc=4)
plt.show()

#  Regression how it works

# Classification K Nearest Neighbors

# Classification Support Vector Machin (SVM)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style as style
style.use('ggplot')
from sklearn import preprocessing, cross_validation, neighbors

df = pd.read_csv('breast-cancer-wisconsin.data.csv', sep=',',
                  names = ['id',
                     'ClumpThickness',
                     'UniformityCellSize',
                     'UniformityCellShape',
                     'MarginalAdhesion',
                     'SingleEpithelialCellSize',
                     'BareNuclei',
                     'BlandChromatin',
                     'NormalNucleoli',
                     'Mitoses',
                     'Class'])

#print(df.head())
#print(df.count(axis=0))
df.replace("?", -99999, inplace=True)
 exclude the 'id' variable
df.set_index('id', inplace=True)
X = np.array(df.drop(['Class'],1))
y = np.array(df['Class'])
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
#clf = pd.to_pickle(clf,'knmodel1.pickle')
accuracy = clf.score(X_test,y_test)
print("The accuracy is ",accuracy)

 include the 'id' variable
# df.reset_index(level=0, drop=False, inplace=True, col_level=0, col_fill='')
# X = np.array(df.drop(['Class'],1))
# y = np.array(df['Class'])
# X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)
# clf = neighbors.KNeighborsClassifier()
# clf.fit(X_train, y_train)
# #clf = pd.to_pickle(clf,'knmodel2.pickle')
# accuracy = clf.score(X_test,y_test)
# print("The accuracy is ",accuracy)

 example to predict
# example_measures = np.array([[4,2,1,1,1,2,3,2,1],
#                              [1,10,2,1,4,2,10,2,4]])
# # print(example_measures.shape)
# # example_measures.reshape(len(example_measures),-1)
# # print(example_measures)
# prediction = clf.predict(example_measures)
# print(prediction)

#  Euclidean Distance and k_nearest_neighbors function
# from math import sqrt
# plot1 = [1,3]
# plot2 = [2,5]
# def euclidean_distance(point1,point2):
#     euc_distance = sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)
#     return euc_distance
# print(euclidean_distance(plot1,plot2))

# import warnings
# from collections import Counter
#
# dataset ={'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
# new_features = [5,7]
# # for i  in dataset:
# #     for ii in dataset[i]:
# #         plt.scatter(ii[0], ii[1], s=100, color=i)
# # [[plt.scatter(ii[0],ii[1], s= 100, color=i) for ii in dataset[i]] for i in dataset]
# # plt.scatter(new_features[0],new_features[1], s=100, color='b')
# # plt.show()
#
# def k_nearest_neighbors(data,predict,k=3):
#     if len(data) >= k:
#         warnings.warn('K is set to a value less than total voting groups!')
#     #knnalgos
#     distances = []
#     for group in data:
#         for features in data[group]:
#             euc_distance = np.sqrt(np.sum((np.array(features)-np.array(predict))**2))
#             #euc_distance = np.linalg.norm(np.array(features) - np.array(predict))
#             distances.append([euc_distance,group])
#     votes = [i[1] for i in sorted(distances)[:k]]
#     #print(votes)
#     print(Counter(votes).most_common(1))
#     vote_result = Counter(votes).most_common(1)[0][0]
#     return vote_result
# result = k_nearest_neighbors(dataset, new_features)
# # [[plt.scatter(ii[0],ii[1], s= 100, color=i) for ii in dataset[i]] for i in dataset]
# # plt.scatter(new_features[0],new_features[1], s=100, color=result)
# # plt.show()
#
#  testing the algorithm
# import random as rd
#
# full_data = df.astype(float).values.tolist()
# # print(full_data[:5])
# rd.shuffle(full_data)
# # print(20*'#')
# # print(full_data[:5])
#
# test_size =0.2
# train_set = {2:[],4:[]}
# test_set = {2:[],4:[]}
# train_data = full_data[:-int(test_size*len(full_data))]
# test_data = full_data[-int(test_size*len(full_data)):]
#
# for i in train_data:
#     train_set[i[-1]].append(i[:-1])
# for i in test_data:
#     test_set[i[-1]].append(i[:-1])
# correct = 0
# total = 0
# for group in test_set:
#     for data in test_set[group]:
#         vote = k_nearest_neighbors(train_set, data, k=5)
#         if group == vote:
#             correct += 1
#         total += 1
# print("Accuracy: ", correct/total *100, "%")


# Support VECTOR machine
# from sklearn import svm
# clf = svm.SVC()
# clf.fit(X_train, y_train)
# print("The accuracy of svm is ", clf.score(X_test,y_test))
#  Vector Basics


 Unsupervised Machine Learning
# Flat K-means clustering
# from sklearn.cluster import KMeans
# # x = [1, 5, 1.5, 8, 1, 9]
# # y = [2, 8, 1.8, 8, 0.6, 11]
# # plt.scatter(x,y,color='r')
# # plt.show()
# X = np.array([[1,2],[5,8],[1.5,1.8],[8,8],[1, 0.6],[9,11]])
# kmeans = KMeans(n_clusters=2)
# kmeans.fit(X)
# centroids = kmeans.cluster_centers_
# labels = kmeans.labels_
# print(centroids)
# print(labels)
# # z = np.zeros((len(X), len(X[0])+1))
# # z[:,:-1] = X
# # #z[:,-1:] = labels
# # print(z)
# colors = ["g.", "r.", "c.", "y."]
# for i in range(len(X)):
#     print("coordinate:", X[i], "label:", labels[i])
#     plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)
#
# plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=150, linewidths=5, zorder=10)
#
# plt.show()

# Hierarchical K-means clustering
# import numpy as np
# from sklearn.cluster import MeanShift
# from sklearn.datasets.samples_generator import make_blobs
# import matplotlib.pyplot as plt
# from matplotlib import style
#
# style.use("ggplot")
#
# centers = [[1, 1], [5, 5], [3, 10]]
# X, y = make_blobs(n_samples=500, centers=centers, cluster_std=1)
# # plt.scatter(X[:, 0], X[:, 1])
# # plt.show()
# ms = MeanShift()
# ms.fit(X)
# labels = ms.labels_
# cluster_centers_ = ms.cluster_centers_ # print(cluster_centers_)
# n_clusters_ = len(np.unique(labels))
# print("Number of estimated clusters:", n_clusters_)
# colors = 10*['r.','g.','b.','c.','k.','y.','m.']
# # print(colors)
# # print(labels)
# for i in range(len(X)):
#     plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)
#
# plt.scatter(cluster_centers_[:, 0], cluster_centers_[:, 1],
#             marker="x", color='k', s=150, linewidths=5, zorder=10)
#
# plt.show()

 Handling Non-numerical Data

#https://pythonprogramming.net/static/downloads/machine-learning-data/titanic.xls
from matplotlib import pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing, cross_validation
import pandas as pd

'''
Pclass Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
survival Survival (0 = No; 1 = Yes)
name Name
sex Sex
age Age
sibsp Number of Siblings/Spouses Aboard
parch Number of Parents/Children Aboard
ticket Ticket Number
fare Passenger Fare (British pound)
cabin Cabin
embarked Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
boat Lifeboat
body Body Identification Number
home.dest Home/Destination
'''

df = pd.read_excel('titanic.xls')
#print(df.head())
df.drop(['body','name'], 1, inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)
#print(df.head())

def handle_non_numerical_data(df):
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))

    return df

df = handle_non_numerical_data(df)
print(df.head())
df.drop(['ticket','sex'], 1, inplace=True)
#df.drop(['boat'], 1, inplace=True)
print(df.head())
X = np.array(df.drop(['survived'],1).astype(float))
X = preprocessing.scale(X)
y = np.array(np.array(df['survived']))

clf = KMeans(n_clusters=2)
clf.fit(X)
centroids = clf.cluster_centers_
labels = clf.labels_
correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1
print(correct/len(X))
# colors = ["g.", "r.", "c.", "y."]
# for i in range(len(X)):
#     print("coordinate:", X[i], "label:", labels[i])
#     plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)
#
# plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=150, linewidths=5, zorder=10)
#
# plt.show()


# Neural Networks
import tensorflow as tf
