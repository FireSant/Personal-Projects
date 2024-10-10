from sklearn.datasets import load_iris
iris_ds=load_iris()

#returns the iris object. "It's a bunch objetct", wich is like a dictionary. contains keys and values
#print (("keys of iris dataset: \n{}".format(iris_ds.keys())))
    
#returns the data at the key DESCR. Looks like description
#print(iris_ds ['DESCR'] [:193])

#returns the data at target names key. doubt to the {} you must use .format and (). 
#print("target names:{}".format (iris_ds ['target_names'] ))

#
#print("feature names:{}".format (iris_ds ['feature_names'] ))

#return the type of data contained in the key
#print("data type: {}".format(type(iris_ds["data"])))

#returns the shape of the matrix in the dict. In this case you have 150 rows and 4 columns
#print("Shape of data: {}".format (iris_ds['data'].shape))

#return the amount you specify. \n start a new line
#print('First five rows\n{}'.format(iris_ds['data'][:5]))

#shows the target data shape
#print("Target shape {}".format (iris_ds['target'].shape))

#show target data
#print("targat matrix \n{}".format(iris_ds['target']))

from sklearn.model_selection import train_test_split
import pandas as pd
#
X_train, X_test, y_train, y_test= train_test_split(
iris_ds['data'], iris_ds['target'], random_state=0)

#print(X_train.shape)
#print(y_train.shape)

#print(X_test.shape)
#print(y_test.shape)

from pandas.plotting import scatter_matrix
import mglearn
import matplotlib.pyplot as plt
import numpy as np
 # create dataframe from data in X_train
 # label the columns using the strings in iris_dataset.feature_names
iris_df = pd.DataFrame(X_train, columns=iris_ds.feature_names)
 # create a scatter matrix from the dataframe, color by y_train
grr = scatter_matrix(iris_df, c=y_train, figsize=(15, 15), marker='o',
hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)
# WAS FORGETING TO INSTALL MGLEARN, PYPLOT from MATPLOTLIB AND SCATTER MATRIX FROM PANDAS.PLOTTING


#KNeighbors algorithm for regression and classification tasks. It works by finding the nearest k points to a new uknown data point
#For classification, the majority label among neighbors, or near points, is assignew to a new point
#For regression, the average value between near points is used as prediction
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
X_new = np.array([[5, 2.9, 1, 0.2]])

#prediction = knn.predict(X_new)
#print("Prediction: {}".format(prediction))
#print("Predicted target name: {}".format(iris_ds['target_names'][prediction]))
y_pred = knn.predict(X_test)
#print("Test set predictions:\n {}".format(y_pred))
#print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))

print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))