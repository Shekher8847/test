import os
# os.chdir(C:\\Users\cd42146\PycharmProjects\untitled\venv\Lib\site-packages\plotly\package_data\datasets\iris.csv)
import pandas as pd
import numpy as np

dataset = pd.read_csv('C:\\Users\cd42146\Downloads\iris.csv')
# print(dataset.head())
# print(dataset.iloc[:,0:4])
# print(dataset.iloc[:,1:4].values)   ### we took the x input contents Values conver it in to an array

x = dataset.iloc[:,0:4].values
print(x)
y = dataset.iloc[:,4:5].values
# print(y)

from sklearn.preprocessing import LabelEncoder
labelencoder_y =LabelEncoder()
y =labelencoder_y.fit_transform(y)

# print(y)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

### using Logistics regresion(classification)
# from sklearn.linear_model import  LogisticRegression
# logmodel =LogisticRegression()
# print(logmodel.fit(x_train,y_train))
#
# y_pred =logmodel.predict(x_test)
#
# print(y_pred)
# print(y_test)
#
#
from sklearn.metrics import confusion_matrix
# print(confusion_matrix(y_test,y_pred))


#### Using K-nearest neighboru algorithme
### In confusion matrix Rows represents the actual and column represents predicted values . Where there is a digit and row and column cross each other
## It means the data is correct e.g. cross section of second row an column is 13 , it means the predicted and actual both reacords areright for a category (true posetives)
### Accurecy of the model will be (sum of all diagonals) /(sum of all the data)
### accuracy here will be = (4+13+12)/(4+13+1+12) = 29/30 = 0.96 x!00 = 96.6%


# from sklearn.neighbors import KNeighborsClassifier
# classifire_knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski',p=2)
#
# ## here neighbores = 5  means how many neighbour data will see to put a new data in class
# ## here p= 2 , misowaski has given two type of distance 1st id Eucladian Distance and Manhatton distance
# print(classifire_knn.fit(x_train,y_train))
#
# y_pred = classifire_knn.predict(x_test)
#
# # print(y_pred)
# # print(y_test)
#
# print(confusion_matrix(y_test,y_pred)

### accurecy is (12+9+8) /(12+9+ 1 +8) = 0.966 0r 96 percent

###Using Naive Bayes classification algorithm

# from sklearn.naive_bayes import GaussianNB
# classifire_NB =GaussianNB()
# classifire_NB.fit(x_train,y_train)
#
# y_pred = classifire_NB.predict(x_test)
#
# # print(y_pred)
# # print(y_test)
#
# print(confusion_matrix(y_test,y_pred))

### accurecy is (12+9+9) /(12+9+9) = 1 0r 100 percent

### Using Support Vector Machine using sigmoid kernal

# from sklearn.svm import SVC
#
# classifire_svm_sigmoid = svc(kernal ='sigmoid')
# classifire_svm_sigmoid.fit(x_train,y_train)
#
# y_pred = classifire_svm_sigmoid.predict(x_test)
#
# print(confusion_matrix(y_test,y_pred))


### Using svm by changing Liner kernal .
# from sklearn.svm import SVC
#
# classifire_svm_liner = SVC(kernal = 'linear')
#
# classifire_svm_sigmoid.fit(x_train,y_train)
#
# y_pred = classifire_svm_sigmoid.predict(x_test)
#
# print(confusion_matrix(y_test,y_pred))


### Using svm by changing Liner kernal .
# from sklearn.svm import SVC
#
# classifire_svm_liner = SVC(kernal = 'linear')
#
# classifire_svm_linear.fit(x_train,y_train)
#
# y_pred = classifire_svm_linear.predict(x_test)
#
# print(confusion_matrix(y_test,y_pred))

### Using svm by changing Radial Basis Funciton (RBF) kernal .
# from sklearn.svm import SVC
#
# classifire_svm_RBF = SVC(kernal = 'RBF')
#
# classifire_svm_RBF.fit(x_train,y_train)
#
# y_pred = classifire_svm_RBF.predict(x_test)
#
# print(confusion_matrix(y_test,y_pred))

### Using svm by changing Polynomial kernal .
# from sklearn.svm import svc
#
# classifire_svm_poly = SVC(kernal = 'poly')
#
# classifire_svm_poly.fit(x_train,y_train)
#
# y_pred = classifire_svm_RBF.predict(x_test)
#
# print(confusion_matrix(y_test,y_pred))


### Using Decision Tree clsssifire
# from sklearn.tree import DecisionTreeClassifier
#
# classifire_dt = DecisionTreeClassifier(criterion='entropy') # Entropy is the the measure of impurity what decided if impurty is low or high in data set
#
# classifire_dt.fit(x_train,y_train)
#
# y_pred = classifire_dt.predict(x_test)

# print(y_pred)
# print(y_test)

# print(confusion_matrix(y_test,y_pred))
# without using entropy -accurecy came around (11+6+11) /(11+6+1+1+11) = 0.93 or 93% accurecy , which is seems low

# with entropy is around 96 % with Decision Tree

#### Now we will use algorithem Random Forest
### Ensemble learning means , when a sub model runs under a big model , like Decision Tree works under Random Forest

# from sklearn.ensemble import RandomForestClassifier
# classifire_rf = RandomForestClassifier(n_estimators=3) # n_estimators = 3 means , how many time we wanterd to run the decision tree, criterion ="entropy"
# classifire_rf.fit(x_train,y_train)
#
# y_pred = classifire_rf.predict(x_test)
#
#
# print(y_pred)
# print(y_test)
# print(confusion_matrix(y_test, y_pred))

### Accurecy is around 96% acuurecy with Random Forest