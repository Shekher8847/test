
# coding: utf-8

# In[9]:

get_ipython().system(' pip install scikit-learn')


# In[8]:

get_ipython().system('python -m pip install --upgrade pip')


# In[2]:

# File Path for dataset --->  C:\\Users\cd42146\PycharmProjects\untitled\venv\Lib\site-packages\plotly\package_data\datasets\iris.csv


# In[18]:

import os 
os.chdir('C:\\Users\\cd42146\\Downloads\\')


# In[19]:

os.getcwd()


# In[12]:

import pandas as pd 
import numpy as np



# In[17]:

# impor the data 
dataset = pd.read_csv("C:\\Users\cd42146\Downloads\iris.csv")
dataset


# In[37]:

# creating x variable as input 
x= dataset.iloc[:,0:4].values 
print(x)


# In[36]:

# creating y variable as expecting outputs
y= dataset.iloc[:,-1].values 
print(y)


# In[40]:

#encode the data as per classes as computer can not read string classes of the data 
from sklearn.preprocessing import LabelEncoder
labelencoder_y =LabelEncoder()
y = labelencoder_y.fit_transform(y)
print(y) 


# In[42]:

# as the all the data is in the same scale , so we need not to do the feature scaling
# now we will have to data in train and test split 
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.2)


# In[44]:

# now we have to run model 
# we will start with Logistic regression

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(x_train,y_train)


# In[46]:

y_pred = logmodel.predict(x_test)
y_pred


# In[47]:

y_test


# In[52]:

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)


# In[57]:

#accurecy = Diagonal values (10+8+9)/total records(10+8+3+0)
27/30 *100 


# In[63]:

# K-neighbor classifire
from sklearn.neighbors import KNeighborsClassifier

classifire_Knn = KNeighborsClassifier(n_neighbors =5, metric='minkowski', p=2)
classifire_Knn.fit(x_train,y_train)


# In[64]:

# prediction for KNN 
y_pred = classifire_Knn.predict(x_test)


# In[65]:

y_pred


# In[68]:

#confusion matrix 
from sklearn.naive_bayes import GaussianNB
classifier_nb = GaussianNB()
classifier_nb.fit(x_train,y_train)


# In[70]:

y_pred = classifier_nb.predict(x_test)


# In[73]:

confusion_matrix(y_test,y_pred)


# In[80]:

# Support Vector Machine :
from sklearn.svm import SVC
classifire_svm_sigmoid = SVC(kernel='sigmoid')
classifire_svm_sigmoid.fit(x_train, y_train)


# In[81]:

y_pred = classifire_svm_sigmoid.predict(x_test)


# In[83]:

confusion_matrix(y_test,y_pred)


# In[84]:

from sklearn.svm import SVC
classifire_svm_linear = SVC(kernel='linear')
classifire_svm_linear.fit(x_train, y_train)


# In[85]:

y_pred = classifire_svm_linear.predict(x_test)


# In[86]:

confusion_matrix(y_test,y_pred)


# In[90]:

from sklearn.svm import SVC
classifire_svm_rbf = SVC(kernel='rbf')
classifire_svm_rbf.fit(x_train, y_train)


# In[91]:

y_pred = classifire_svm_rbf.predict(x_test)


# In[92]:

confusion_matrix(y_test,y_pred)


# In[93]:

from sklearn.svm import SVC
classifire_svm_poly = SVC(kernel='poly')
classifire_svm_poly.fit(x_train, y_train)


# In[94]:

y_pred = classifire_svm_poly.predict(x_test)


# In[95]:

confusion_matrix(y_test,y_pred)


# In[98]:

from sklearn.tree import DecisionTreeClassifier
classifire_dt = DecisionTreeClassifier(criterion ='entropy')
classifire_dt.fit(x_train, y_train)


# In[101]:

y_pred =classifire_dt.predict(x_test)


# In[102]:

confusion_matrix(y_test,y_pred)


# In[ ]:

import os
import ipyparallel as ipp

cluster = ipp.Cluster(n=4)
with cluster as rc:
    ar = rc[:].apply_async(os.getpid)
    pid_map = ar.get_dict()


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



