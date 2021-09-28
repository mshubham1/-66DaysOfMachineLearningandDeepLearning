#!/usr/bin/env python
# coding: utf-8

# # Diabetes Prediction

# The objective of the project is to classify weather a person is having diabetes or not.
# The datset consist of servral dependent and independent variable.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data=pd.read_csv("C:/Users/hp/Documents/datasets/diabetes.csv")
data.head()


# In[3]:


data.shape


# In[4]:


data.info()


# In[5]:


data.describe()


# In[6]:


data.isnull().sum()


# In[7]:


data=data.drop_duplicates()


# In[8]:


#checking for 0 values in features
print(data[data['BloodPressure']==0].shape[0])
print(data[data['Glucose']==0].shape[0])
print(data[data['SkinThickness']==0].shape[0])
print(data[data['Insulin']==0].shape[0])
print(data[data['BMI']==0].shape[0])


# In[9]:


# replacing 0 value with the mean of that column
data['BloodPressure']=data['BloodPressure'].replace(0,data['BloodPressure'].mean())
data['Glucose']=data['Glucose'].replace(0,data['Glucose'].mean())
data['SkinThickness']=data['SkinThickness'].replace(0,data['SkinThickness'].mean())
data['Insulin']=data['Insulin'].replace(0,data['Insulin'].mean())
data['BMI']=data['BMI'].replace(0,data['BMI'].mean())


# In[10]:


## countplot for checking balanace of data
import warnings #avoid warning flash
warnings.filterwarnings('ignore')
sns.countplot(data['Outcome'])


# In[11]:


# the distribution of data skewed or normal distibution
data.hist(bins=10,figsize=(10,10))
plt.show


# In[12]:


sns.pairplot(data=data,hue='Outcome')
plt.show()


# In[13]:


plt.figure(figsize=(20,15))
sns.heatmap(data.corr(),annot=True)


# In[14]:


for feature in data.columns:
    sns.boxplot(data[feature])
    plt.title(feature)
    plt.figure(figsize=(15,15))


# In[15]:


## handling outliers
y=data['Outcome']
X=data.drop('Outcome',axis=1)


# In[16]:


X.columns


# In[21]:


from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
scaler.fit(X)
SSX=scaler.transform(X)


# In[22]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(SSX,y,test_size=0.2,random_state=2)


# In[23]:


X_train.shape,y_train.shape


# In[24]:


X_test.shape,y_test.shape


# In[25]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver='liblinear',multi_class='ovr') #creating an object
lr.fit(X_train,y_train) #training the model


# In[26]:


lr_pred=lr.predict(X_test)


# In[27]:


lr.score(X_train,y_train)


# In[28]:


lr.score(X_test,y_test)


# In[29]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(X_train,y_train)


# In[30]:


knn_pred=knn.predict(X_test)


# In[31]:


knn.score(X_test,y_test) 


# In[32]:


knn.score(X_train,y_train)


# In[33]:


from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(X_train,y_train)


# In[34]:


nb.predict(X_test)


# In[35]:


nb.score(X_test,y_test) #test score


# In[36]:


nb.score(X_train,y_train) #training score


# In[37]:


from sklearn.svm import SVC
sv=SVC()
sv.fit(X_train,y_train)


# In[38]:


sv.predict(X_test)


# In[39]:


sv.score(X_test,y_test) #test score


# In[40]:


sv.score(X_train,y_train) #training score


# In[41]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)


# In[42]:


dt_pred=dt.predict(X_test)


# In[43]:


dt.score(X_test,y_test) #test score


# In[44]:


dt.score(X_train,y_train) #training score


# In[45]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(X_train,y_train)


# In[46]:


rf.predict(X_test)


# In[47]:


rf.score(X_test,y_test) #test score


# In[48]:


rf.score(X_test,y_test) #test score

