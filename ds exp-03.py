#!/usr/bin/env python
# coding: utf-8

# In[158]:


import pandas as pd 


# In[159]:


import numpy as np


# In[160]:


import seaborn as sns


# In[161]:


df=pd.read_csv("C:\\Users\\bujji\\Desktop\\Downloads\\titanic_dataset.csv")


# In[162]:


df.info()


# In[163]:


df.head()


# In[164]:


df.isnull().sum()


# In[165]:


df.drop(columns=['Cabin'],inplace=True)


# In[166]:


df.info


# In[167]:


df.isnull().sum()


# In[168]:


df["Age"] = df["Age"].fillna(df["Age"].median())


# In[169]:


df.boxplot()


# In[170]:


df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])


# In[ ]:





# In[171]:


df["Embarked"].value_counts()


# In[172]:


df["Pclass"].value_counts()


# In[173]:


df["Survived"].value_counts()


# In[174]:


sns.countplot(x="Survived",data=df)


# In[175]:


sns.countplot(x="Pclass",data=df)


# In[176]:


sns.countplot(x="Sex",data=df)


# In[177]:


df.info()


# In[178]:


sns.displot(df["Fare"])


# In[179]:


sns.countplot(x="Pclass",hue="Survived",data=df)


# In[180]:


sns.countplot(x="Sex",hue="Survived",data=df)


# In[181]:


sns.displot(df[df["Survived"]==0]["Age"])


# In[182]:


pd.crosstab(df["Pclass"],df["Survived"])


# In[183]:


pd.crosstab(df["Sex"],df["Survived"])


# In[184]:


df.corr()


# In[185]:


sns.heatmap(df.corr(),annot=True)


# In[ ]:




