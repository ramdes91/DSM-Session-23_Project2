
# coding: utf-8

# In[1]:


import sqlite3
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit, RandomizedSearchCV
from sklearn.pipeline import make_pipeline

import pickle


# In[2]:


# Create your connection.
cnx = sqlite3.connect('database.sqlite')
df = pd.read_sql_query("SELECT * FROM Player_Attributes", cnx)


# In[3]:


dd = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", cnx)


# In[4]:


print(dd)


# In[5]:


df.head()


# In[6]:


target = df.pop('overall_rating')


# In[7]:


df.shape


# In[8]:


target.head()


# In[9]:


#input target function
target.isnull().values.sum()#missing values in target function


# In[10]:


target.describe()


# In[11]:


plt.hist(target, 30, range=(33, 94))


# In[12]:


y = target.fillna(target.mean())


# In[13]:


y.isnull().values.any()


# In[14]:


#data exploration

df.columns


# In[15]:


for col in df.columns:
    unique_cat = len(df[col].unique())
    print("{col}--> {unique_cat}..{typ}".format(col=col, unique_cat=unique_cat, typ=df[col].dtype))


# In[16]:


dummy_df = pd.get_dummies(df, columns=['preferred_foot', 'attacking_work_rate', 'defensive_work_rate'])
dummy_df.head()


# In[17]:


X = dummy_df.drop(['id', 'date'], axis=1)


# In[19]:


#Using Linear Regression

pipe = make_pipeline(StandardScaler(),             #preprocessing(standard scalling)
                     LinearRegression())           #estimator(linear regression)

cv = ShuffleSplit(random_state=0)   #defining type of cross_validation(shuffle spliting)

param_grid = {'linearregression__n_jobs': [-1]}     #parameters for model tunning

grid = GridSearchCV(pipe, param_grid=param_grid, cv=cv)


# In[21]:


grid.fit(X_train, y_train)          #training


# In[22]:


grid.best_params_


# In[23]:


lin_reg = pickle.dumps(grid)


# In[24]:


#Using decision tree

pipe = make_pipeline(StandardScaler(),                  #preprocessing
                     DecisionTreeRegressor(criterion='mse', random_state=0))          #estimator

cv = ShuffleSplit(n_splits=10, random_state=42)        #cross validation

param_grid = {'decisiontreeregressor__max_depth': [3, 5, 7, 9, 13]}

grid = GridSearchCV(pipe, param_grid=param_grid, cv=cv)


# In[25]:


grid.fit(X_train, y_train)          #training


# In[26]:


grid.best_params_


# In[27]:


Dectree_reg = pickle.dumps(grid)


# In[28]:


lin_reg = pickle.loads(lin_reg)
Dectree_reg = pickle.loads(Dectree_reg)


# In[30]:


print("""Linear Regressor accuracy is {lin}
DecisionTree Regressor accuracy is {Dec}""".format(lin=lin_reg.score(X_test, y_test),
                                                       Dec=Dectree_reg.score(X_test, y_test)))


# By accuracy comparision performed above we can say that Decision Tree regressor gives better result than linear regression model and it can predict the target function with approx 93% accuracy.

# In[ ]:




