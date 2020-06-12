#!/usr/bin/env python
# coding: utf-8

# In[52]:


import numpy as np
import sklearn
import scipy as sp


# In[53]:


import pandas as pd
import matplotlib.pyplot as plt


# In[54]:


from sklearn.datasets import load_boston
breastcancer=load_boston()
print(breastcancer.DESCR)
print(breastcancer.keys())


# In[55]:


print(breastcancer.target)


# In[56]:


print(breastcancer.feature_names)


# In[57]:


x=pd.DataFrame(breastcancer.data,columns=breastcancer.feature_names)


# In[58]:


y=pd.DataFrame(breastcancer.target,columns=["GUDDA"])


# In[59]:


x.head()


# In[60]:


y.head()


# In[ ]:





# In[61]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[62]:


x_train.shape


# In[63]:


y_train.shape


# In[64]:


from sklearn.linear_model import LinearRegression
linear_model=LinearRegression(normalize=True).fit(x_train,y_train)


# In[65]:


linear_model.score(x_train,y_train)


# In[66]:


y_pred=linear_model.predict(x_test)


# In[67]:


print(y_pred)
y_pred1=np.array(y_pred)


# In[68]:


plt.scatter(y_pred,y_test)
plt.show()


# In[69]:


y_test.shape


# In[70]:


y_pred.shape


# In[71]:


y_test1=np.array(y_test)


# In[75]:


we=y_test1-y_pred1


# In[80]:


print(we)


# In[ ]:




