#!/usr/bin/env python
# coding: utf-8

# # RFR

# In[19]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# In[4]:


df= pd.read_csv('RF.csv')
df.head()
df.count()


# In[87]:


x= df[['speed']]
y= df['Time']


# In[88]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.3, random_state=42)
h = RandomForestRegressor(n_estimators=100, random_state=42)


# In[89]:


print(x_train.shape)
print(df.shape)


# In[90]:


print(y_train.shape)
print(df.shape)


# In[91]:


h.fit(x_train,y_train)
y_pred=h.predict(x_test)
mean_squared_error(y_test, y_pred)


# In[92]:


import numpy as np


# In[108]:


new_data = np.array([[10]])


# In[109]:


waiting_time_prediction = h.predict(new_data)


# In[110]:


print("Predicted Waiting Time:", waiting_time_prediction[0])


# In[96]:


y_pred


# In[97]:


y_pred=pd.DataFrame(y_pred, columns=['ypredict'])


# In[98]:


y_pred


# In[99]:


y_test


# In[104]:


plt.figure(figsize= (15,4))
plt.scatter(y_test, y_pred, color= 'red', label = 'comparison of prediction')

plt.grid()
plt.title('Waiting Time Prediction Model using RF Regression (vehicle speed parameter)')
plt.xlabel('Predicted Waiting Time')
plt.ylabel('Actual Waiting Time')
plt.show()


# In[101]:


waiting_time = h.predict(speed_value_2d)


# In[102]:


model_ranks=pd.Series(h.feature_importances_,index=x_train.columns, name='Importance').sort_index


# In[103]:


feat_importances = pd.Series(h.feature_importances_, index=x_train.columns )
feat_importances.nlargest(15).plot(kind='barh')
plt.xlabel('Importance Rank')
plt.title('Waiting Time Prediction Model using RF Regression')


# In[ ]:




