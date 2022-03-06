#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import datetime


# In[71]:


def parser(x):
    return datetime.strptime('190'+x, '%Y-%m')
df=pd.read_csv("https://raw.githubusercontent.com/blue-yonder/pydse/master/pydse/data/sales-of-shampoo-over-a-three-ye.csv",sep=";")


# In[72]:


df


# In[73]:


series=df.set_index('Month',inplace=True)


# In[74]:


series = df.dropna()


# In[75]:


series.plot()
plt.show()


# In[76]:


X = series.values


# In[77]:


X


# In[78]:


size = int(len(X) * 0.60)
print(len(X))
print(size)


# In[79]:


train, test = X[0:size], X[size:len(X)]


# In[80]:


history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history,order=(5,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print(f'Predicted={yhat},Expected ={obs}')
error = mean_squared_error(test,predictions)
print(f"TEST MSE :{error}")


# In[81]:


plt.plot(test)
plt.plot(predictions, color='red')
plt.show()


# In[ ]:




