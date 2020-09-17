#!/usr/bin/env python
# coding: utf-8

#                                                ### Facebook Prophet Model 

# In[1]:


#import libraries
from fbprophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


#read the csv file
df = pd.read_csv(r'C:\Users\NQE00254\Desktop\Power BI Reports\Data Science Courses\Python\AirPassengers.csv', encoding = 'unicode_escape')


# In[3]:


#print the dataframe
print (df)


# In[4]:


#create a Prophet Model
model = Prophet()


# In[5]:


#Change the column names to ds and y
df.columns = ['ds','y']


# In[6]:


model = Prophet()


# In[8]:


#Now fit the model
model.fit(df)


# In[9]:


#Lets create the future dataframe
future = model.make_future_dataframe(periods = 30,freq = 'D')


# In[10]:


future.tail()


# In[11]:


#Predict the future analysis
forecast=model.predict(future)


# In[17]:


#To get the information of the dataframe
forecast


# In[18]:


#To get the information of the dataframe
forecast.info()


# In[13]:


#import
import matplotlib as plt


# In[14]:


#Select the columns
forecast[['ds','yhat_lower','yhat_upper','yhat']].tail()


# In[15]:


#plot the model
model.plot(forecast)


# In[16]:


#Plotting the Components
fig1 = model.plot_components(forecast)


# In[ ]:




