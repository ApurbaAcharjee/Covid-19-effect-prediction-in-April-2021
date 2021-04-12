#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
bd=pd.read_excel('bgd-covid19-subnational.xlsx')
bd.head()


# In[2]:


bd.shape


# In[3]:


import seaborn as sns
from matplotlib import pyplot as plt


# In[4]:


#Total cases per day
sns.set(rc={'figure.figsize':(15,10)})
sns.lineplot(x="Updated Date",y="Total Cases in District",data=bd)
plt.show()


# In[5]:


sns.set(rc={'figure.figsize':(15,10)})
sns.barplot(x="Division",y="Total Cases in District",data=bd,hue="Updated Date")
plt.show()


# In[6]:


#Predicting latest data
last_day_cases=bd[bd["Updated Date"]=="12.15.2020"]
last_day_cases


# In[7]:


#Maximum Cases per day latest data
max_cases_country=last_day_cases.sort_values(by="Total Cases in District",ascending=False)
max_cases_country


# In[8]:


# Highlighting Data frame
bd.style.background_gradient(cmap='Reds')


# In[9]:


dhaka_case=bd[bd["Division"]=="Dhaka"]


# In[12]:


import plotly
import plotly.express as px
fig = px.bar(dhaka_case, x="District/City",y="Total Cases in District", barmode='group',height=400)
fig.update_layout(title_text='Trend of Coronavirus Cases in Dhaka Division on daily basis',plot_bgcolor='rgb(230,230,230)')
fig.show()


# In[14]:


dhaka_case=bd[bd["Division"]=="Dhaka"]
ctg_case=bd[bd["Division"]=="Chattogram"]
Rj_case=bd[bd["Division"]=="Rajshahi"]
Kh_case=bd[bd["Division"]=="Khulna"]

import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(
rows=2, cols=2,
specs=[[{}, {}],
      [{"colspan":2}, None]],
subplot_titles=("Dhaka","Chattogram","Rajshahi","Khulna"))

fig.add_trace(go.Bar(x=dhaka_case['District/City'],y=dhaka_case['Total Cases in District'],marker=dict(color=dhaka_case['Total Cases in District'],coloraxis="coloraxis")),1,1)

fig.add_trace(go.Bar(x=ctg_case['District/City'],y=ctg_case['Total Cases in District'],marker=dict(color=ctg_case['Total Cases in District'],coloraxis="coloraxis")),1,2)

fig.add_trace(go.Bar(x=Rj_case['District/City'],y=Rj_case['Total Cases in District'],marker=dict(color=Rj_case['Total Cases in District'],coloraxis="coloraxis")),2,1)

#fig.add_trace(go.Bar(x=Kh_case['District/City'],y=Kh_case['Total Cases in District'],marker=dict(color=Kh_case['Total Cases in District'],coloraxis="coloraxis")),2,2)

fig.update_layout(coloraxis=dict(colorscale='Bluered_r'), showlegend=False, title_text="Trend of Coronavirus Cases in Bangladesh on daily basis")
                  
fig.update_layout(plot_bgcolor='rgb(230,230,230)')

fig.show()


# In[15]:


#linear regression
from sklearn.model_selection import train_test_split


# In[16]:


# converting string date to date-time
import datetime as dt
bd['Updated Date'] = pd.to_datetime(bd['Updated Date'])
bd.head()


# In[17]:


# converting date-time to ordinal

bd['Updated Date'] = bd['Updated Date'].map(dt.datetime.toordinal)
bd.head()


# In[18]:


# Getting Dependent varial & independent variable
x=bd['Updated Date']
y=bd['Total Cases in District']


# In[19]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[20]:


from sklearn.linear_model import LinearRegression


# In[21]:


lr = LinearRegression()


# In[22]:


import numpy as np

lr.fit(np.array(x_train).reshape(-1,1),np.array(y_train).reshape(-1,1))


# In[23]:


bd.tail()


# In[24]:


y_pred=lr.predict(np.array(x_test).reshape(-1,1))


# In[25]:


from sklearn.metrics import mean_squared_error


# In[26]:


mean_squared_error(x_test,y_pred)


# In[27]:


lr.predict(np.array([[737775]]))


# In[ ]:




