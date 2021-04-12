#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
covid=pd.read_csv('covid.csv')
covid.tail()


# In[2]:


#This means how many data is in here.
covid.shape


# In[3]:


#country base prediction
covid["location"].value_counts()


# In[4]:


#for see the categorical description
covid.describe(include='O')


# In[5]:


#for all description
covid.describe()


# In[6]:


#checking if there are any null values in column
covid.isna().any()


# In[7]:


#sum of null values
covid.isna().sum()


# In[8]:


bd_case=covid[covid["location"]=="Bangladesh"]


# In[9]:


bd_case.head()


# In[10]:


bd_case.tail(10)


# In[11]:


import seaborn as sns
from matplotlib import pyplot as plt


# In[12]:


#Total cases per day
sns.set(rc={'figure.figsize':(15,10)})
sns.lineplot(x="date",y="total_cases",data=bd_case)
plt.show()


# In[55]:


import plotly
import plotly.express as px
fig = px.bar(bd_case, x="date",y="new_cases", barmode='group',height=400)
fig.update_layout(title_text='Trend of Coronavirus Cases in Bangladesh on daily basis',plot_bgcolor='rgb(230,230,230)')
fig.show()


# In[13]:


#Last 5 days
bd_last_5_days=bd_case.tail()


# In[56]:


import plotly
import plotly.express as px
fig = px.bar(bd_last_5_days, x="date",y="new_cases", barmode='group',height=400)
fig.update_layout(title_text='Trend of Coronavirus Cases in Bangladesh on daily basis',plot_bgcolor='rgb(230,230,230)')
fig.show()


# In[14]:


#Total cases per last 5 days
sns.set(rc={'figure.figsize':(15,10)})
sns.lineplot(x="date",y="total_cases",data=bd_last_5_days)
plt.show()


# In[15]:


#Total test cases per day
sns.set(rc={'figure.figsize':(15,10)})
sns.lineplot(x="date",y="total_tests",data=bd_case)
plt.show()


# In[16]:


#Total cases per last 5 days
sns.set(rc={'figure.figsize':(15,10)})
sns.lineplot(x="date",y="total_tests",data=bd_last_5_days)
plt.show()


# In[17]:


#Understanding cases between Bd, In, Britain
bd_in_usa=covid[(covid["location"]=="Bangladesh") | (covid["location"]=="India") | (covid["location"]=="United States")]


# In[18]:


sns.set(rc={'figure.figsize':(15,10)})
sns.barplot(x="location",y="total_cases",data=bd_in_usa,hue="date")
plt.show()


# In[60]:


bd_case=covid[covid["location"]=="Bangladesh"]
in_case=covid[covid["location"]=="India"]
us_case=covid[covid["location"]=="United States"]

import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(
rows=2, cols=2,
specs=[[{}, {}],
      [{"colspan":2}, None]],
subplot_titles=("Bangladesh","India","United States"))

fig.add_trace(go.Bar(x=bd_case['date'],y=bd_case['total_cases'],marker=dict(color=bd_case['total_cases'],coloraxis="coloraxis")),1,1)

fig.add_trace(go.Bar(x=in_case['date'],y=in_case['total_cases'],marker=dict(color=in_case['total_cases'],coloraxis="coloraxis")),1,2)

fig.add_trace(go.Bar(x=us_case['date'],y=us_case['total_cases'],marker=dict(color=us_case['total_cases'],coloraxis="coloraxis")),2,1)

fig.update_layout(coloraxis=dict(colorscale='Bluered_r'), showlegend=False, title_text="Trend of Coronavirus Cases in Bangladesh on daily basis")
                  
fig.update_layout(plot_bgcolor='rgb(230,230,230)')

fig.show()


# In[19]:


#Brazil and Italy Case Understanding
bz_it = covid[(covid["location"]=="Brazil") | (covid["location"]=="Italy")]


# In[20]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# In[22]:


sns.set(rc={'figure.figsize':(15,10)})
colors = sns.color_palette('Reds_d', n_colors=len(bz_it))
sns.barplot(x="location",y="total_cases",data=bz_it,hue="date")
plt.show()


# In[23]:


#Predicting latest data
last_day_cases=covid[covid["date"]=="2021-04-08"]
last_day_cases


# In[24]:


#Maximum Cases per day latest data
max_cases_country=last_day_cases.sort_values(by="total_cases",ascending=False)
max_cases_country


# In[25]:


#Top 5 country of maximum cases
max_cases_country[1:6]


# In[26]:


#Making barplot for countries with top cases
sns.barplot(x="location",y="total_cases",data=max_cases_country[1:6],hue="location")
plt.show()


# In[27]:


bd_case.head()


# In[28]:


max_cases_country.style.background_gradient(cmap='Reds')


# In[29]:


#linear regression
from sklearn.model_selection import train_test_split


# In[30]:


# converting string date to date-time
import datetime as dt
bd_case['date'] = pd.to_datetime(bd_case['date'])
bd_case.head()


# In[31]:


# converting date-time to ordinal

bd_case['date'] = bd_case['date'].map(dt.datetime.toordinal)
bd_case.head()


# In[42]:


# Getting Dependent varial & independent variable
x=bd_case['date']
y=bd_case['total_cases']


# In[43]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[44]:


from sklearn.linear_model import LinearRegression


# In[45]:


lr = LinearRegression()


# In[46]:


import numpy as np

lr.fit(np.array(x_train).reshape(-1,1),np.array(y_train).reshape(-1,1))


# In[47]:


bd_case.tail()


# In[48]:


y_pred=lr.predict(np.array(x_test).reshape(-1,1))


# In[49]:


from sklearn.metrics import mean_squared_error


# In[50]:


mean_squared_error(x_test,y_pred)


# In[51]:


lr.predict(np.array([[737892]]))


# In[52]:


lr.predict(np.array([[737890]]))


# In[ ]:




