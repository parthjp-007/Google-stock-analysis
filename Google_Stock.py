#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import dtale
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


df= pd.read_csv(r"D:\datasets\Time series\GOOGL_2006-01-01_to_2018-01-01.csv")
df.head()


# In[3]:


df['Date'] = pd.to_datetime(df['Date'], yearfirst=True)
df.head()


# In[4]:


dtale.show(df)


# In[5]:


# DISCLAIMER: 'df' refers to the data you passed in when calling 'dtale.show'

import pandas as pd

if isinstance(df, (pd.DatetimeIndex, pd.MultiIndex)):
	df = df.to_frame(index=False)

# remove any pre-existing indices for ease of use in the D-Tale code, but this is not required
# df = df.reset_index().drop('index', axis=1, errors='ignore')
df.columns = [str(c) for c in df.columns]  # update columns to strings in case they are numbers

chart_data = pd.concat([
	df['Date'],
	df['High'],
], axis=1)
# chart_data = chart_data.sort_values(['Date'])
chart_data = chart_data.rename(columns={'Date': 'x'})
chart_data = chart_data.dropna()

import plotly.graph_objs as go

charts = []
line_cfg = {'line': {'shape': 'spline', 'smoothing': 0.3}, 'mode': 'lines'}
charts.append(go.Scatter(
	x=chart_data['x'], y=chart_data['High'], name='High', **line_cfg
))
figure = go.Figure(data=charts, layout=go.Layout({
    'legend': {'orientation': 'h'},
    'title': {'text': 'High by Date'},
    'xaxis': {'title': {'text': 'Date'}},
    'yaxis': {'title': {'text': 'High'}, 'type': 'linear'}
}))

# If you're having trouble viewing your chart in your notebook try passing your 'chart' into this snippet:
#
# from plotly.offline import iplot, init_notebook_mode
#
# init_notebook_mode(connected=True)
# chart.pop('id', None) # for some reason iplot does not like 'id'
# iplot(chart)


# In[6]:


figure


# In[4]:


df.set_index(df['Date'], inplace=True)
df.head()


# In[5]:


series = df.resample('D', on='Date').High.sum()
series.head()


# In[6]:


train = series[:-1]
test = series[-1:]


# In[ ]:





# In[8]:


from statsmodels.tsa.seasonal import seasonal_decompose
fig = seasonal_decompose(train, model = 'additve')
fig.plot()


# In[9]:


from statsmodels.tsa.stattools import adfuller

def check_stationarity(timeseries): 

    #----------------------------------------------------------
    #rolling statics
    rol_mean = timeseries.rolling(window = 12).mean()
    rol_std = timeseries.rolling(window = 12).std() 
    #----------------------------------------------------------
    #plot rolling statistics
    plt.figure(figsize=(14,4))
    plt.plot(timeseries, color = 'b', label = 'Original')
    plt.plot(rol_mean, color = 'r', label = 'Rolling Mean')
    plt.plot(rol_std, color = 'g', label = 'Rolling Std')
    plt.legend()
    #----------------------------------------------------------
    print('DickeyFuller Test') 
    result=adfuller(timeseries) 
    print(f'ADF Statistic: {result[0]}') 
    print(f'p-Value: {result[1]}') 
    print('Critical Values:') 
    for key, value in result[4].items(): 
        print('\t%s : %0.3f'%(key, value))   


# In[10]:


check_stationarity(train)


# In[12]:


series_diff = train.diff(2)
check_stationarity(series_diff[2:])


# In[13]:


from statsmodels.tsa.seasonal import seasonal_decompose 
fig=seasonal_decompose(series_diff[2:], model='additive')
fig.plot() 
plt.title('Noise', color='red', size=15)  


# In[14]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
#-------------------------------------------------------------
plot_acf(series_diff[2:], lags=10, alpha=0.05)  
plt.title('ACF', color='red', size=15) 
plt.show() 
#-------------------------------------------------------------
plot_pacf(series_diff[2:], lags=10,alpha=0.05, method='ols') 
plt.title('PACF', color='red', size=15) 
plt.show() 


# In[ ]:


from statsmodels.tsa.arima_model import ARIMA
model=ARIMA(series_diff[2:], order=(9,2,0)) #(p, d, q) if d=0, this model will be same as ARMA.
result=model.fit() 


# In[ ]:


# from statsmodels.tsa.arima_model import ARIMA
# model=ARIMA(series_diff[2:], order=(9,2,0)) #(p, d, q) if d=0, this model will be same as ARMA.
# Result=model.fit() 


# In[ ]:


series.tail()


# In[ ]:


forecast= train[-1] + np.cumsum(result.forecast(5,)[0]).cumsum()
# Forecast= train[-1] + np.cumsum(Result.forecast(5,)[0]).cumsum()
x= np.arange(len(series))
plt.figure(figsize=(12,6))
plt.plot(x,series, label='original')
x= x[-5:] 
plt.plot(x,forecast, color='red', label='forecast') 
plt.legend()
plt.show()


# In[ ]:


Forecast= train[-1] + np.cumsum(Result.forecast(5,)[0]).cumsum()
x= np.arange(len(series))
plt.figure(figsize=(12,6))
plt.plot(x,series, label='original')
x= x[-5:] 
plt.plot(x,Forecast, color='red', label='forecast') 
plt.legend()
plt.show()


# In[ ]:


forecast


# In[35]:


series[-100:-95]


# In[ ]:


Forecast


# In[ ]:


new = Forecast + forecast 
new = new/2
new


# In[ ]:


x= np.arange(len(series))
plt.figure(figsize=(12,6))
plt.plot(x,series, label='original')
x= x[-5:] 
plt.plot(x,new, color='red', label='forecast') 
plt.legend()
plt.show()

