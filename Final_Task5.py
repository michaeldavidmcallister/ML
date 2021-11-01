#!/usr/bin/env python
# coding: utf-8

# # Task: Covid-19 Data Analysis
# ### This notebook is used to understand the comprehension of Data Analysis techniques using Pandas library.

# ### Data Source: 
# https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_daily_reports
# 
# ### File naming convention
# 
# MM-DD-YYYY.csv in UTC.
# 
# ### Field description
# 
# - Province_State: China - province name; US/Canada/Australia/ - city name, state/province name; Others - name of the event (e.g., "Diamond Princess" cruise ship); other countries - blank.
# 
# - Country_Region: country/region name conforming to WHO (will be updated).
# 
# - Last_Update: MM/DD/YYYY HH:mm (24 hour format, in UTC).
# 
# - Confirmed: the number of confirmed cases. For Hubei Province: from Feb 13 (GMT +8), we report both clinically diagnosed and lab-confirmed cases. For lab-confirmed cases only (Before Feb 17), please refer to who_covid_19_situation_reports. For Italy, diagnosis standard might be changed since Feb 27 to "slow the growth of new case numbers." (Source)
# 
# - Deaths: the number of deaths.
# 
# - Recovered: the number of recovered cases.

# ### Question 1

# #### Read the dataset

# In[1]:


import pandas as pd


# In[3]:


df = pd.read_csv (r'C:\Users\Mike\Documents\AI - ML I\Assignments\Task 5\09-01-2020.txt')
print (df)


# #### Display the top 5 rows in the data

# In[5]:


df.head()


# #### Show the information of the dataset

# In[6]:


df.info()


# #### Show the sum of missing values of features in the dataset

# In[7]:


df.isna().sum()


# ### Question 2

# #### Show the number of Confirmed cases by Country

# In[4]:


df[['Country_Region','Confirmed']].groupby('Country_Region').Confirmed.sum()


# #### Show the number of Deaths by Country

# In[5]:


df[['Country_Region','Deaths']].groupby('Country_Region').Deaths.sum()


# #### Show the number of Recovered cases by Country

# In[6]:


df[['Country_Region','Recovered']].groupby('Country_Region').Recovered.sum()


# #### Show the number of Active Cases by Country

# In[7]:


df[['Country_Region','Active']].groupby('Country_Region').Active.sum()


# #### Show the latest number of Confirmed, Deaths, Recovered and Active cases Country-wise

# In[8]:


df.groupby('Country_Region').agg({'Confirmed': 'sum','Deaths':'sum', 'Recovered':'sum', 'Active':'sum'})


# ### Question 3

# ### Show the countries with no recovered cases

# In[9]:


df2 = df[['Country_Region','Confirmed']].groupby('Country_Region').Confirmed.sum().reset_index()


# In[83]:


df2.loc[df2['Confirmed']==0]


# #### Show the countries with no confirmed cases

# In[88]:


df2 = df[['Country_Region','Confirmed']].groupby('Country_Region').Confirmed.sum().reset_index()


# In[90]:


df2.loc[df2['Confirmed']==0]


# #### Show the countries with no deaths

# In[92]:


df3 = df[['Country_Region','Deaths']].groupby('Country_Region').Deaths.sum().reset_index()


# In[93]:


df3.loc[df3['Deaths']==0]


# In[99]:


df4 = df.groupby('Country_Region').agg({'Confirmed': 'sum','Deaths':'sum', 'Recovered':'sum'}).reset_index()


# In[101]:


df4.loc[df4['Deaths']==0]


# ### Question 4

# #### Show the Top 10 countries with Confirmed cases

# In[102]:


#Used DF4 from previous question, as it already summed up the total for each country
df4.sort_values('Confirmed',ascending=False).head(10)


# #### Show the Top 10 Countries with Active cases

# In[103]:


df5 = df.groupby('Country_Region').agg({'Confirmed': 'sum','Deaths':'sum', 'Recovered':'sum', 'Active':'sum'}).reset_index()


# In[104]:


df5.sort_values('Active',ascending=False).head(10)


# ### Question 5

# #### Plot Country-wise Total deaths, confirmed, recovered and active casaes where total deaths have exceeded 50,000

# In[13]:


import matplotlib.pyplot as plt


# In[36]:


#Used oringal df as it already had sum aggregates by country

df_plot = df.groupby('Country_Region').agg({'Confirmed': 'sum','Deaths':'sum', 'Recovered':'sum', 'Active':'sum'}).reset_index()
df_plot = df_plot.sort_values(by='Deaths', ascending=False)
df_plot = df_plot[df_plot['Deaths']>50000].reset_index()
df_plot
plt.figure(figsize=(6, 5))
plt.plot(df_plot['Country_Region'], df_plot['Deaths'],color='red')
plt.plot(df_plot['Country_Region'], df_plot['Confirmed'],color='green')
plt.plot(df_plot['Country_Region'], df_plot['Recovered'], color='blue')
plt.plot(df_plot['Country_Region'], df_plot['Active'], color='black')
 
plt.title('Deaths over 50k, Confirmed, Recovered and Active Cases by Country')
plt


# ### Question 6

# ### Plot Province/State wise Deaths in USA

# In[9]:


import plotly.express as px
pio.renderers.default = 'iframe' # or 'notebook' or 'colab' or 'jupyterlab'


# In[3]:


covid_data= pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/01-09-2021.csv')


# In[4]:


covid_data.columns


# In[10]:


us_data = covid_data[covid_data['Country_Region']=='US']
state = us_data.groupby('Province_State').Deaths.sum().reset_index()
state_deaths = px.bar(state, x='Province_State', y='Deaths', title='State wise deaths reported of COVID-19 in USA', text='Deaths')
state_deaths.show()


# ### Question 7

# ### Plot Province/State Wise Active Cases in USA

# In[6]:


covid_data['Active'] = covid_data['Confirmed'] - covid_data['Deaths'] - covid_data['Recovered']
us_data = covid_data[covid_data['Country_Region']=='US'].drop(['Country_Region'], axis=1)
us_data = us_data[us_data.sum(axis = 1) > 0]
 
us_data = us_data.groupby(['Province_State'])['Active'].sum().reset_index()
us_data_death = us_data[us_data['Active'] > 0]
state_fig = px.bar(us_data_death, x='Province_State', y='Active', title='State wise recovery cases of COVID-19 in USA', text='Active')
state_fig.show()


# ### Question 8

# ### Plot Province/State Wise Confirmed cases in USA

# In[19]:


covid_data['Active'] = covid_data['Confirmed'] - covid_data['Deaths'] - covid_data['Recovered']
combine_us_data = covid_data[covid_data['Country_Region']=='US'].drop(['Country_Region'], axis=1)
combine_us_data = combine_us_data[combine_us_data.sum(axis = 1) > 0]
combine_us_data = combine_us_data.groupby(['Province_State'])['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
combine_us_data = pd.melt(combine_us_data, id_vars='Province_State', value_vars=['Confirmed', 'Deaths', 'Recovered', 'Active'], value_name='Count', var_name='Case')
fig = px.bar(combine_us_data, x='Province_State', y='Count', text='Count', barmode='group', color='Case', title='USA State wise combine number of confirmed, deaths, recovered, active COVID-19 cases')
fig.show()


# ### Question 9

# ### Plot Worldwide Confirmed Cases over time

# In[8]:


import plotly.express as px
import plotly.io as pio


# In[20]:


grouped = covid_data.groupby('Last_Update')['Last_Update', 'Confirmed', 'Deaths'].sum().reset_index()
fig = px.line(grouped, x="Last_Update", y="Confirmed",
             title="Worldwide Confirmed Novel Coronavirus(COVID-19) Cases Over Time")
fig.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




