#!/usr/bin/env python
# coding: utf-8

# ## Multiple Linear Regression Model for BoomBikes services
# 
#     - A multiple linear regression model will be created for BoomBikes services. 
#     - Will check with the differrent features available in the dataset which is affecting our predictor variable 
#     - i.e. 'cnt'
#     - Variables Classifications :
#     - numerical variables:
#         - 'cnt', 'temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered'
#     - categorical variables:
#         - 'season', 'yr', 'mnth', 'weekday', 'weathersit'
#     - binary variables:
#         - 'holiday', 'workingday'

# ## Reading, Understanding, EDA & Visualisation.

# In[1]:


# loading the dataset using our python libraries by importing the libraries
# importing the libraries as & whenever required.

import pandas as pd
import numpy as np
import datetime

import sklearn
import statsmodels.api as sm


# In[2]:


# loading the data using pandas

boombikes_df = pd.read_csv("/Users/ashutosh/UG/Linear_Regression/day.csv")

# after loading will just have look on the dataset
boombikes_df.head(10)


# In[3]:


# performing the EDA on the above dataset

# droping the variables which are not important for our analysis.
# 'instant' variable which just showcase the index & it won't help in our modeling.

if 'instant' in boombikes_df.columns:
    boombikes_df = boombikes_df.drop('instant', axis=1)

# we will drop 'dteday' columns to as we already have month & year columns for analysis.
if 'dteday' in boombikes_df.columns:
    boombikes_df = boombikes_df.drop('dteday', axis=1)

# we will map the categorical variables to their values mentioned in the data dictionary given.

# season (1:spring, 2:summer, 3:fall, 4:winter)
boombikes_df['season'] = boombikes_df['season'].map({1: 'spring', 2: 'summer', 3: 'fall', 4:'winter'})

# year:yr (0: 2018, 1:2019)
boombikes_df['yr'] = boombikes_df['yr'].map({0: '2018', 1: '2019'})

# month (1:January ... 12:December)
boombikes_df['mnth'] = boombikes_df['mnth'].map({1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'April', 5: 'May', 6: 'June',
                                                7: 'July', 8: 'Aug', 9: 'Sept', 10: 'Oct', 11: 'Nov', 12: 'Dec'})

# day of the week
# 0: Monday & so on. 
boombikes_df['weekday'] = boombikes_df['weekday'].map({0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thurs', 4: 'Fri', 
                                                       5: 'Sat', 6: 'Sun'})

# weathersit : 
# - 1: Clear, Few clouds, Partly cloudy, Partly cloudy ('Partly Cloudy')
# - 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist (Mist)
# - 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds (Low Rain)
# - 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog (Not present in the dataset)  (Heavy Rain)
boombikes_df['weathersit'] = boombikes_df['weathersit'].map({1: 'Partly Cloudy', 2: 'Mist', 3: 'Low Rain', 
                                                       4: 'Heavy Rain'})


# In[4]:


boombikes_df.info()


# In[5]:


# visualising the numerical variables using the scatter plots
# importing the visualisation libraries as required

import matplotlib.pylab as plt
import seaborn as sns

sns.pairplot(boombikes_df[['cnt','casual', 'registered', 'temp', 'atemp', 'hum', 'windspeed']])
plt.show()


# In[6]:


# visualising our categorical data with the total count of users (registered & casual)
# 'season', 'yr', 'mnth', 'weekday', 'weathersit'

plt.figure(figsize=(20, 16))

# season (1:spring, 2:summer, 3:fall, 4:winter)
plt.subplot(2, 3, 1)
sns.boxplot(x="season", y="cnt", data=boombikes_df)

# year:yr (0: 2018, 1:2019)
plt.subplot(2, 3, 2)
sns.boxplot(x="yr", y="cnt", data=boombikes_df)

# month (1 to 12)
plt.subplot(2, 3, 3)
sns.boxplot(x="mnth", y="cnt", data=boombikes_df)

# day of the week
# 0: Sunday & so on. 
plt.subplot(2, 3, 4)
sns.boxplot(x="weekday", y="cnt", data=boombikes_df)

# weathersit : 
# - 1: Clear, Few clouds, Partly cloudy, Partly cloudy
# - 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
# - 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
# - 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog (Not present in the dataset)
plt.subplot(2, 3, 5)
sns.boxplot(x="weathersit", y="cnt", data=boombikes_df)

plt.show()


# In[7]:


# visualising our categorical data with 'registered users'
# 'season', 'yr', 'mnth', 'weekday', 'weathersit'

plt.figure(figsize=(20, 16))

# season (1:spring, 2:summer, 3:fall, 4:winter)
plt.subplot(2, 3, 1)
sns.boxplot(x="season", y="registered", data=boombikes_df)

# year:yr (0: 2018, 1:2019)
plt.subplot(2, 3, 2)
sns.boxplot(x="yr", y="registered", data=boombikes_df)

# month (1 to 12)
plt.subplot(2, 3, 3)
sns.boxplot(x="mnth", y="registered", data=boombikes_df)

# day of the week
# 0 is treated as Sunday & so on. 
plt.subplot(2, 3, 4)
sns.boxplot(x="weekday", y="registered", data=boombikes_df)

# weathersit : 
# - 1: Clear, Few clouds, Partly cloudy, Partly cloudy
# - 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
# - 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
# - 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog (Not present in the dataset)
plt.subplot(2, 3, 5)
sns.boxplot(x="weathersit", y="registered", data=boombikes_df)

plt.show()


# In[8]:


# visualising our categorical data with 'casual users'
# 'season', 'yr', 'mnth', 'weekday', 'weathersit'

plt.figure(figsize=(20, 16))

# season (1:spring, 2:summer, 3:fall, 4:winter)
plt.subplot(2, 3, 1)
sns.boxplot(x="season", y="casual", data=boombikes_df)

# year:yr (0: 2018, 1:2019)
plt.subplot(2, 3, 2)
sns.boxplot(x="yr", y="casual", data=boombikes_df)

# month (1 to 12)
plt.subplot(2, 3, 3)
sns.boxplot(x="mnth", y="casual", data=boombikes_df)

# day of the week
# 0 is treated as Sunday & so on. 
plt.subplot(2, 3, 4)
sns.boxplot(x="weekday", y="casual", data=boombikes_df)

# weathersit : 
# - 1: Clear, Few clouds, Partly cloudy, Partly cloudy
# - 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
# - 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
# - 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog (Not present in the dataset)
plt.subplot(2, 3, 5)
sns.boxplot(x="weathersit", y="registered", data=boombikes_df)

plt.show()


# ##  Data preparation for modeling
#     - Encoding:
#         - Converting binary variables to 1/0
#         - Other categorical variables to dummy variables
#     - Splitting into train & test
#     - Rescalling of variables

# - numerical variables:
#     - 'cnt', 'temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered'
# - categorical variables:
#     - 'season', 'yr', 'mnth', 'weekday', 'weathersit'
# - binary variables:
#     - 'holiday', 'workingday'

# ## ENCODING THE VARIABLES
# ### Dummy variables using One Hot Encoding technique

# In[9]:


boombikes_df.head(10)


# In[10]:


# displaying all the unique values of our categorical variables
print(boombikes_df['season'].unique())
print(boombikes_df['yr'].unique())
print(boombikes_df['mnth'].unique())
print(boombikes_df['weekday'].unique())
print(boombikes_df['weathersit'].unique())


# In[11]:


# Encoding the variables
# Let's convert the categorical variables to dummy variables so tht we can use them in our modeling.
# as MLR needs numerical data for performing modeling

# creating dummy variables for all of out categorical variables
dummy_new = pd.get_dummies(boombikes_df[['season', 'yr', 'mnth', 'weekday', 'weathersit']], 
                                dtype=int, drop_first=True)


# In[12]:


# so using the 'One Hot Encoding' technique using pandas get_dummies() method to create 'n-1' dummy variables,
# we got the below output of dummy variables

dummy_new.head(10)


# In[13]:


# Concat the dummy variables to our original dataframe i.e. 'boombikes_df'
boombikes_df = pd.concat([boombikes_df, dummy_new], axis=1)

# Drop the original categorical variables
boombikes_df = boombikes_df.drop(['season', 'yr', 'mnth', 'weekday', 'weathersit'], axis=1)


# In[14]:


boombikes_df.head(10)


# ## Splitting into train & test data

# In[15]:


# for this will use sklearn library & their method train_test_split()
from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(boombikes_df, train_size=0.7, test_size=0.3, random_state=100)


# In[16]:


# Now let's standardise our dataset.
# For that will use MinMaxScaler method of sklearn.preprocessing which will get all our values between 0 and 1.

from sklearn.preprocessing import MinMaxScaler

# 1. Initialise the object
scaler = MinMaxScaler()

# create a list of numeric variables (we haven't involved the binary variable & categorical dummy variables)
num_vars = ['cnt','casual', 'registered', 'temp', 'atemp', 'hum', 'windspeed']

# 2. Fit on data
df_train[num_vars] = scaler.fit_transform(df_train[num_vars])
df_train.head()


# In[17]:


len(df_train.columns)
print(df_train.columns)


# In[18]:


# first will visualise a heatmap as we need to check the corelation 
# between the independent variable & dependent variable

df_train = df_train[['cnt','casual',
       'registered','holiday', 'workingday', 'temp', 'atemp', 'hum', 'windspeed',
                     'season_spring', 'season_summer', 'season_winter',
       'yr_2019', 'mnth_Aug', 'mnth_Dec', 'mnth_Feb', 'mnth_Jan', 'mnth_July',
       'mnth_June', 'mnth_Mar', 'mnth_May', 'mnth_Nov', 'mnth_Oct',
       'mnth_Sept', 'weekday_Mon', 'weekday_Sat', 'weekday_Sun',
       'weekday_Thurs', 'weekday_Tue', 'weekday_Wed', 'weathersit_Mist',
       'weathersit_Partly Cloudy']]


# In[19]:


df_train.head(10)


# ## Training the Model

# In[20]:


# first we will draw a heatmap as we need to check the corelation 
# between the independent variable & dependent variable

plt.figure(figsize=(26, 26))
sns.heatmap(df_train.corr(), annot=True, cmap="YlGnBu")
plt.show()


# In[21]:


# X_train, y_train

y_train = df_train.pop('cnt')
X_train = df_train


# In[22]:


# will check first with our highly corelated variable i.e. 'registered'

# add a constant
X_train_sm = sm.add_constant(X_train['registered'])

# create model
lr = sm.OLS(y_train, X_train_sm)

# fit the model
lr_model = lr.fit()

# summary
lr_model.summary()


# - as we can see that we have R-squared value of 0.899 which is highly significant & it also shows that our 
# - dependent variable is highly corelated to each other
# - but we cannot justify with only 1 variable.
# - so we will go further & keep adding 1 variable at a time to check how our model is working.

# In[23]:


# will add 1 more variable now which highly correlated i.e. 'casual'

# add a constant
X_train_sm = sm.add_constant(X_train[['registered', 'casual']])

# create model
lr = sm.OLS(y_train, X_train_sm)

# fit the model
lr_model = lr.fit()

# summary
lr_model.summary()


# - over here we can observe a multicollinearity issue in our modeling
# - so we will drop one of our variable & check again.

# In[24]:


# will just check with 'casual' variable that how significant it is & weather we can keep it or drop it.

# add a constant
X_train_sm = sm.add_constant(X_train['casual'])

# create model
lr = sm.OLS(y_train, X_train_sm)

# fit the model
lr_model = lr.fit()

# summary
lr_model.summary()


# - over here we can observe the R-squared value of 0.450 which means 45% of the variability observed in the dependent variable
# - but in our first model much more high R-squared value so with this we drop this variable
# - also this variable is creating multicollinearity issue in our model.

# In[25]:


# will now add 1 more variable now which highly correlated i.e. 'atemp'

# add a constant
X_train_sm = sm.add_constant(X_train[['registered', 'atemp']])

# create model
lr = sm.OLS(y_train, X_train_sm)

# fit the model
lr_model = lr.fit()

# summary
lr_model.summary()


# - over here we can observe the R-squared value is extremly good i.e. 0.918
# - we can say that 'atemp' variables is also statistically significant till now.

# In[26]:


# now will add 1 more variable now which highly correlated i.e. 'temp'

# add a constant
X_train_sm = sm.add_constant(X_train[['registered', 'atemp', 'temp']])

# create model
lr = sm.OLS(y_train, X_train_sm)

# fit the model
lr_model = lr.fit()

# summary
lr_model.summary()


# - over here we don't have much change in R-squared value but the p-value we observe.
# - p-value for 'atemp' variable is 0.497 as compared to our other variables which is not statistically significant.
# - so we will drop the 'atemp' variable here & add another list of variables which are corelated.

# In[27]:


# till now in our analysis we have only two significant variables i.e. 'registered', 'temp'
# we have drop the variables 'casual' & 'atemp' from our analysis.
# now we will add all the variables which are corelated & will remove variables at a time which has high p-value.

# add a constant
X_train_sm = sm.add_constant(X_train[['holiday', 'workingday', 'temp', 'hum', 'windspeed',
       'registered', 'season_spring', 'season_summer', 'season_winter',
       'yr_2019', 'mnth_Aug', 'mnth_Dec', 'mnth_Feb', 'mnth_Jan', 'mnth_July',
       'mnth_June', 'mnth_Mar', 'mnth_May', 'mnth_Nov', 'mnth_Oct',
       'mnth_Sept', 'weekday_Mon', 'weekday_Sat', 'weekday_Sun',
       'weekday_Thurs', 'weekday_Tue', 'weekday_Wed', 'weathersit_Mist',
       'weathersit_Partly Cloudy']])

# create model
lr = sm.OLS(y_train, X_train_sm)

# fit the model
lr_model = lr.fit()

# summary
lr_model.summary()


# - Now we added all the variables in our modeling.
# - We have statistically significant R-squared value but we can observe that we have high p-value for some of the independent variables
# - we will drop the variables with high p-value after checking the VIF i.e. Variable Inflation Factor
# - High p-value, High VIF (remove them)
# - High-low
#     - High p, low VIF : remove these first
#     - Low p, High VIF: remove these after the ones above
# - Low p, Low VIF (Keep such variables)

# In[28]:


# will use the statsmodel library to check VIF

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
X_train.drop(['casual', 'atemp'], axis=1, inplace=True) # we drop our 2 columns which are not siginificant
vif['Factors'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# In[29]:


# First will check for significance level of High p-value > 0.05 & significance level of High VIF > 10
# we can see 'season_spring' has p-value of 0.777 & also the VIF is 10.87 which is high.
# will drop this


# In[30]:


if 'season_spring' in X_train.columns:
    X_train.drop('season_spring', axis=1, inplace=True)


# In[31]:


# add a constant
X_train_sm = sm.add_constant(X_train[['holiday', 'workingday', 'temp', 'hum', 'windspeed',
       'registered', 'season_summer', 'season_winter',
       'yr_2019', 'mnth_Aug', 'mnth_Dec', 'mnth_Feb', 'mnth_Jan', 'mnth_July',
       'mnth_June', 'mnth_Mar', 'mnth_May', 'mnth_Nov', 'mnth_Oct',
       'mnth_Sept', 'weekday_Mon', 'weekday_Sat', 'weekday_Sun',
       'weekday_Thurs', 'weekday_Tue', 'weekday_Wed', 'weathersit_Mist',
       'weathersit_Partly Cloudy']])

# create model
lr = sm.OLS(y_train, X_train_sm)

# fit the model
lr_model = lr.fit()

# summary
lr_model.summary()


# - we can straight away see that we multicollinearity issue still in our model from the above error message in point no.2
# - The smallest eigenvalue is 1.17e-27 which indicates the eigenvalues of the correlation matrix of the predictor variables can provide insights into multicollinearity.

# In[32]:


# now lets check for VIF

vif = pd.DataFrame()
vif['Factors'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# In[33]:


# again we will repeat the above steps
# we will now drop 'weathersit_Partly Cloudy' variable as it has the high p-value of 0.102 and high VIF of 12.64
# considering the same significance level above

if 'weathersit_Partly Cloudy' in X_train.columns:
    X_train.drop('weathersit_Partly Cloudy', axis=1, inplace=True)


# In[34]:


# again will check for p-values

# add a constant
X_train_sm = sm.add_constant(X_train[['holiday', 'workingday', 'temp', 'hum', 'windspeed',
       'registered', 'season_summer', 'season_winter',
       'yr_2019', 'mnth_Aug', 'mnth_Dec', 'mnth_Feb', 'mnth_Jan', 'mnth_July',
       'mnth_June', 'mnth_Mar', 'mnth_May', 'mnth_Nov', 'mnth_Oct',
       'mnth_Sept', 'weekday_Mon', 'weekday_Sat', 'weekday_Sun',
       'weekday_Thurs', 'weekday_Tue', 'weekday_Wed', 'weathersit_Mist']])

# create model
lr = sm.OLS(y_train, X_train_sm)

# fit the model
lr_model = lr.fit()

# summary
lr_model.summary()


# In[35]:


# now lets check for VIF

vif = pd.DataFrame()
vif['Factors'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# - now from above we can see that VIF has been reduced drastically & we don't have high p-values for high VIF.
# - we will now check for 
# - High-low
#     - High p, low VIF : remove these first
#     - Low p, High VIF: remove these after the ones above

# In[36]:


# We have 'mnth_Sept' variable for high-p & low VIF, so we will drop this first

if 'mnth_Sept' in X_train.columns:
    X_train.drop('mnth_Sept', axis=1, inplace=True)


# In[37]:


# let's check again for p-values 

# add a constant
X_train_sm = sm.add_constant(X_train[['holiday', 'workingday', 'temp', 'hum', 'windspeed',
       'registered', 'season_summer', 'season_winter',
       'yr_2019', 'mnth_Aug', 'mnth_Dec', 'mnth_Feb', 'mnth_Jan', 'mnth_July',
       'mnth_June', 'mnth_Mar', 'mnth_May', 'mnth_Nov', 'mnth_Oct', 'weekday_Mon', 'weekday_Sat', 'weekday_Sun',
       'weekday_Thurs', 'weekday_Tue', 'weekday_Wed', 'weathersit_Mist']])

# create model
lr = sm.OLS(y_train, X_train_sm)

# fit the model
lr_model = lr.fit()

# summary
lr_model.summary()


# In[38]:


# now lets check for VIF

vif = pd.DataFrame()
vif['Factors'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# In[39]:


# now will remove 'holiday' as it has high p-value of 0.926 & low VIF of 3.78

if 'holiday' in X_train.columns:
    X_train.drop('holiday', axis=1, inplace=True)


# In[40]:


# let's check again for p-values 

# add a constant
X_train_sm = sm.add_constant(X_train[['workingday', 'temp', 'hum', 'windspeed',
       'registered', 'season_summer', 'season_winter',
       'yr_2019', 'mnth_Aug', 'mnth_Dec', 'mnth_Feb', 'mnth_Jan', 'mnth_July',
       'mnth_June', 'mnth_Mar', 'mnth_May', 'mnth_Nov', 'mnth_Oct', 'weekday_Mon', 'weekday_Sat', 'weekday_Sun',
       'weekday_Thurs', 'weekday_Tue', 'weekday_Wed', 'weathersit_Mist']])

# create model
lr = sm.OLS(y_train, X_train_sm)

# fit the model
lr_model = lr.fit()

# summary
lr_model.summary()


# - Now from above we can see that the multicollinearity issue is not present in our modeling from the above selected variables. But we cannot be sure as we still have high p-values.

# In[41]:


# let's check for VIF

vif = pd.DataFrame()
vif['Factors'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# In[42]:


# we still have high p-value for 'mnth_Nov' dummy variable & it has low VIF of 4.01 so we will drop this also.

if 'mnth_Nov' in X_train.columns:
    X_train.drop('mnth_Nov', axis=1, inplace=True)


# In[43]:


# let's check again for p-values 

# add a constant
X_train_sm = sm.add_constant(X_train[['workingday', 'temp', 'hum', 'windspeed',
       'registered', 'season_summer', 'season_winter',
       'yr_2019', 'mnth_Aug', 'mnth_Dec', 'mnth_Feb', 'mnth_Jan', 'mnth_July',
       'mnth_June', 'mnth_Mar', 'mnth_May', 'mnth_Oct', 'weekday_Mon', 'weekday_Sat', 'weekday_Sun',
       'weekday_Thurs', 'weekday_Tue', 'weekday_Wed', 'weathersit_Mist']])

# create model
lr = sm.OLS(y_train, X_train_sm)

# fit the model
lr_model = lr.fit()

# summary
lr_model.summary()


# - we still have p-values but let's check for VIF.

# In[44]:


# let's check for VIF

vif = pd.DataFrame()
vif['Factors'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# In[45]:


# we still have high p-value for 'yr_2019' of 0.809 & it has low VIF of 5.45 so we will drop this also.

if 'weekday_Sun' in X_train.columns:
    X_train.drop('weekday_Sun', axis=1, inplace=True)


# In[46]:


# let's check again for p-values 

# add a constant
X_train_sm = sm.add_constant(X_train[['workingday', 'temp', 'hum', 'windspeed',
       'registered', 'season_summer', 'season_winter',
       'mnth_Aug', 'mnth_Dec', 'mnth_Feb', 'mnth_Jan', 'mnth_July',
       'mnth_June', 'mnth_Mar', 'mnth_May', 'mnth_Oct', 'weekday_Mon', 'weekday_Sat', 'yr_2019',
       'weekday_Thurs', 'weekday_Tue', 'weekday_Wed', 'weathersit_Mist']])

# create model
lr = sm.OLS(y_train, X_train_sm)

# fit the model
lr_model = lr.fit()

# summary
lr_model.summary()


# In[47]:


# let's check for VIF

vif = pd.DataFrame()
vif['Factors'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# In[48]:


# Now will check for low p-values & high VIF which is not relative to our modeling.
# will drop 'hum' which is not relative to our modeling so will drop it.

if 'hum' in X_train.columns:
    X_train.drop('hum', axis=1, inplace=True)


# In[49]:


# let's check again for p-values 

# add a constant
X_train_sm = sm.add_constant(X_train[['workingday', 'temp', 'windspeed',
       'registered', 'season_summer', 'season_winter',
       'mnth_Aug', 'mnth_Dec', 'mnth_Feb', 'mnth_Jan', 'mnth_July',
       'mnth_June', 'mnth_Mar', 'mnth_May', 'mnth_Oct', 'weekday_Mon', 'weekday_Sat', 'yr_2019',
       'weekday_Thurs', 'weekday_Tue', 'weekday_Wed', 'weathersit_Mist']])

# create model
lr = sm.OLS(y_train, X_train_sm)

# fit the model
lr_model = lr.fit()

# summary
lr_model.summary()


# In[50]:


# let's check for VIF

vif = pd.DataFrame()
vif['Factors'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# ## Now let's do some Residual Analysis

# In[51]:


# will first compute the predictor values of y_train

y_train_pred = lr_model.predict(X_train_sm)


# In[52]:


# compute the residuals by plotting a histogram
res = y_train - y_train_pred
sns.displot(res)


# - As we can see our histogram as proper standard deviation of 0. So we can say that our predictor residuals are good.

# ## Now let's do Predictions & Evaluation on our test set

# - We need to do same transformation on the test set which we did on the train set

# In[53]:


# create a list of numeric variables (we haven't involved the binary variable & categorical dummy variables)
num_vars = ['cnt','casual', 'registered', 'temp', 'atemp', 'hum', 'windspeed']

# we will not fit the test set.
# we will directly transform the test set with our scaler as it will as already learned the mean/max values from training data set.

# Transform the test data 
df_test[num_vars] = scaler.transform(df_test[num_vars])
df_test.describe()


# In[54]:


y_test = df_test.pop('cnt')
X_test = df_test


# In[55]:


# Add a constant to make prediction to X_test
X_test_sm = sm.add_constant(X_test)


# In[56]:


# Now will drop the variables which are we have dropped from training set also as it will give us the error 
# when the our constant will try to multiply with our variables whiich are no their.


# In[57]:


X_test_sm = X_test_sm.drop(["hum","weekday_Sun","mnth_Nov","holiday","mnth_Sept",
                            "weathersit_Partly Cloudy","season_spring","atemp","casual"], axis=1)


# In[58]:


# Let's do prediction
y_test_pred = lr_model.predict(X_test_sm)


# In[59]:


# now evaluate the model
# i.e. compute the r-squared
# for that will import the method r2_score from sklearn

from sklearn.metrics import r2_score

r2_score(y_true=y_test, y_pred=y_test_pred)


# ### So from above r2_score we can say that our test data set is also performing well as the R-squared of our training data set is quite similar.

# - So from above analysis we can get the below MLR equation for our predicted variables.
# - ve coefficients in our model indicate that there is an inverse relationship between the independent variable and the dependent variable. 
# - as the value of the independent variable decreases, the predicted value of the dependent variable tends to decrease as well, and vice versa.
# - our +ve variables shows us that if some unit increase then our dependent variable also increase.

# ### EQUATION 
# 
# $ cnt = 0.0629−0.1094*workingday+0.1016*temp−0.0417*windspeed+0.9462*registered+0.0238*season_summer-0.0223*season_winter-0.0079*mnth_Aug-0.0194*mnth_Dec-0.0177*mnth_Feb-0.0131*mnth_Jan-0.0128*mnth_July-0.0145*mnth_June+0.0073*mnth_Mar-0.0119*mnth_May+0.0236*mnth_Oct+0.0004*weekday_Mon+0.0129*weekday_Sat-0.0045*yr_2019-0.0128*weekday_Thurs-0.0064*weekday_Tue-0.0093*weekday_Wed-0.0079*weathersit_Mist $

# In[61]:


# Validating the assumptions of Linear Regression

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Plotting the diagonal line
plt.xlabel('Observed')
plt.ylabel('Predicted')
plt.title('Observed vs Predicted Values')
plt.show()


# - As we can see that there is a linear relationship between our observed values & our predicted values. Our scatter plot shows that there is no voilation of linearity assumption
