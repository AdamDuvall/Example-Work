#Loading Libraries
#%%
import pandas as pd
import numpy as np
import re
import sklearn
import xgboost as xgb
import seaborn as sns
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import matplotlib.pyplot as plt
from statsmodels.discrete.discrete_model import Probit
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn.metrics import mean_squared_error
import statsmodels.formula.api as smf 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, ElasticNet, Lasso, LassoCV
#%%
df = pd.read_csv(r'C:\Users\Adam\Desktop\Andy Assignment 3\Building_Permits.csv')

#%%
#Create Variable For Difference from Filing and Issue Dates
df['Issued Date'] = pd.to_datetime(df['Issued Date'])
df['Filed Date']  = pd.to_datetime(df['Filed Date'])

datediff = df['Issued Date'] - df['Filed Date']

df['datediff']=datediff.values

#%%
#Encoding variables of interest
df['Neighborhoods - Analysis Boundaries']= df['Neighborhoods - Analysis Boundaries'].astype('category')
df['Neighborhoods Cat']= df['Neighborhoods - Analysis Boundaries'].cat.codes

df["TIDF Compliance"] = df["TIDF Compliance"].astype('category')
df["TIDF Cat"] = df["TIDF Compliance"].cat.codes

df["Fire Only Permit"] = df["Fire Only Permit"].astype('category')
df["Fire Only Cat"] = df["Fire Only Permit"].cat.codes

df['Issued Date'] = pd.to_datetime(df['Issued Date'])
df['Filed Date']  = pd.to_datetime(df['Filed Date'])
datediff = df['Issued Date'] - df['Filed Date']
df['datediff']=datediff.values

df['datediff'] = pd.to_numeric(df['datediff'].dt.days, downcast='integer')

#%%
#Cleaning unnecessary features
del df['Permit Number']
del df['Permit Type Definition']
del df['Block']
del df['Lot']
del df['Street Number']
del df['Street Number Suffix']
del df['Street Name']
del df['Street Suffix']
del df['Unit']
del df['Unit Suffix']
del df['Description']
del df['Current Status Date']
del df['Current Status']
del df['Completed Date']
del df['First Construction Document Date']
del df['Structural Notification']
del df['Voluntary Soft-Story Retrofit']
del df['Fire Only Permit']
del df['Permit Expiration Date']
del df['Existing Use']
del df['Proposed Use']
del df['Record ID']
del df['Location']
del df['Supervisor District']
del df['Site Permit']
del df['Existing Construction Type Description']
del df['Proposed Construction Type Description']
del df['Permit Creation Date']
del df['Revised Cost']
del df['TIDF Compliance']
del df['Proposed Construction Type']
del df['Proposed Units']
del df['Number of Proposed Stories']
del df['Filed Date']
del df['Issued Date']
del df['Neighborhoods - Analysis Boundaries']


#%%
#Drop NA's
df.dropna(inplace=True)

#%%
#Categorizing amount of wait time for each of our entries 
def rating_df(df):
    if (df['datediff'] <=7):
        return 'Within One Week'
    elif (df['datediff'] >= 8) and (df['datediff'] <=31):
        return 'Two Weeks To One Month'
    elif (df['datediff'] >= 32) and (df['datediff'] <=62):
        return 'One to Two Months'
    elif (df['datediff'] >= 63) and (df['datediff'] <=182):
        return 'Three Months To Six Months'
    elif (df['datediff'] >= 183) and (df['datediff'] <=365):
        return 'Six Months To One Year'
    elif (df['datediff'] >= 366):
        return 'Longer Than One Year'
    
#%%
df['datediff'] = df['datediff'].astype('category')
df['DatedDiffCat'] = df['datediff'].cat.codes

df['datediff']=df.apply(rating_df, axis = 1)

y= df.DatedDiffCat

x=df.drop(['datediff', 'DatedDiffCat'], axis=1)
x1_train, x1_test, y1_train, y1_test=train_test_split(x,y,test_size=0.2)

#%%
#Regression Tree
tree = DecisionTreeRegressor(random_state=1,ccp_alpha=0.004)
tree.fit(x1_train,y1_train)
y_pred_tree = tree.predict(x1_test)
plot_tree(tree,proportion=True,filled=True, class_names=True, feature_names=["Permit Type", "Number of Existing Stories", "Estimated Cost","Estimated Cost","Existing Units", "Plansets", "Existing Construction Type", "Zipcode", "Neighborhood Cat", "Fire Cat","TIDF Cat"])
MSE_Tree = mean_squared_error(y1_test, y_pred_tree)
MSE_Tree

#%%
#KNN
import statistics
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import KFold

from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics

# k1 1

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2)

model = KNeighborsRegressor(n_neighbors=1)
model.fit(xtrain,ytrain)
pred_y = model.predict(xtest)

K1mse1 = mean_squared_error(ytest, pred_y)

# %%
# k1 2

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2)

model = KNeighborsRegressor(n_neighbors=1)
model.fit(xtrain,ytrain)
pred_y = model.predict(xtest)

K1mse2 = mean_squared_error(ytest, pred_y)

# %%
# k1 3

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2)

model = KNeighborsRegressor(n_neighbors=1)
model.fit(xtrain,ytrain)
pred_y = model.predict(xtest)

K1mse3 = mean_squared_error(ytest, pred_y)

# %%
# k1 4

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2)

model = KNeighborsRegressor(n_neighbors=1)
model.fit(xtrain,ytrain)
pred_y = model.predict(xtest)

K1mse4 = mean_squared_error(ytest, pred_y)

# %%
# k1 5

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2)

model = KNeighborsRegressor(n_neighbors=1)
model.fit(xtrain,ytrain)
pred_y = model.predict(xtest)

K1mse5 = mean_squared_error(ytest, pred_y)

# %%
# k1 MSE, cross validating through averages

MSE1 = ((K1mse1+K1mse2+K1mse3+K1mse4+K1mse5)/5)


# %%
# k2 1

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2)

model = KNeighborsRegressor(n_neighbors=2)
model.fit(xtrain,ytrain)
pred_y = model.predict(xtest)

K2mse1 = mean_squared_error(ytest, pred_y)

# %%
# k2 2

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2)

model = KNeighborsRegressor(n_neighbors=2)
model.fit(xtrain,ytrain)
pred_y = model.predict(xtest)

K2mse2 = mean_squared_error(ytest, pred_y)

# %%
# k2 3

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2)

model = KNeighborsRegressor(n_neighbors=2)
model.fit(xtrain,ytrain)
pred_y = model.predict(xtest)

K2mse3 = mean_squared_error(ytest, pred_y)

# %%
# k2 4

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2)

model = KNeighborsRegressor(n_neighbors=2)
model.fit(xtrain,ytrain)
pred_y = model.predict(xtest)

K2mse4 = mean_squared_error(ytest, pred_y)

# %%
# k2 5

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2)

model = KNeighborsRegressor(n_neighbors=2)
model.fit(xtrain,ytrain)
pred_y = model.predict(xtest)

K2mse5 = mean_squared_error(ytest, pred_y)

# %%
#k2 MSE

MSE2 = ((K2mse1+K2mse2+K2mse3+K2mse4+K2mse5)/5)

# %%
# k3 1

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2)

model = KNeighborsRegressor(n_neighbors=3)
model.fit(xtrain,ytrain)
pred_y = model.predict(xtest)

K3mse1 = mean_squared_error(ytest, pred_y)

# %%
# k3 2

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2)

model = KNeighborsRegressor(n_neighbors=3)
model.fit(xtrain,ytrain)
pred_y = model.predict(xtest)

K3mse2 = mean_squared_error(ytest, pred_y)

# %%
# k3 3

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2)

model = KNeighborsRegressor(n_neighbors=3)
model.fit(xtrain,ytrain)
pred_y = model.predict(xtest)

K3mse3 = mean_squared_error(ytest, pred_y)

# %%
# k3 4

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2)

model = KNeighborsRegressor(n_neighbors=3)
model.fit(xtrain,ytrain)
pred_y = model.predict(xtest)

K3mse4 = mean_squared_error(ytest, pred_y)

# %%
# k3 5

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2)

model = KNeighborsRegressor(n_neighbors=3)
model.fit(xtrain,ytrain)
pred_y = model.predict(xtest)

K3mse5 = mean_squared_error(ytest, pred_y)

# %%
#k3 MSE

MSE3 = ((K3mse1+K3mse2+K3mse3+K3mse4+K3mse5)/5)

# %%
# k4 1

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2)

model = KNeighborsRegressor(n_neighbors=4)
model.fit(xtrain,ytrain)
pred_y = model.predict(xtest)

K4mse1 = mean_squared_error(ytest, pred_y)

# %%
# k4 2

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2)

model = KNeighborsRegressor(n_neighbors=4)
model.fit(xtrain,ytrain)
pred_y = model.predict(xtest)

K4mse2 = mean_squared_error(ytest, pred_y)

# %%
# k4 3

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2)

model = KNeighborsRegressor(n_neighbors=4)
model.fit(xtrain,ytrain)
pred_y = model.predict(xtest)

K4mse3 = mean_squared_error(ytest, pred_y)

# %%
# k4 4

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2)

model = KNeighborsRegressor(n_neighbors=4)
model.fit(xtrain,ytrain)
pred_y = model.predict(xtest)

K4mse4 = mean_squared_error(ytest, pred_y)

# %%
# k4 5

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2)

model = KNeighborsRegressor(n_neighbors=4)
model.fit(xtrain,ytrain)
pred_y = model.predict(xtest)

K4mse5 = mean_squared_error(ytest, pred_y)

# %%
# k4 MSE

MSE4 = ((K4mse1+K4mse2+K4mse3+K4mse4+K4mse5)/5)


# %%
# k5 1

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2)

model = KNeighborsRegressor(n_neighbors=5)
model.fit(xtrain,ytrain)
pred_y = model.predict(xtest)

K5mse1 = mean_squared_error(ytest, pred_y)

# %%
# k5 2

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2)

model = KNeighborsRegressor(n_neighbors=5)
model.fit(xtrain,ytrain)
pred_y = model.predict(xtest)

K5mse2 = mean_squared_error(ytest, pred_y)

# %%
# k5 3

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2)

model = KNeighborsRegressor(n_neighbors=5)
model.fit(xtrain,ytrain)
pred_y = model.predict(xtest)

K5mse3 = mean_squared_error(ytest, pred_y)

# %%
# k5 4

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2)

model = KNeighborsRegressor(n_neighbors=5)
model.fit(xtrain,ytrain)
pred_y = model.predict(xtest)

K5mse4 = mean_squared_error(ytest, pred_y)

# %%
# k5 5

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2)

model = KNeighborsRegressor(n_neighbors=5)
model.fit(xtrain,ytrain)
pred_y = model.predict(xtest)

K5mse5 = mean_squared_error(ytest, pred_y)

# %%
# k5 MSE

MSE5 = ((K5mse1+K5mse2+K5mse3+K5mse4+K5mse5)/5)

# %%
# k6 1

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2)

model = KNeighborsRegressor(n_neighbors=6)
model.fit(xtrain,ytrain)
pred_y = model.predict(xtest)

K6mse1 = mean_squared_error(ytest, pred_y)

# %%
# k6 2

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2)

model = KNeighborsRegressor(n_neighbors=6)
model.fit(xtrain,ytrain)
pred_y = model.predict(xtest)

K6mse2 = mean_squared_error(ytest, pred_y)

# %%
# k6 3

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2)

model = KNeighborsRegressor(n_neighbors=6)
model.fit(xtrain,ytrain)
pred_y = model.predict(xtest)

K6mse3 = mean_squared_error(ytest, pred_y)

# %%
# k6 4

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2)

model = KNeighborsRegressor(n_neighbors=6)
model.fit(xtrain,ytrain)
pred_y = model.predict(xtest)

K6mse4 = mean_squared_error(ytest, pred_y)

# %%
# k6 5

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2)

model = KNeighborsRegressor(n_neighbors=6)
model.fit(xtrain,ytrain)
pred_y = model.predict(xtest)

K6mse5 = mean_squared_error(ytest, pred_y)

# %%
# k6 MSE

MSE6 = ((K6mse1+K6mse2+K6mse3+K6mse4+K6mse5)/5)

# %%
#Compare Averaged MSEs to see which is most accurate

MSE1
MSE2
MSE3
MSE4
MSE5
MSE6

#%%
#Ridge
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2)

ridreg = Ridge(normalize = True)

model = ridreg.fit(xtrain, ytrain)

#estimating MSE on test data
ypredict = model.predict(xtest)
ridgemse = (mean_squared_error(ytest,ypredict))

tuning_paramater = 10**np.linspace(10,-2,100)*0.5

ridreg_cv = RidgeCV(alphas = tuning_paramater, scoring = "neg_mean_squared_error", cv =10, normalize = True)
ridreg_cv.fit(xtrain, ytrain)
ridreg_cv.alpha_

#fitting the final model with the right level of alpha
ridreg_tuned = Ridge(alpha = ridreg_cv.alpha_).fit(xtrain,ytrain)
y_pred = ridreg_tuned.predict(xtest)
ridge_mse_tuned = (mean_squared_error(ytest,y_pred))

print(ridgemse)
print(ridge_mse_tuned)
