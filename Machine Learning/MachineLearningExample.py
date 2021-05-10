# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 15:36:03 2021

@author: Adam
"""

#%%
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import mglearn
from sklearn import preprocessing
import statsmodels.formula.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from xgboost import plot_importance
#%%
df= pd.read_csv(r'C:\Users\Adam\Desktop\HRemployee_attrition.csv')
#%%
le = preprocessing.LabelEncoder()

Attrition_encoded = le.fit_transform(df["Attrition"])
df['Attrition_encoded']=Attrition_encoded
del df['Attrition']

Travel_encoded = le.fit_transform(df["BusinessTravel"])
df['Travel_encoded']=Travel_encoded
del df['BusinessTravel']

Dept_encoded=le.fit_transform(df['Department'])
df['Dept_encoded']=Dept_encoded
del df['Department']

Edutype_encoded=le.fit_transform(df['EducationField'])
df['Edutype_encoded']=Edutype_encoded
del df['EducationField']

Gen_encoded=le.fit_transform(df['Gender'])
df['Gen_encoded']=Gen_encoded
del df['Gender']

Marital_encoded=le.fit_transform(df['MaritalStatus'])
df['Marital_encoded']=Marital_encoded
del df['MaritalStatus']

del df['Over18']

OT_=le.fit_transform(df['OverTime'])
df['OT_']=OT_
del df['OverTime']

Role_encoded=le.fit_transform(df['JobRole'])
df['Role_encoded']=Role_encoded
del df['JobRole']
#%% #split the data
target_column = ['Attrition_encoded'] 
predictors = list(set(list(df.columns))-set(target_column))
df[predictors] = df[predictors]/df[predictors].max()
df.describe().transpose()
x = df[predictors].values
y = df[target_column].values
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.20, random_state=40)
mlp = MLPClassifier(hidden_layer_sizes=(10, 5), activation='relu', solver='adam', max_iter=500, random_state = 40)
mlp.fit(xtrain,ytrain.ravel())
ypred = mlp.predict(xtest)
print("Accuracy:", round(accuracy_score(ytest, ypred), 3))
#%% #heatmap
mat = confusion_matrix(ypred, ytest)
names = np.unique(ytest)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=names, yticklabels=names)
plt.xlabel('Truth')
plt.ylabel('Predicted')
plt.show()
#%% #neural
mlp = MLPClassifier(hidden_layer_sizes=(10, 5), activation='relu', solver='adam', max_iter=500, random_state = 40, alpha=0.001)
mlp.fit(xtrain,ytrain.ravel())
ypred = mlp.predict(xtest)
print("Accuracy:", round(accuracy_score(ytest, ypred), 3))
#%% #random forest
df.head()
y = df.loc[:,'Attrition_encoded']
x = df.drop('Attrition_encoded', axis = 1)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=1)


regressor100 = RandomForestClassifier(n_estimators=100, random_state=40)  
regressor1000 = RandomForestClassifier(n_estimators=1000,random_state=40)  

regressor100.fit(xtrain, ytrain)
regressor1000.fit(xtrain, ytrain)
ypred100 = regressor100.predict(xtest)
ypred1000 = regressor1000.predict(xtest)

print('Root Mean Squared Error w 100 trees:', np.sqrt(metrics.accuracy_score(ytest, ypred100)))
print('Root Mean Squared Error w 1000 trees:', np.sqrt(metrics.accuracy_score(ytest, ypred1000)))

#%% 
model = XGBClassifier(n_estimators=1000, random_state=40, learning_rate=0.01)
model.fit(xtrain, ytrain)
ypredbxg = model.predict(xtest)
#Enter in employee factors where x_test is located.
print(ypredbxg)
print('Root Mean Squared Error:', np.sqrt(metrics.accuracy_score(ytest, ypredbxg)))
#%%
#%%
#creating the function for feature importance
def plot_feature_importance(importance,names,model_type):

    feature_importance = np.array(importance)
    feature_names = np.array(names)

    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)

    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

    plt.figure(figsize=(10,8))

    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])

    plt.title(model_type + 'FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
#%% #plot it
plot_feature_importance(model.feature_importances_,xtrain.columns,'XG BOOST')
#%%
