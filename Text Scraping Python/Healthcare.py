# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 16:10:19 2021

@author: Adam
"""

# %%

#import packages

import pandas as pd
import recordlinkage

# %%

#import file

df = pd.read_excel (r'/Users/Adam/Desktop/Scraping/HEALTH FINAL.xlsx')
print (df)

# %%

# insert comp ID variable into df

df.insert(0, 'Comp_ID', range(1, 1 + len(df)))

#duplicate for comparison

df_a = pd.DataFrame(df)
df_b = pd.DataFrame(df)

# %%

# initialize compare comand (by company ID, so we can use name as string)

indexer = recordlinkage.Index()
indexer.full()
candidate_links = indexer.index(df_a)
c = recordlinkage.Compare()

# %%

c.string('Executives', 'Executives', method='jarowinkler', threshold=0.95,label='Executives')

c.string('Company', 'Company', method='jarowinkler', threshold=0.85,label='Company')

c.string('Corporate Family', 'Corporate Family', method='jarowinkler', threshold=0.85, label='Corporate Family')

c.exact('Phone Number', 'Phone Number',label='Phone Number')
# %%

features = c.compute(candidate_links, df_a, df_b)
features['totallinks']= features['Executives'] + features['Company'] + features['Corporate Family'] +features['Phone Number']


print(features)
#%%
featuresab = features[features['totallinks']==2]
featuresab = featuresab[featuresab['Executives'] == 1] 
featuresab = featuresab[featuresab['Company'] == 1]


featuresac = features[features['totallinks']==2]
featuresac = featuresac[featuresac['Executives'] == 1] 
featuresac = featuresac[featuresac['Corporate Family'] == 1]

featuresad = features[features['totallinks']==2]
featuresad = featuresad[featuresad['Executives'] == 1] 
featuresad = featuresad[featuresad['Phone Number'] == 1]

featuresbc = features[features['totallinks']==2]
featuresbc = featuresbc[featuresbc['Company'] == 1] 
featuresbc = featuresbc[featuresbc['Corporate Family'] == 1]

featuresbd = features[features['totallinks']==2]
featuresbd = featuresbd[featuresbd['Company'] == 1] 
featuresbd = featuresbd[featuresbd['Phone Number'] == 1]

featurescd = features[features['totallinks']==2]
featurescd = featurescd[featurescd['Corporate Family'] == 1] 
featurescd = featurescd[featurescd['Phone Number'] == 1]

features.to_excel(r'/Users/Adam/Desktop/features.xlsx', sheet_name='features', index = False)
featuresab.to_excel(r'/Users/Adam/Desktop/featuresab.xlsx', sheet_name='featuresab', index = False)
featuresac.to_excel(r'/Users/Adam/Desktop/featuresac.xlsx', sheet_name='featuresac', index = False)
featuresad.to_excel(r'/Users/Adam/Desktop/featuresad.xlsx', sheet_name='featuresad', index = False)
featuresbc.to_excel(r'/Users/Adam/Desktop/featuresbc.xlsx', sheet_name='featuresbc', index = False)
featuresbd.to_excel(r'/Users/Adam/Desktop/featuresbd.xlsx', sheet_name='featuresbd', index = False)
featurescd.to_excel(r'/Users/Adam/Desktop/featurescd.xlsx', sheet_name='featurescd', index = False)

#%%
featuresabc3 = features[features['totallinks']==3]
featuresabc3 = featuresabc3[featuresabc3['Executives'] == 1] 
featuresabc3 = featuresabc3[featuresabc3['Company'] == 1]
featuresabc3 = featuresabc3[featuresabc3['Corporate Family'] == 1]

featuresabd3 = features[features['totallinks']==3]
featuresabd3 = featuresabd3[featuresabd3['Executives'] == 1] 
featuresabd3 = featuresabd3[featuresabd3['Company'] == 1]
featuresabd3 = featuresabd3[featuresabd3['Phone Number'] == 1]

featuresbcd3 = features[features['totallinks']==3]
featuresbcd3 = featuresbcd3[featuresbcd3['Company'] == 1] 
featuresbcd3 = featuresbcd3[featuresbcd3['Corporate Family'] == 1]
featuresbcd3 = featuresbcd3[featuresbcd3['Phone Number'] == 1]

featuresabcd = features[features['totallinks']==4]

featuresabc3.to_excel(r'/Users/Adam/Desktop/featuresabc.xlsx', sheet_name='featuresabc', index = False)
featuresabd3.to_excel(r'/Users/Adam/Desktop/featuresabd.xlsx', sheet_name='featuresabd', index = False)
featuresbcd3.to_excel(r'/Users/Adam/Desktop/featuresbcd.xlsx', sheet_name='featuresbcd', index = False)
featuresabcd.to_excel(r'/Users/Adam/Desktop/featuresabcd.xlsx', sheet_name='featuresabcd', index = False)

# %%

ecm = recordlinkage.ECMClassifier()

matchdf = ecm.fit_predict(features)
matchdffinal = matchdf.to_frame(index=False)

matchdffinal.to_excel(r'/Users/Adam/Desktop/HEALTHCAREMATCHES.xlsx', sheet_name='HEALTHCAREMATCHES', index = False)
