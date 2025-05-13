# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 18:41:29 2024

@author: MADHU SAGAR
"""
import pip
pip.main(["install","pandas"])
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
df1 = pd.read_csv('datas2.csv')
df1['Who completed the test_NM'] = df1['Who completed the test_NM'].str.lower()
df1
sns.set_style('whitegrid')
ax = sns.countplot(data=df1, x='Class/ASD Traits ')

plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Count Plot of Class Column')

# Annotating each bar with its respective count value
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.show()
# Columns to be label encoded
columns_encode = ['Sex','Ethnicity','Jaundice','Family_mem_with_ASD','Who completed the test_NM','Class/ASD Traits ']

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Loop through each column and perform label encoding
for col in columns_encode:
    df1[col] = label_encoder.fit_transform(df1[col])

# The DataFrame 'df' now contains the label-encoded values for the specified columns
df1
x=df1.drop('Case_No',axis=1)
x=x.drop('Class/ASD Traits ',axis=1)
#x=x.drop('Qchat-25-Score',axis=1)
x=x.drop('Sex',axis=1)
x=x.drop('Who completed the test_NM',axis=1)

y=df1['Class/ASD Traits ']
print(x)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=40)
# Create the XGBoost classifier
clf = xgb.XGBClassifier()

# Train the classifier on the training data
clf.fit(x_train, y_train)
#prediction


import pickle
pickle.dump(clf,open('autism.pkl','wb'))