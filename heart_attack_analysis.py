# -*- coding: utf-8 -*-
"""
Name:               Karina Jonina 
Github:             https://github.com/kjonina/
Linkedin:           https://www.linkedin.com/in/karinajonina/    
Tableau Public:     https://public.tableau.com/profile/karina.jonina#!/ 
Data Gathered:	    https://www.kaggle.com/andrewmvd/heart-failure-clinical-data

Task Details
Create interactive graphs using Plotly
Create a model to assess the likelihood of a death by heart failure event.
This can be used to help hospitals in assessing the severity of patients with cardiovascular diseases.
"""


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE  # imblearn library can be installed using pip install imblearn
from sklearn.ensemble import RandomForestClassifier


# read the CSV file
df = pd.read_csv('heart_failure_clinical_records_dataset.csv')

# Will ensure that all columns are displayed
pd.set_option('display.max_columns', None) 

# prints out the top 5 values for the datasef
print(df.head())

# checking shape
print("The dataset has {} rows and {} columns.".format(*df.shape))
# The dataset has 299 rows and 13 columns.

# ... and duplicates
print("It contains {} duplicates.".format(df.duplicated().sum()))
#It contains 0 duplicates.

# prints out names of columns
print(df.columns)



# This tells us which variables are object, int64 and float 64. 
print(df.info())

# checking for missing data
df.isnull().sum() 

# getting the statistics such as the mean, standard deviation, min and max for numerical variables
print(df.describe()) 

# =============================================================================
# Changing variables dtypes
# =============================================================================
df['diabetes'] = df['diabetes'].astype('category')
df['anaemia'] = df['anaemia'].astype('category')
df['sex'] = df['sex'].astype('category')
df['high_blood_pressure'] = df['high_blood_pressure'].astype('category')
df['smoking'] = df['smoking'].astype('category')
df['DEATH_EVENT'] = df['DEATH_EVENT'].astype('category')



# This tells us which variables are object, int64 and float 64. 
print(df.info())

# =============================================================================
# Checking out Outlier in Price
# =============================================================================
sns.set()


plt.figure(figsize = (12, 8))
spotting_outliers = plt.scatter(x= 'creatinine_phosphokinase', y = 'ejection_fraction', data = df) 
plt.title('Checking for outliers',  fontsize = 20)
plt.ylabel('ejection_fraction', fontsize = 14)
plt.xlabel('creatinine_phosphokinase', fontsize = 14)
plt.show()

#Save the graph
spotting_outliers.figure.savefig('spotting_outliers.png')



# =============================================================================
# Examining EDA
# =============================================================================
sns.set()

#examining the data in Councils
print(df.groupby(['diabetes']).size().sort_values(ascending=False))
#0    174
#1    125

# create a graph
plt.figure(figsize = (12, 8))
diabetes_graph = sns.countplot(x = 'diabetes', data = df, palette = 'terrain',order = df['diabetes'].value_counts().index)
diabetes_graph.set_title('Breakdown of Patients with Diabetes', fontsize = 20)
diabetes_graph.set_ylabel('Number of Patients', fontsize = 14)
diabetes_graph.set_xlabel('diabetes', fontsize = 14)
plt.show()

#Save the graph
diabetes_graph.figure.savefig('diabetes_graph.png')


# examining data in Parishes
print(df.groupby(['anaemia']).size().sort_values(ascending=False))

# create a graph for Parishes
plt.figure(figsize = (12, 8))
anaemia_graph = sns.countplot(x = 'anaemia', data = df, palette = 'terrain',order = df['anaemia'].value_counts().head(20).index)
anaemia_graph.set_title('Breakdown of Patients with Anaemia', fontsize = 20)
anaemia_graph.set_ylabel('Number of Patients', fontsize = 14)
anaemia_graph.set_xlabel('Anaemia', fontsize = 14)
plt.show()

##Save the graph
#anaemia_graph.figure.savefig('anaemia_graph.png')


# examining data in Parishes
print(df.groupby(['sex']).size().sort_values(ascending=False))
#1    194
#0    105

#creating a graph of counties
plt.figure(figsize = (12, 8))
sex_graph = sns.countplot(x = 'sex', data = df, palette = 'magma',order = df['sex'].value_counts().index)
sex_graph.set_title('Number of Women and Men', fontsize = 20)
sex_graph.set_ylabel('Number of Patients', fontsize = 14)
sex_graph.set_xlabel('sex', fontsize = 14)
plt.show()

##Save the graph
#sex_graph.figure.savefig('sex_graph.png')


# counting numbers of Room Types
df.groupby(['high_blood_pressure']).size().sort_values(ascending=False)
#0    194
#1    105


#Plot the graph of High Blood Pressure
plt.figure(figsize = (12, 8))
HB_graph = sns.countplot(x= 'high_blood_pressure', data = df, palette = 'terrain',order = df['high_blood_pressure'].value_counts().index)
HB_graph.set_title('Number of Patients with High Blood Pressure', fontsize = 20)
HB_graph.set_xlabel('High Blood Pressure', fontsize = 14)
HB_graph.set_ylabel('Number of Patients', fontsize = 14)
plt.show()

##Save the graph
#HB_graph.figure.savefig('HB_graph.png')




# counting numbers of Room Types
df.groupby(['high_blood_pressure']).size().sort_values(ascending=False)
#0    194
#1    105


#Plot the graph of Smokers
plt.figure(figsize = (12, 8))
smoking_graph = sns.countplot(x= 'smoking', data = df, palette = 'terrain',order = df['smoking'].value_counts().index)
smoking_graph.set_title('Number of Patients who Smoke', fontsize = 20)
smoking_graph.set_xlabel('Smoking', fontsize = 14)
smoking_graph.set_ylabel('Number of Patients', fontsize = 14)
plt.show()

#Save the graph
smoking_graph.figure.savefig('smoking_graph.png')





# counting numbers of Room Types
df.groupby(['DEATH_EVENT']).size().sort_values(ascending=False)
#0    203
#1     96

#Plot the graph of Room Type
plt.figure(figsize = (12, 8))
death_graph = sns.countplot(x= 'DEATH_EVENT', data = df, palette = 'terrain',order = df['DEATH_EVENT'].value_counts().index)
death_graph.set_title('Number of Patients who Died', fontsize = 20)
death_graph.set_xlabel('Death Event', fontsize = 14)
death_graph.set_ylabel('Number of Patients', fontsize = 14)
plt.show()

#Save the graph
death_graph .figure.savefig('death_graph .png')

# =============================================================================
# Looking at EDA
# =============================================================================
# prints out names of columns
print(df.info())

#Data columns (total 13 columns):
# #   Column                    Non-Null Count  Dtype   
#---  ------                    --------------  -----   
# 0   age                       299 non-null    float64 
# 1   anaemia                   299 non-null    category
# 2   creatinine_phosphokinase  299 non-null    int64   
# 3   diabetes                  299 non-null    category
# 4   ejection_fraction         299 non-null    int64   
# 5   high_blood_pressure       299 non-null    category
# 6   platelets                 299 non-null    float64 
# 7   serum_creatinine          299 non-null    float64 
# 8   serum_sodium              299 non-null    int64   
# 9   sex                       299 non-null    category
# 10  smoking                   299 non-null    category
# 11  time                      299 non-null    int64   
# 12  DEATH_EVENT               299 non-null    category


plt.figure(figsize = (12, 8))
age_graph  = df['age'].hist(bins=10)
age_graph.set_title('Breakdown of Patient\'s Age', fontsize = 20)
age_graph.set_xlabel('Age', fontsize = 14)
age_graph.set_ylabel('Number of Patients', fontsize = 14)
plt.show()

#Save the graph
age_graph.figure.savefig('age_graph .png')





plt.figure(figsize = (12, 8))
cp_graph  = df['creatinine_phosphokinase'].hist(bins=10)
cp_graph.set_title('Breakdown of Patient\'s creatinine_phosphokinase', fontsize = 20)
cp_graph.set_xlabel('creatinine_phosphokinase', fontsize = 14)
cp_graph.set_ylabel('Number of Patients', fontsize = 14)
plt.show()

#Save the graph
cp_graph.figure.savefig('cp_graph .png')



plt.figure(figsize = (12, 8))
ef_graph  = df['ejection_fraction'].hist(bins=10)
ef_graph.set_title('Breakdown of Patient\'s ef_graph', fontsize = 20)
ef_graph.set_xlabel('ef_graph', fontsize = 14)
ef_graph.set_ylabel('Number of Patients', fontsize = 14)
plt.show()

#Save the graph
ef_graph.figure.savefig('ef_graph .png')



plt.figure(figsize = (12, 8))
time_graph  = df['time'].hist(bins=10)
time_graph.set_title('Breakdown of Patient\'s Follow Up Time', fontsize = 20)
time_graph.set_xlabel('Follow-Up Time', fontsize = 14)
time_graph.set_ylabel('Number of Patients', fontsize = 14)
plt.show()

#Save the graph
time_graph.figure.savefig('time_graph .png')


plt.figure(figsize = (12, 8))
ss_graph  = df['serum_sodium'].hist(bins=10)
ss_graph.set_title('Breakdown of Patient\'s serum_sodium', fontsize = 20)
ss_graph.set_xlabel('serum_sodium', fontsize = 14)
ss_graph.set_ylabel('Number of Patients', fontsize = 14)
plt.show()






plt.figure(figsize = (12, 8))
plat_graph  = df['platelets'].hist(bins=10)
plat_graph.set_title('Breakdown of Patient\'s platelets', fontsize = 20)
plat_graph.set_xlabel('platelets', fontsize = 14)
plat_graph.set_ylabel('Number of Patients', fontsize = 14)
plt.show()

#Save the graph
plat_graph.figure.savefig('plat_graph .png')






#Save the graph
ss_graph.figure.savefig('ss_graph .png')




plt.figure(figsize = (12, 8))
ss_graph  = df['serum_sodium'].hist(bins=10)
ss_graph.set_title('Breakdown of Patient\'s serum_sodium', fontsize = 20)
ss_graph.set_xlabel('serum_sodium', fontsize = 14)
ss_graph.set_ylabel('Number of Patients', fontsize = 14)
plt.show()

#Save the graph
ss_graph.figure.savefig('ss_graph .png')


# =============================================================================
# 
# =============================================================================

# creating a new dataset with Room Type and 
age_death = pd.DataFrame({'age': df['age'],
                   'DEATH_EVENT': df['DEATH_EVENT']})


age_death.groupby('DEATH_EVENT')['age'].describe()
#             count       mean        std   min   25%   50%   75%   max
#DEATH_EVENT                                                           
#0            203.0  58.761906  10.637890  40.0  50.0  60.0  65.0  90.0
#1             96.0  65.215281  13.214556  42.0  55.0  65.0  75.0  95.0 

 

# boxplot for  price for the room 
plt.figure(figsize = (12, 8))
age_death_boxplot = sns.boxplot(x = 'DEATH_EVENT', y = 'age',
            data = age_death, fliersize = 0)
age_death_boxplot.set_title('Breakdown of Death by Age', fontsize = 20)
age_death_boxplot.set_ylabel('Patients\' Age', fontsize = 14)
age_death_boxplot.set_xlabel('Death Event', fontsize = 14)
plt.show()


# creating a new dataset with Room Type and 
age_death_avaibility = pd.DataFrame({'age': df['age'],
                   'DEATH_EVENT': df['DEATH_EVENT']})


age_death_avaibility.groupby('DEATH_EVENT')['age'].describe()
#                   count        mean         std   min   25%    50%     75%        max
#room_type                                                                     
#Entire home/apt  15349.0  138.737703  104.924566  11.0  80.0  110.0  155.00    1000.0  
#Hotel room         320.0   96.550000   67.870132   0.0  62.0   82.0  107.25     500.0    
#Private room     11119.0   66.937494   57.303045  12.0  40.0   55.0   75.00    1000.0    
#Shared room        239.0   56.108787  109.101972  10.0  20.0   30.0   52.50     999.0    

 

# boxplot for  price for the room 
plt.figure(figsize = (12, 8))
room_price_boxplot = sns.boxplot(x = 'DEATH_EVENT', y = 'age',
            data = age_death_avaibility, fliersize = 0)
room_price_boxplot.set_title('Breakdown of Death by Age', fontsize = 20)
room_price_boxplot.set_ylabel('Number of Patients', fontsize = 14)
room_price_boxplot.set_xlabel('Death Event', fontsize = 14)
plt.show()


# =============================================================================
# Splitting the data into training and test data and OVERSAMPLING
# =============================================================================

X = df.drop('DEATH_EVENT', axis = 1) 
Y = df['DEATH_EVENT'] # Labels
print(type(X))
print(type(Y))
print(X.shape) # so we now rows shown and have 22 independent variables (14 + additional var when month split - the target var = 22)
print(Y.shape) # total rows and a single column

## Normalizing numerical features so that each feature has mean 0 and variance 1
feature_scaler = StandardScaler()

#this is now a normalised feature set
X_scaled = feature_scaler.fit_transform(X) 

## Dividing dataset into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split( X_scaled, Y, test_size = 0.3, random_state = 100)

#70 % chosen random set
print(X_train.shape)

# 30 % randomly set test set
print(X_test.shape) 

## Implementing Oversampling to balance the dataset; SMOTE stands for Synthetic Minority Oversampling TEchnique
print("Number of observations in each class before oversampling (training data): \n", pd.Series(Y_train).value_counts())

smote = SMOTE(random_state = 101)
X_train,Y_train = smote.fit_sample(X_train,Y_train)

print("Number of observations in each class after oversampling (training data): \n", pd.Series(Y_train).value_counts())
## ABOVE RETURNS BALANCE DATA SET 144 EACH

# =============================================================================
# Decision Tree
# =============================================================================

dtree = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 5) 
## ENTROPY MEANS USING INFORMATION GAME TO SPLIT VARS .  Max depth only going to 5 levels, otherwise get overfit
## we don't know what is the optimal level - so experiment, we start with 5
dtree.fit(X_train, Y_train)
featimp = pd.Series(dtree.feature_importances_, index=list(X)).sort_values(ascending=False)
## tells us which features are most important in predicting which web visits result in a transaction
## var with values 0 means they weren't use to construct the decision trees, because we went to level 5

print(featimp) # won't give a decision tree pictorial - 
#
## Evaluating Decision Tree Model by constructing a decision tree matrix  
Y_pred = dtree.predict(X_test)
print("Prediction Accuracy: ", metrics.accuracy_score(Y_test, Y_pred)) 
conf_mat = metrics.confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(conf_mat,annot=True)
plt.title("Confusion_matrix")
plt.xlabel("Predicted Class")
plt.show()
print('Confusion matrix: \n', conf_mat)
print('TP: ', conf_mat[1,1])
print('TN: ', conf_mat[0,0])
print('FP: ', conf_mat[0,1])
print('FN: ', conf_mat[1,0])


