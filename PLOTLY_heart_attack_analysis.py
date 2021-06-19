"""
Name:               Karina Jonina 
Github:             https://github.com/kjonina/
Linkedin:           https://www.linkedin.com/in/karinajonina/    
Tableau Public:     https://public.tableau.com/profile/karina.jonina#!/ 
Data Gathered:	    https://www.kaggle.com/andrewmvd/heart-failure-clinical-data

Task Details
Create interactive graphs using Plotly
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import datetime


# Downloading plotly packages
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

pio.renderers.default='browser'
#pio.renderers.default='svg'


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
# 
# =============================================================================
df['diabetes'] = df['diabetes'].replace(0,'normal')
df['diabetes'] = df['diabetes'].replace(1,'diabetes')


df['DEATH_EVENT'] = df['DEATH_EVENT'].replace(0,'alive')
df['DEATH_EVENT'] = df['DEATH_EVENT'].replace(1,'death')

# =============================================================================
# Continuous Variables
# =============================================================================

#renaming of the columns
df_cont_varialbes = pd.DataFrame({'age': df['age'],
                   'creatinine_phosphokinase': df['creatinine_phosphokinase'],
                   'ejection_fraction': df['ejection_fraction'],
                   'platelets': df['platelets'],
                   'serum_creatinine': df['serum_creatinine'],
                   'serum_sodium': df['serum_sodium'],
                   'time': df['time']})


# creating a histogram and a boxplot for Creatinine Phosphokinase
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    subplot_titles=[
                    'Histogram of Creatinine Phosphokinase',
                    'Box plot of Creatinine Phosphokinase'])
fig.update_layout({'title': {'text': 'Plots of Creatinine Phosphokinase',
                             'x': 0.5, 'y': 0.9}})
fig.add_trace( 
        go.Histogram(x=df['creatinine_phosphokinase'], nbinsx=20, name = 'Histogram'),
                     row=1, col=1)
fig.add_trace(go.Box(x=df['creatinine_phosphokinase'], name = 'Box plot'),row=2, col=1)

fig.write_html("creatinine_phosphokinase.html")



# =============================================================================
# Creating loop the graph
# =============================================================================

df_cont_varialbes.head()

variable_list = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets',
             'serum_creatinine', 'serum_sodium','time']



for stock in stocklist:
    df = yf.download(stock, start, end)
    filename = stock.lower() + '.png'
    mpf.plot(df,type='candlestick',figratio=(16,6), 
             mav=(short_sma,long_sma), 
             volume=True, title=stock,style='default',
             savefig=dict(fname=filename,bbox_inches="tight"))


for variable in variable_list:
    filename = variable.lower() +'_boxplot'
    print(filename)

# my first for loop
for variable in variable_list:
    df1 = df_cont_varialbes
    filename = variable.lower() +'_boxplot'
    fig = px.box(data_frame = df1, x= df1[variable], hover_name = df1[variable])
    fig.write_html(str(filename) + ".html")
    fig.show()

for variables in variable_list:
    df1 = df_cont_varialbes
    fig = px.histogram(data_frame = df1, x= df1[variable], nbins=20, hover_name = df1[variable])
    fig.show()


# More complex for loop
fig = make_subplots(rows=2, cols=1, shared_xaxes=True) 
for variable in variable_list:
    df1 = df_cont_varialbes
    fig.add_trace(go.Histogram(x = df1[variable], nbinsx=20), row=1, col=1)
    fig.add_trace(go.Box(x = df1[variable]), row=2, col=1)
    fig.show()


# =============================================================================
#  Checking Correlations
# =============================================================================

cr = df_cont_varialbes.corr(method='pearson')
print(cr)

fig = go.Figure(go.Heatmap(x=cr.columns, 
                           y=cr.columns,
                           z=cr.values.tolist(),
                           colorscale='rdylgn', 
                           zmin=-1, zmax=1))
fig.show()


# =============================================================================
# 
# =============================================================================

month_district_ac = df.groupby(['diabetes','DEATH_EVENT']).size().to_frame('size').reset_index()
print(month_district_ac)

################################################
# Create the base figure
fig = go.Figure()

# Loop through the species
for diabete in ['normal', 'diabetes']:
  # Add a bar chart trace
  fig.add_trace(go.Bar(x = month_district_ac['DEATH_EVENT'], 
                       y = month_district_ac[month_district_ac.diabetes == diabete]['size'], name = diabete))
# Show the figure
fig.show()


###############################################

fig = go.Figure()

# Loop through the species
for diabete in ['normal', 'diabetes']:
  # Add a bar chart trace
  fig.add_trace(go.Box(x = df['DEATH_EVENT'], 
                       y = df[df.diabetes == diabete]['creatinine_phosphokinase'], name = diabete))
# Show the figure
fig.show()

fig = go.Figure(go.Heatmap(x=cr.columns, 
                           y=cr.columns,
                           z=cr.values.tolist(),
                           colorscale='rdylgn', 
                           zmin=-1, zmax=1))
fig.show()
