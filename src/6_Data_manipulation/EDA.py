# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 17:18:50 2025

@author: miche
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


path  = 'C:/Notebooks/PyDS/data/'
file1 = path + 'iris.csv'
file2 = path + 'titanic.csv'

df1 = pd.read_csv(file1, header=None)
df2 = pd.read_csv(file2)

df1.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
df2['Pclass'] = df2.Pclass.astype(str)



# %% Histogram

fig, ax = plt.subplots(figsize =(12,8))
ax = sns.histplot(df2[df2['Survived']==1].Fare, label='Survived')
ax = sns.histplot(df2[df2['Survived']==0].Fare, label='Died')
ax.legend()



# %% Scatterplot

plt.figure()
sns.scatterplot(x='Age', y='Fare', data=df2)



# %% Pairplot

sns.pairplot(df1)

sns.pairplot(df1, hue='species')



# %% Jointplot

sns.jointplot(data=df1, x="sepal_length", y="sepal_width", hue="species")



# %% Boxplot

plt.figure()
sns.boxplot(data=df1)

plt.figure()
sns.boxplot(x=df2["Age"])

plt.figure()
sns.boxplot(data=df2, x="Age", y="Pclass")

plt.figure()
sns.boxplot(data=df2, x="Age", y="Pclass", hue="Survived")



# %% Violin plot

plt.figure()
sns.violinplot(data=df2, x="Age", y="Sex")

plt.figure()
sns.violinplot(data=df2, x="Age", y="Sex", hue="Survived")



# %% Heatmap

import numpy as np

X = df1.iloc[:,:4].values
cm = np.corrcoef(X.T)   # From NumPy

plt.figure()
sns.heatmap(cm, annot=True, fmt=".2f")



# %% RadViz plot

from pandas.plotting import radviz

X = df2.copy()
X = df2[['Survived','Pclass','Age','SibSp','Parch','Fare']]
X['Pclass'] = X.Pclass.astype(int)

plt.figure()
radviz(X, 'Survived')



# %% Parallel coordinates plot

from pandas.plotting import parallel_coordinates

plt.figure()
parallel_coordinates(X[:10], "Survived")

