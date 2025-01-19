# -*- coding: utf-8 -*-
"""
Example of using Pandas to load datasets.

Created on Sat Jan  7 12:38:31 2023

@author: Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""

import pandas as pd


# %% Reading the dataset

data = pd.read_csv('iris.csv', header=None)


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
data = pd.read_csv(url, header=None)


X = data.iloc[:, :4].values
y = data.iloc[:, 4].values



# %% Using Seaborn

import seaborn as sns

iris = sns.load_dataset('iris')
iris.head()

X = iris.drop('species', axis=1)
y = iris['species']


sns.set()
sns.pairplot(iris, hue='species', height=1.5)



# %% Using Scikit-learn

from sklearn import datasets

iris = datasets.load_iris()

X = iris.data[:, :4]
y = iris.target

