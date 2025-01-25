# -*- coding: utf-8 -*-
"""
Examples of data encoding in Scikit-learn.

Created on Sun Jan 22 17:40:21 2023

@author: Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""


# %% Label encoding

from sklearn.preprocessing import LabelEncoder

y = ['M', 'F', 'F', 'M', 'F', 'M', 'M']

le = LabelEncoder()
y_enc = le.fit_transform(y)


list(le.classes_)
list(le.inverse_transform([0, 0, 1, 0, 1]))


# %% Handling ordinal feature

from sklearn.preprocessing import OrdinalEncoder

X = [['M'], ['S'], ['XL'], ['L'], ['S'], ['S'], ['M'], ['L'], ['XL'], ['L']]

oe = OrdinalEncoder()
X_enc = oe.fit_transform(X)
list(oe.categories_)
list(oe.inverse_transform([[2], [0], [1], [0], [3]]))


# %% Handling ordinal feature with Pandas

import pandas as pd

df = pd.DataFrame([['green', 'M', 10.1, 'class2'],
                   ['red', 'L', 13.5, 'class1'],
                   ['blue', 'XL', 15.3, 'class2'],
  				   ['red', 'S', 8.4, 'class1']])

df.columns = ['color', 'size', 'price', 'classlabel']

size_mapping = {'XL': 4, 'L': 3, 'M': 2, 'S': 1}

df['size'] = df['size'].map(size_mapping)
df


# %% One-hot encoding

from sklearn.preprocessing import OneHotEncoder

X = [['green'], ['red'], ['green'], ['blue']]

ohe = OneHotEncoder()
X_enc = ohe.fit_transform(X).toarray()
print(X_enc)


# %% One-hot encoding with Pandas

df = pd.get_dummies(df[['price', 'color', 'size']])
df


# %% Binary encoding

from sklearn.preprocessing import LabelBinarizer

x = [1, 2, 6, 4, 2]
lb = LabelBinarizer()
y = lb.fit_transform(x)
print(y)


from sklearn.preprocessing import MultiLabelBinarizer

X = [[1, 2], [6, 1], [4, 2]]
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(X)
print(Y)


# %% Binning

import numpy as np
from sklearn.preprocessing import Binarizer

x = np.array([1.0, 0.5, 4.2, 5.6, 2.2, 3.5, 1.8, 6.8, 2.5, 3.1])

bi = Binarizer(threshold=3.0)
y  = bi.fit_transform(x.reshape(-1, 1))
print(y)


from sklearn.preprocessing import KBinsDiscretizer

kbd = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
ybd = kbd.fit_transform(x.reshape(-1, 1))
print(ybd)
