# -*- coding: utf-8 -*-
"""
Example of using pre-processing in Scikit-learn.

Created on Sun Jan 22 17:56:20 2023

@author: Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""

# %% Spliting data into a training and test sets

from sklearn.model_selection import train_test_split
from sklearn import datasets

# Loading the dataset (from sklearn)
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y)

print("Shape of training set:", X_train.shape[0])
print("Shape of test set:", X_test.shape[0])


# %% Normalization

import numpy as np

ex = np.array([0, 1, 2, 3, 4, 5])

# Manual normalization
ex_norm = (ex - ex.min())/(ex.max() - ex.min())
print(ex_norm)

# Scikit-learn normalization
from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()
mms.fit(ex[:,np.newaxis])
ex_norm = mms.transform(ex[:,np.newaxis])
# Or use directly
# ex_norm = mms.fit_transform(ex[:,np.newaxis])
print(ex_norm)


# Applying the MMS to the Iris dataset
X_train_norm = mms.fit_transform(X_train)
X_test_norm  = mms.transform(X_test)


# %% Standardization

ex = np.array([0, 1, 2, 3, 4, 5])

# Manual standardization
ex_std= (ex - ex.mean())/ex.std()
print(ex_std)

# Scikit-learn standardization
from sklearn.preprocessing import StandardScaler

stdsc = StandardScaler()
stdsc.fit(ex[:,np.newaxis])
ex_std = stdsc.transform(ex[:,np.newaxis])
# Or use directly
# ex_std = stdsc.fit_transform(ex[:,np.newaxis])
print(ex_std)


# Applying the MMS to the Iris dataset
X_train_std = stdsc.fit_transform(X_train)
X_test_std  = stdsc.transform(X_test)


# %% Outlier reduction

from sklearn.preprocessing import RobustScaler

rsc = RobustScaler()
X_train_norm = rsc.fit_transform(X_train)
X_test_norm  = rsc.transform(X_test)


# Or changing the quantile
rsc = RobustScaler(quantile_range=(10.0, 90.0))
X_train_norm = rsc.fit_transform(X_train)
X_test_norm  = rsc.transform(X_test)

