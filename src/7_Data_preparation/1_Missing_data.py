# -*- coding: utf-8 -*-
"""
Example of handling missing data in Scikit-learn.

Created on Sun Jan 22 17:36:45 2023

@author: Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""

import numpy as np


# %% The simple imputer

from sklearn.impute import SimpleImputer

X = [[1, 2, np.nan], [3, 4, 3], [np.nan, 6, 5], [8, 8, 7]]

imp_mean = SimpleImputer(strategy='mean')
X_impm = imp_mean.fit_transform(X)
print(X_impm)


# %% The kNN imputer

from sklearn.impute import KNNImputer

X = [[1, 2, np.nan], [3, 4, 3], [np.nan, 6, 5], [8, 8, 7]]

imp_knn = KNNImputer(n_neighbors=2)
X_impk = imp_knn.fit_transform(X)
print(X_impk)



