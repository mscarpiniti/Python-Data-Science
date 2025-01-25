# -*- coding: utf-8 -*-
"""
Example of recursive feature elimination (RFE).

Created on Sat Jan 21 20:45:19 2023

@author: Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""

import numpy as np
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn import datasets


# Loading the dataset (from sklearn)
wine = datasets.load_wine()
X = wine.data
y = wine.target


# Splitting in training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.33, random_state=42)


# Standardize all features
sc = StandardScaler()
Xtrain_std = sc.fit_transform(Xtrain)
Xtest_std  = sc.fit_transform(Xtest)


# %% Using the RFE method

# Fitting the feature selector
NF = 5  # Number of features to select
model = LogisticRegression()
selector = RFE(model, n_features_to_select=NF)
selector.fit(Xtrain_std, ytrain)


# Select only the selected features
X_train_select = selector.transform(Xtrain_std)
X_test_select  = selector.transform(Xtest_std)


# Fitting the model
model.fit(X_train_select, ytrain)


# Making predictions on test set
y_model = model.predict(X_test_select)

# Evaluating the trained model
acc = accuracy_score(ytest, y_model)
print("Test accuracy: {}%".format(round(100*acc,2)))

# Check for overfitting
y_model_tr = model.predict(X_train_select)
acc_tr = accuracy_score(ytrain, y_model_tr)
print("Train accuracy: {}%".format(round(100*acc_tr,2)))


print(" ", end='\n')
print("Complete report: ", end='\n')
print(classification_report(ytest, y_model))
print(" ", end='\n')

# Evaluating and showing the Confusion Matrix (CM)
cm = confusion_matrix(ytest, y_model)
class_names = np.unique(y)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues')


# Print selected features
print(selector.get_feature_names_out(wine.feature_names))



# %% Using the RFECV method

from sklearn.feature_selection import RFECV

# Fitting the feature selector
NF = 3  # Minimu number of features to select
model = LogisticRegression()
selector = RFECV(model, min_features_to_select=NF, cv=5)
selector.fit(Xtrain_std, ytrain)


# Select only the selected features
X_train_select = selector.transform(Xtrain_std)
X_test_select  = selector.transform(Xtest_std)


# Fitting the model
model.fit(X_train_select, ytrain)


# Making predictions on test set
y_model = model.predict(X_test_select)

# Evaluating the trained model
acc = accuracy_score(ytest, y_model)
print("Test accuracy: {}%".format(round(100*acc,2)))

# Check for overfitting
y_model_tr = model.predict(X_train_select)
acc_tr = accuracy_score(ytrain, y_model_tr)
print("Train accuracy: {}%".format(round(100*acc_tr,2)))


print(" ", end='\n')
print("Complete report: ", end='\n')
print(classification_report(ytest, y_model))
print(" ", end='\n')

# Evaluating and showing the Confusion Matrix (CM)
cm = confusion_matrix(ytest, y_model)
class_names = np.unique(y)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues')


# Print selected features
print(selector.get_feature_names_out(wine.feature_names))
