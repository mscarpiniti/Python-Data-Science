# -*- coding: utf-8 -*-
"""
Example of sequential feature selection.

Created on Sat Jan 21 17:59:16 2023

@author: Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""

import numpy as np
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector
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


# Fitting the feature selector
NF = 3  # Number of features to select
direction = 'backward'
# direction = 'forward'
knn = KNeighborsClassifier(n_neighbors=5)
sfs = SequentialFeatureSelector(knn, n_features_to_select=NF, direction=direction)
sfs.fit(Xtrain_std, ytrain)


# Select only the selected features
X_train_select = sfs.transform(Xtrain_std)
X_test_select  = sfs.transform(Xtest_std)


# Fitting the model
knn.fit(X_train_select, ytrain)


# Making predictions on test set
y_model = knn.predict(X_test_select)

# Evaluating the trained model
acc = accuracy_score(ytest, y_model)
print("Test accuracy: {}%".format(round(100*acc,2)))

# Check for overfitting
y_model_tr = knn.predict(X_train_select)
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
print(sfs.get_feature_names_out(wine.feature_names))
