# -*- coding: utf-8 -*-
"""
Example of L1 regularization for feature selection.

Created on Sat Jan 21 17:29:31 2023

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
from sklearn import datasets


# Loading the dataset (from sklearn)
wine = datasets.load_wine()
X = wine.data
y = wine.target


# Splitting in training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.33)


# Standardize all features
sc = StandardScaler()
Xtrain_std = sc.fit_transform(Xtrain)
Xtest_std  = sc.fit_transform(Xtest)


# Fitting the model
model = LogisticRegression(penalty='l1', C=0.1, solver='liblinear')
model.fit(Xtrain_std, ytrain)

# Making predictions on test set
y_model = model.predict(Xtest_std)

# Evaluating the trained model
acc = accuracy_score(ytest, y_model)
print("Test accuracy: {}%".format(round(100*acc,2)))

# Check for overfitting
y_model_tr = model.predict(Xtrain_std)
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

# Print model coefficients
print(model.coef_[0,:])  # only for first class

# Print selected feature names
print(" ", end='\n')
print('Selected features:')
for i in range(X.shape[1]):
    if model.coef_[0,i] != 0.0:
        print("{}: {}".format(i+1,wine.feature_names[i]))

