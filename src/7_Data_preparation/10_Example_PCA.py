# -*- coding: utf-8 -*-
"""
Example of Principal Component Analysis (PCA) to reduce data dimensionality.

Created on Sat Jan 21 22:21:01 2023

@author: Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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


# Fitting the PCA
pca = PCA(n_components=2)
# pca = PCA(n_components=0.9)
Xtrain_pca = pca.fit_transform(Xtrain_std)
Xtest_pca  = pca.transform(Xtest_std)
print("Number of selected components:", pca.n_components_)
print("Percentage of variance explained:", sum(pca.explained_variance_ratio_))


# Fitting the model
clf = LogisticRegression()
clf.fit(Xtrain_pca, ytrain)

# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(Xtrain_pca, ytrain)


# Making predictions on test set
y_model = clf.predict(Xtest_pca)
# y_model = knn.predict(Xtest_pca)

# Evaluating the trained model
acc = accuracy_score(ytest, y_model)
print("Test accuracy: {}%".format(round(100*acc,2)))

# Check for overfitting
y_model_tr = clf.predict(Xtrain_pca)
# y_model_tr = knn.predict(Xtrain_pca)
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


# %% Plotting the decision regions

from utils import plot_decision_regions


plot_decision_regions(Xtest_pca, ytest, clf)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
