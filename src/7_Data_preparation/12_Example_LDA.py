# -*- coding: utf-8 -*-
"""
Example of Linear Discriminant Analysis (LDA) to reduce data dimensionality.

Created on Sat Jan 21 22:54:14 2023

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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
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


# Fitting the LDA
lda = LDA(n_components=2)
Xtrain_lda = lda.fit_transform(Xtrain_std, ytrain)
Xtest_lda  = lda.transform(Xtest_std)
print("Percentage of variance explained:", sum(lda.explained_variance_ratio_))


# Fitting the model
clf = LogisticRegression()
clf.fit(Xtrain_lda, ytrain)

# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(Xtrain_lda, ytrain)


# Making predictions on test set
y_model = clf.predict(Xtest_lda)
# y_model = knn.predict(Xtest_lda)

# Evaluating the trained model
acc = accuracy_score(ytest, y_model)
print("Test accuracy: {}%".format(round(100*acc,2)))

# Check for overfitting
y_model_tr = clf.predict(Xtrain_lda)
# y_model_tr = knn.predict(Xtrain_lda)
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


plot_decision_regions(Xtest_lda, ytest, clf)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='upper right')
