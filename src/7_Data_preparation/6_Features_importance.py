# -*- coding: utf-8 -*-
"""
Example of measuring the features importance with Random Forests.

Created on Sat Jan 21 18:42:19 2023

@author: Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
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


# Fitting the model
clf = RandomForestClassifier()
clf.fit(Xtrain_std, ytrain)


# Making predictions on test set
y_model = clf.predict(Xtest_std)

# Evaluating the trained model
acc = accuracy_score(ytest, y_model)
print("Test accuracy: {}%".format(round(100*acc,2)))

# Check for overfitting
y_model_tr = clf.predict(Xtrain_std)
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


# Print and plot the features importance
importance = clf.feature_importances_
indices = np.argsort(importance)[::-1]
feat_labels = wine.feature_names
print(importance)
for f in range(X.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importance[indices[f]]))

feat_labels = [feat_labels[i] for i in indices]

plt.figure()
plt.bar(range(X.shape[1]), importance[indices], align='center')
plt.title('Feature Importance')
plt.xticks(range(X.shape[1]), feat_labels, rotation=90)
plt.xlim([-1, X.shape[1]])
plt.tight_layout()


# %% Select the feature from the used model

from sklearn.feature_selection import SelectFromModel

sfm = SelectFromModel(clf, threshold=0.1, prefit=True)
X_select = sfm.transform(Xtrain_std)
print('Number of features that meet this threshold criterion:', X_select.shape[1])
