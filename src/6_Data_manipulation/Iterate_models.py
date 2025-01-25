# -*- coding: utf-8 -*-
"""
Simple example of iterating over different models and packing results.

Created on Sat Jan 18 22:45:01 2025

@author: Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""

# import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



# Loading the dataset (from sklearn)
iris = datasets.load_iris()
X = iris.data[:, :4]  # we only take the first two features.
y = iris.target


# Split in train and test sets
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size= 0.33, random_state= 4)


# Set the models and related parameters
model1 = KNeighborsClassifier(n_neighbors=3)
model2 = LogisticRegression()
model3 = DecisionTreeClassifier(max_depth=5)
model4 = GaussianNB()


names = ["kNN", "LR",  "DT", "NB"]

# Set the parameters of the chosen classifiers
models = [model1, model2, model3, model4]



# %% First version: simple print of accuracy

accuracy = []

for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracy.append(acc)
    
print(accuracy)


# %% Second version: saving several metrics to an excel file

results = {'Model': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-score': []}

for name, model in zip(names, models):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results['Model'].append(name)
    results['Accuracy'].append(acc)
    results['Precision'].append(pre)
    results['Recall'].append(rec)
    results['F1-score'].append(f1)
    
    print("%4s: %0.2f\n" % (name, round(100*acc, 2)))
    

df = pd.DataFrame(results)
df.to_excel("Results.xlsx", index=False)
