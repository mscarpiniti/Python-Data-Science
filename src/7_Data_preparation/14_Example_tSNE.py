# -*- coding: utf-8 -*-
"""
Example of using the T-distributed Stochastic Neighbor Embedding (t-SNE) 
for reducing the dimensionality of a dataset to 2D, in order to 
make a plot of the resulting embedding.

Created on Sun Jan 22 16:34:11 2023

@author: Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""

# import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import datasets


# Loading the dataset (from sklearn)
digits = datasets.load_digits()
X = digits.data
y = digits.target


# Performing the embeddings
# tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3)
tsne = TSNE(n_components=2)
X_embedded = tsne.fit_transform(X)

print("Dimension of input space:", X.shape[1])
print("Dimension of embedded space:", X_embedded.shape[1])


# Visualizing the embedded space
plt.figure()
sc = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, 
            edgecolor='none', alpha=0.5, cmap='tab10')
plt.xlabel('t1')
plt.ylabel('t2')
plt.title('2-dimensional TSNE projections')
cb = plt.colorbar(sc, label='Class number', ticks=range(10))
# cb.set_ticklabels(np.arange(1,11))
plt.clim(-0.5, 9.5)

