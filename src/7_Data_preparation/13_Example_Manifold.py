# -*- coding: utf-8 -*-
"""
Example of using the manifold learning techniques (MDS, ISOMAP, LLE, SE)  
for reducing the dimensionality of a dataset to 2D, in order to 
make a plot of the resulting embedding.

Created on Sat Jan 25 16:04:56 2025

@author: Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""

# import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


# Loading the dataset (from sklearn)
digits = datasets.load_digits()
X = digits.data
y = digits.target



# %% Performing the MDS embeddings

from sklearn.manifold import MDS

mds = MDS(n_components=2, normalized_stress='auto')

X_mds = mds.fit_transform(X[:300])

print("Dimension of input space:", X.shape[1])
print("Dimension of embedded space:", X_mds.shape[1])


# Visualizing the embedded space
plt.figure()
sc = plt.scatter(X_mds[:, 0], X_mds[:, 1], c=y[:300], 
            edgecolor='none', alpha=0.5, cmap='tab10')
plt.xlabel('c1')
plt.ylabel('c2')
plt.title('2-dimensional MDS projections')
cb = plt.colorbar(sc, label='Class number', ticks=range(10))
# cb.set_ticklabels(np.arange(1,11))
plt.clim(-0.5, 9.5)



# %% Performing the ISOMAP embeddings

from sklearn.manifold import Isomap

isomap = Isomap(n_components=2)

X_isomap = isomap.fit_transform(X[:300])

print("Dimension of input space:", X.shape[1])
print("Dimension of embedded space:", X_isomap.shape[1])


# Visualizing the embedded space
plt.figure()
sc = plt.scatter(X_isomap[:, 0], X_isomap[:, 1], c=y[:300], 
            edgecolor='none', alpha=0.5, cmap='tab10')
plt.xlabel('c1')
plt.ylabel('c2')
plt.title('2-dimensional ISOMAP projections')
cb = plt.colorbar(sc, label='Class number', ticks=range(10))
# cb.set_ticklabels(np.arange(1,11))
plt.clim(-0.5, 9.5)



# %% Performing the Locally linear embedding (LLE) embeddings

from sklearn.manifold import LocallyLinearEmbedding

lle = LocallyLinearEmbedding(n_components=2)

X_lle = lle.fit_transform(X[:300])

print("Dimension of input space:", X.shape[1])
print("Dimension of embedded space:", X_lle.shape[1])


# Visualizing the embedded space
plt.figure()
sc = plt.scatter(X_lle[:, 0], X_lle[:, 1], c=y[:300], 
            edgecolor='none', alpha=0.5, cmap='tab10')
plt.xlabel('c1')
plt.ylabel('c2')
plt.title('2-dimensional LLE projections')
cb = plt.colorbar(sc, label='Class number', ticks=range(10))
# cb.set_ticklabels(np.arange(1,11))
plt.clim(-0.5, 9.5)



# %% Performing the Spectral Embedding (SE) embeddings

from sklearn.manifold import SpectralEmbedding

se = SpectralEmbedding(n_components=2)

X_se = se.fit_transform(X[:300])

print("Dimension of input space:", X.shape[1])
print("Dimension of embedded space:", X_se.shape[1])


# Visualizing the embedded space
plt.figure()
sc = plt.scatter(X_se[:, 0], X_se[:, 1], c=y[:300], 
            edgecolor='none', alpha=0.5, cmap='tab10')
plt.xlabel('c1')
plt.ylabel('c2')
plt.title('2-dimensional Spectral projections')
cb = plt.colorbar(sc, label='Class number', ticks=range(10))
# cb.set_ticklabels(np.arange(1,11))
plt.clim(-0.5, 9.5)



# %% Whole comparison

fig, ax = plt.subplots(2, 2, figsize=(6 ,6))
ax[0, 0].set_title("MDS")
ax[0, 0].scatter(X_mds[:, 0], X_mds[:, 1], c=y[:300], 
            edgecolor='none', alpha=0.5, cmap='tab10')
ax[0, 1].set_title ("ISOMAP")
ax[0, 1].scatter(X_isomap[:, 0], X_isomap[:, 1], c=y[:300], 
            edgecolor='none', alpha=0.5, cmap='tab10')
ax[1, 0]. set_title ("LLE")
ax[1, 0].scatter(X_lle[:, 0], X_lle[:, 1], c=y[:300], 
            edgecolor='none', alpha=0.5, cmap='tab10')
ax[1, 1]. set_title ("Spectral")
ax[1, 1].scatter(X_se[:, 0], X_se[:, 1], c=y[:300], 
            edgecolor='none', alpha=0.5, cmap='tab10')

ax[1, 1]. set_xlabel ("c1")
ax[1, 0]. set_xlabel ("c1")
ax[0, 0]. set_ylabel ("c2")
ax[1, 0]. set_ylabel ("c2")
