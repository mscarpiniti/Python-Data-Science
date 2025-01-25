# -*- coding: utf-8 -*-
"""
Example of nonlinear Kernel Principal Component Analysis (KPCA) 
to reduce data dimensionality.

Created on Sun Jan 22 14:50:12 2023

@author: Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""

# import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_moons, make_circles


# %% MOON DATASET

# Generating the Moon dataset
X, y = make_moons(n_samples=500, noise=0.0, random_state=42)

# Evaluating PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Evaluating KPCA
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
X_kpca = kpca.fit_transform(X)


# Plotting the original dataset
plt.figure()
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', marker='^', alpha=0.5)
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='o', alpha=0.5)
plt.title('Original Moon dataset')
plt.xlabel('x1')
plt.ylabel('x2')
plt.tight_layout()


# Plotting the projected dataset with PCA
plt.figure()
plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1],
            color='red', marker='^', alpha=0.5)
plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1],
            color='blue', marker='o', alpha=0.5)
plt.title('Projected Moon dataset with PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.tight_layout()


# Plotting the projected dataset with KPCA
plt.figure()
plt.scatter(X_kpca[y == 0, 0], X_kpca[y == 0, 1],
            color='red', marker='^', alpha=0.5)
plt.scatter(X_kpca[y == 1, 0], X_kpca[y == 1, 1],
            color='blue', marker='o', alpha=0.5)
plt.title('Projected Moon dataset with KPCA')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.tight_layout()


# Make a single figure
# fig, (orig_data_ax, pca_proj_ax, kernel_pca_proj_ax) = plt.subplots(
#     ncols=3, figsize=(14, 4))

# orig_data_ax.scatter(X[:, 0], X[:, 1], c=y)
# orig_data_ax.set_ylabel("x2")
# orig_data_ax.set_xlabel("x1")
# orig_data_ax.set_title("Original Moon dataset")

# pca_proj_ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
# pca_proj_ax.set_ylabel("PC2")
# pca_proj_ax.set_xlabel("PC1")
# pca_proj_ax.set_title("Projected Moon dataset with PCA")

# kernel_pca_proj_ax.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y)
# kernel_pca_proj_ax.set_ylabel("PC2")
# kernel_pca_proj_ax.set_xlabel("PC1")
# _ = kernel_pca_proj_ax.set_title("Projected Moon dataset with KPCA")

# plt.tight_layout()



# %% CIRCLE DATASET

# Generating the Circle dataset
X, y = make_circles(n_samples=500, factor=0.3, noise=0.05, random_state=42)

# Evaluating PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Evaluating KPCA
kpca = KernelPCA(n_components=None, kernel='rbf', gamma=10)
X_kpca = kpca.fit_transform(X)


# Plotting the original dataset
plt.figure()
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', marker='^', alpha=0.5)
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='o', alpha=0.5)
plt.title('Original Circle dataset')
plt.xlabel('x1')
plt.ylabel('x2')
plt.tight_layout()


# Plotting the projected dataset with PCA
plt.figure()
plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1],
            color='red', marker='^', alpha=0.5)
plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1],
            color='blue', marker='o', alpha=0.5)
plt.title('Projected Circle dataset with PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.tight_layout()


# Plotting the projected dataset with KPCA
plt.figure()
plt.scatter(X_kpca[y == 0, 0], X_kpca[y == 0, 1],
            color='red', marker='^', alpha=0.5)
plt.scatter(X_kpca[y == 1, 0], X_kpca[y == 1, 1],
            color='blue', marker='o', alpha=0.5)
plt.title('Projected Circle dataset with KPCA')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.tight_layout()


# Make a single figure
# fig, (orig_data_ax, pca_proj_ax, kernel_pca_proj_ax) = plt.subplots(
#     ncols=3, figsize=(14, 4))

# orig_data_ax.scatter(X[:, 0], X[:, 1], c=y)
# orig_data_ax.set_ylabel("x2")
# orig_data_ax.set_xlabel("x1")
# orig_data_ax.set_title("Original Circle dataset")

# pca_proj_ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
# pca_proj_ax.set_ylabel("PC2")
# pca_proj_ax.set_xlabel("PC1")
# pca_proj_ax.set_title("Projected Circle dataset with PCA")

# kernel_pca_proj_ax.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y)
# kernel_pca_proj_ax.set_ylabel("PC2")
# kernel_pca_proj_ax.set_xlabel("PC1")
# _ = kernel_pca_proj_ax.set_title("Projected Circle dataset with KPCA")

# plt.tight_layout()

