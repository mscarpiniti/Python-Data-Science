# -*- coding: utf-8 -*-
"""
Simple examples of using the NumPy library.

Created on Tue Jan  3 16:06:38 2023

@author: Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""

import numpy as np


# %% Creating a 1D and a 2D arrays
myarray = np.array([4, 3, 2])
mybigarray = np.array([[3, 2, 4], [3, 3, 2], [4, 5, 2]])

print(myarray)
print(*myarray)
print(mybigarray)


# %% More sophisticated examples

a = np.arange(5)
print(a)

b = np.arange(3,7,2)
print(b)

c = np.ones((3,4))
print(c)

d = np.eye(3)
print(d)

e = np.eye(3,4)
print(e)

f = np.linspace(3,7,3)
print(f)

g = np.r_[1:4,0,4]
print(g)

h = np.r_[2,1:7:3j]
print(h)


# %% Reshaping and manipulating arrays

a = np.arange(6).reshape(3,2)
print(a)

print(np.ndim(a))

print(np.size(a))

print(np.shape(a))

print(np.ravel(a))

print(np.transpose(a))

print(a[::-1])

print(np.sum(a))

print(np.sum(a,axis=0))

print(np.sum(a,axis=1))


# %% Short-cuts

print(a.min())
print(a.max())
print(a.T)


# %% Matrix algebra

a = np.arange(6).reshape(3,2)
b = np.arange(3,9).reshape(3,2)
c = np.transpose(b)

print(a+b)

print(a-b)

print(a*b)

print(np.dot(a,c))

print(np.power(a,2))

print(np.power(2,a))


# %% Find elements under logical conditions

a = np.arange(6).reshape(3,2)

x = np.where(a>2, 0, 1)
print(x)

indices = np.where((a[:,0]>3) | (a[:,1]<3))
print(indices)


# %% Adding and removing dimensions

a_1D = np.array([2, 3, 1, 5])
print(a_1D.shape)

a_2D = a_1D[:, np.newaxis]
print(a_2D.shape)

b_1D = np.squeeze(a_2D)
print(b_1D.shape)


# %% Broadcasting

a = np.arange(3) + 5
print(a)

b = np.ones((3,3)) + np.arange(3)
print(b)

c = np.arange(3).reshape((3,1)) + np.arange(3)
print(c)


# %% Random number and linear algebra

a = np.random.uniform(3,6,5)
print(a)

b = np.random.randn(3,4)
print(b)

c = np.random.randint(5,9,(3,3))
print(c)

print(np.linalg.det(c))

d = np.linalg.inv(c)
print(d)

e, f = np.linalg.eig(c)
print(e)
print(f)


# %% Things to be aware of

a = np.ones((3,3))
print(a)
print(np.shape(a))

b = a[:,1]
print(np.shape(b))

c = a[1,:]
print(np.shape(c))

print(c)
print(c.T)


# Solution #1
c = a[0:1,:]
print(np.shape(c))

# Solution #2
c = a[0,:].reshape(1,len(a))
print(np.shape(c))











