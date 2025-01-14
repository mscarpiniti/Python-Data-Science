# -*- coding: utf-8 -*-
"""
Example of unconstrained and constrained minimization using SciPy.

Created on Sat May  2 22:07:35 2020

@author: Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""

import numpy as np
from scipy import optimize as opt

##### UNCONSTRAINED OPTIMIZATION #####

# Defining the function to be minimized
def f1(x):
    return -np.exp(-(x - 0.7)**2)


# Set the initial Guess and minimize
x1_0 = 0.0
sol1 = opt.minimize(f1, x1_0)
print(sol1)
print(" ", end='\n\n')




##### CONSTRAINED OPTIMIZATION #####

# Defining the function to be minimized
def f2(x):
    return np.sqrt((x[0] - 3)**2 + (x[1] - 2)**2)

# Defining the constrains function
def cnst(x):
    return np.atleast_1d(1.5 - np.sum(np.abs(x)))


# Set the initial Guess and minimize
x2_0 = np.array([0, 0])
sol2 = opt.minimize(f2, x2_0, constraints={"fun": cnst, "type": "ineq"})
print(sol2)
