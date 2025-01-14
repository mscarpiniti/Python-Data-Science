# -*- coding: utf-8 -*-
"""
A further example of constrained minimization using SciPy.

Created on Sat May  2 20:04:49 2020

@author: Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""


import numpy as np
from scipy import optimize as opt


# Function to be minimized
def f(x):
    return x[0]**2 + x[1]**2


# Constraints
def cnst (x):
    return np.atleast_1d(1.0 - 0.25*(x[0] - 2)**2 - (x[1] - 2)**2)



# Solving the minimization problem of Example 1, Slide 27 in Lesson 6
x0 = np.array([0, 0])
sol = opt.minimize(f, x0, constraints={"fun":cnst, "type":"ineq"})

print("Solutions are: ", round(sol.x[0],2), " and ", round(sol.x[1],2), end='\n')
print(" ", end='\n')
print(sol)
