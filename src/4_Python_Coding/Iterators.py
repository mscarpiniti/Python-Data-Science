# -*- coding: utf-8 -*-
"""
Example of Python iterators.

Created on Wed Jan  4 22:05:07 2023

@author: Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""


# %% ENUMERATE

L = [2, 4, 6, 8]
for i in range(len(L)):
    print(i, L[i])
       
print('\n')
        
for i, val in enumerate(L):
    print(i, val)   
        

# %% ZIP

L = [2, 4, 6, 8, 10]
R = [3, 6, 9, 12, 15]

for lval, rval in zip(L, R):
    print(lval, rval)