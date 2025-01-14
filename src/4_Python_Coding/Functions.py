# -*- coding: utf-8 -*-
"""
Example of defining and using functions

Created on Wed Jan  4 22:14:53 2023

@author: Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""

def pythagoras(a=3, b=4):
    """ Computes the hypotenuse of two arguments """
    c = pow(a**2+b**2, 0.5)   # pow(x,0.5) is the square root
    return c



help(pythagoras)

l1 = pythagoras(2,1)
print(l1)

cat1 = 5
cat2 = 6
l2 = pythagoras(cat1, cat2)
print(l2)

l3 = pythagoras()
print(l3)

l4 = pythagoras(cat1)
print(l4)

l5 = pythagoras(b=cat2, a=cat1)
print(l5)


# %% Generator function

def gen():
    for n in range(12):
		    yield n ** 2
				
G = gen()
print(*G)

