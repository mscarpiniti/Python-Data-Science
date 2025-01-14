# -*- coding: utf-8 -*-
"""


Created on Wed Jan  4 22:20:37 2023

@author: Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""

# %% LAMBDA functions

f = lambda x : pow(x,3)+7 
f(7)


add = lambda x, y: x + y
add(1, 2)


# %% MAP command

mylist = [3, 2, 4, 1]
newlist = map(lambda x:pow(x,3)+7, mylist)
print(*newlist)


# %% FILTER command

mylist = [3, 2, 4, 1]
newlist = filter(lambda x:x>=3, mylist)
print(*newlist)

