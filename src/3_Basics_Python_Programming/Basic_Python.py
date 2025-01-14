# -*- coding: utf-8 -*-
"""
Basics of Python programming: working with variables.

Created on Wed Jan  4 21:31:47 2023

@author: Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""

# %% Scalar variables

a = 3
b = 4.2
C = -1.5

a/2.0

a = 3      # An integer
a = 'pq'   # A string
a = 5.4    # A float

a = 3.1; b = 5.4  # Double assignement in a single line

print(a)


# %% Strings

a = 'hello'
b = 'c' + 'd'
print(b)


# %% List

mylist = [0, 3, 2, 'hi']

newlist = [3, 2, [5, 4, 3], [2, 3, 2]]


newlist[0]
newlist[1]
newlist[2]
newlist[3]

newlist[3][1]
newlist[-1]
newlist[-2]

newlist[2:4]
newlist[0:4:2]
newlist[::-1]

len(newlist)
mylist = [3, 2, 4, 1]
print(*mylist)

mylist.sort()
print(mylist)


# %% Lists are objects

mylist = [3, 2, 4, 1]
alist = mylist
alist[2] = 10
print(mylist)

mylist = [3, 2, 4, 1]
blist = mylist[:]
blist[2] = 10
print(mylist)

mylist = [3, 2, [5, 4, 3], [2, 3, 2]]
clist = mylist[:]
clist[2][1] = 10
print(mylist)


import copy
mylist = [3, 2, [5, 4, 3], [2, 3, 2]]
dlist = copy.deepcopy(mylist)
dlist[2][1] = 10
print(mylist)


# %% Tuples

mytuple = (0, 3, 'Jan', 'Feb', 'March')

print(mytuple)
print(*mytuple)


# %% Dictionaries

months = {'Jan': 31, 'Feb': 28, 'Mar': 31, 'Apr': 30}
months['Feb']

print(months)
print(*months)


m = months.keys()
print(list(m))

days = months.values()
print(list(days))


# %% Sets

primes = {2, 3, 5, 7}
odds = {1, 3, 5, 7, 9}
len(primes)
len(odds)


U = primes | odds   # Union
print(*U)

I = primes & odds   # Intersection
print(*I)

D = primes - odds   # Difference
print(*D)


# Alternatively
U = primes | odds   # Union
print(*U)

I = primes & odds   # Intersection
print(*I)

D = primes - odds   # Difference
print(*D)


# %% Files

input = open('myFile.txt', 'w')

input.writelines('pippo')

input.close()
