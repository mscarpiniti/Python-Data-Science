# -*- coding: utf-8 -*-
"""
Example of list comprehension.

Created on Wed Jan  4 22:07:19 2023

@author: Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""

L = []
for n in range(12):
    L.append(n ** 2)
print(L)


# %% List comprehension.

L1 = [n ** 2 for n in range(12)]
print(L1)


L2 = [(i, j) for i in range(2) for j in range(3)]
print(L2)


L3 = [val for val in range(20) if val % 3 > 0]
print(L3)


L4 = [val if val % 2 else -val for val in range(20) if val % 3]
print(L4)


# %% Dictionary and set comprehension

D = {n: n**2 for n in range(6)}
print(D)


S1 = {n**2 for n in range(12)}
print(S1)


S2 = {a % 3 for a in range(1000)}
print(S2)


# %% Generators

G = (n**2 for n in range(12))
list(G)

list(G)


G = (n**2 for n in range(12))
for n in G:
    print(n, end=' ')
    if n > 30: break
print("\nDoing something in between ...")
for n in G:
    print(n, end=' ')

