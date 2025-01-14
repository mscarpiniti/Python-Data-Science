# -*- coding: utf-8 -*-
"""
Example of control flow in Python.

Created on Wed Jan  4 21:58:13 2023

@author: Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""


# %% IF statement

a = 200
b = 33

if b > a:
    print("b is greater than a")
elif a == b:
    print("a and b are equal")
else:
    print("a is greater than b")
    
    
# %% FOR loop

S = 0

for k in range(101):
    S = S+k
print(S)


# %% WHILE loop

S = 0
k = 1

while k < 101:
    S = S+k
    k = k+1
print(S)


# %% BREAK command

S = 0
k = 1

while 1:
    S = S+k
    k = k+1
    if k > 100:
        print('End!')
        break
print(S)


# %% CONTINUE command

S = 0
k = 1

while k < 101:
    if k == 50:
        k = k+1
        continue
    S = S+k
    k = k+1

print(S)

