# -*- coding: utf-8 -*-
"""
Simple examples of using the Matplotlib library.

Created on Mon Jan  2 18:15:06 2023

@author: Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""

import matplotlib.pyplot as plt
import numpy as np


# %% Example of plotting a Gaussian curve

gaussian = lambda x: np.exp(-(0.5-x)**2/1.5)

x = np.arange(-2,2.5,0.01)
y = gaussian(x)

plt.ion()
plt.figure()
plt.plot(x,y)
plt.grid()
plt.xlabel('x values')
plt.ylabel('exp(-(0.5-x)**2/1.5')
plt.title('Gaussian Function')
#plt.savefig('Gaussian.png')
plt.show()


# %% Example of changing the line properties

x = np.arange(0,2*np.pi,0.01)
y1 = np.sin(x)
y2 = np.sin(2*x)
y3 = np.sin(3*x)
y4 = np.sin(4*x)

plt.figure()
plt.plot(x,y1, color='black', lw=1,   ls='-',  label='sin(x)')
plt.plot(x,y2, color='red',   lw=1.5, ls='--', label='sin(2x)')
plt.plot(x,y3, color='blue',  lw=2,   ls=':',  label='sin(3x)')
plt.plot(x,y4, color='green',lw=0.8, marker='+', label='sin(4x)')
plt.grid()
plt.xlabel('x')
plt.ylabel('Values')
plt.title('Sin Function')
plt.legend(loc='upper right')


# %% Example of using axes

x = np.linspace(0, 50, 500)
y = np.sin(x) * np.exp(-x/10)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, y, lw=2)
ax.set_xlim(0, 45)
ax.set_ylim(-1, 1)
ax.set_xlabel ('x', labelpad=5, fontsize=18, fontname='serif', color='blue')
ax.set_ylabel ('f(x)', labelpad=15, fontsize=18, fontname='serif', color='blue')
ax.set_title('axis labels and title example', fontsize=16, fontname='serif', color='blue')


# %% Example of sub-plots

x = np.arange(0,2*np.pi,0.01)
y1 = np.sin(x)
y2 = np.sin(2*x)
y3 = np.sin(3*x)
y4 = np.sin(4*x)

fig, ax = plt.subplots(2, 2, figsize=(6, 6))

ax[0, 0].set_title("Sin(x)")
ax[0, 0].plot(x, y1)

ax[0, 1].set_title("Sin(2x)")
ax[0, 1].plot(x, y2)

ax[1, 0].set_title("Sin(3x)")
ax[1, 0].plot(x, y3)

ax[1, 1].set_title("Sin(4x)")
ax[1, 1].plot(x, y4)

ax[1, 1].set_xlabel("x")
ax[1, 0].set_xlabel("x")
ax[0, 0].set_ylabel("y")
ax[1, 0].set_ylabel("y")
