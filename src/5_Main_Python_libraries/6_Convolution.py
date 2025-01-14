# -*- coding: utf-8 -*-
"""
Computing two simple convolution between two rectangles.

Created on Mon Jan  2 23:20:17 2023

@author: Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


# Creating three rectangles
x1 = np.ones((100,1))
x2 = np.ones((100,1))
x3 = np.ones((140,1))

# Computing the convolutions
y1 = signal.convolve(x1, x2)
y2 = signal.convolve(x1, x3)

# Plotting the first convolution
plt.figure()
plt.plot(y1)
plt.grid()
plt.title('Convolution between rectangles of same length')
plt.xlabel('Index')
plt.ylabel('Amplitude')

# Plotting the second convolution
plt.figure()
plt.plot(y2)
plt.grid()
plt.title('Convolution between rectangles of different length')
plt.xlabel('Index')
plt.ylabel('Amplitude')
