# -*- coding: utf-8 -*-
"""
Smoothing a noisy signal with a simple low-pass filter.

Created on Mon Jan  2 23:16:15 2023

@author: Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


# Function for the noisy signal generation
def signal_samples(t, f1, f2):
    return (2 * np.sin(2 * np.pi * f1 * t) + 3 * np.sin(2 * np.pi * f2 * t) + 0.5 * np.random.randn(*np.shape(t)))


# Settings
Fs = 60; T = 1
N = Fs*T
f1 = 5; f2 = 8

# Generating the noisy signal
t = np.linspace(0, T, N)
x = signal_samples(t, f1, f2)

# Creating the LP filter
M = 4
h = np.ones((M,))/M
y = signal.lfilter(h, 1, x)  # Filtering

# Plotting the original and smoothed signals
plt.figure()
plt.plot(t, x, label='Original')
plt.plot(t, y, color='red', label='Smoothed')
plt.grid()
plt.title('Smoothing of a signal')
plt.xlabel('t')
plt.ylabel('Amplitude')
plt.legend()
