# -*- coding: utf-8 -*-
"""
Evaluating the spectral analysis of a noisy signal.

Created on Mon Jan  2 23:25:18 2023

@author: Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""

import numpy as np
from scipy import fft
import matplotlib.pyplot as plt


# Function for the noisy signal generation
def signal_samples(t, f1, f2):
    return (2 * np.sin(2 * np.pi * f1 * t) + 3 * np.sin(2 * np.pi * f2 * t) + 2 * np.random.randn(*np.shape(t)))


# Settings
Fs = 60; T = 4
N = Fs*T
f1 = 5; f2 = 22

# Generate the time-domain signal
t = np.linspace(0, T, N)
x = signal_samples(t, f1, f2)

# Compute the FFT
X = fft.fft(x)

# Generating the f-axis and mask of positive frequencies
f = fft.fftfreq(N, 1.0/Fs)
mask = np.where(f >= 0)


# Plotting
fig, ax = plt.subplots(2, 1, figsize=(12, 10))
# Plot of the time-domain signal
ax[0].plot(t, x, label="Time")
ax[0].grid()
ax[0].set_title("Time domain signal")
ax[0].set_xlabel("$t$", fontsize=14)
ax[0].set_ylabel("$x(t)$", fontsize=14)
ax[0].legend()

# Plot of the frequency-domain Amplitude spectrum
ax[1].plot(f[mask], abs(X[mask])/N, label="Amplitude")
ax[1].grid()
ax[1].set_title("Frequency domain signal")
ax[1].set_xlabel("$f$", fontsize=14)
ax[1].set_ylabel("$|X(f)|/N$", fontsize=14)
ax[1].legend()
