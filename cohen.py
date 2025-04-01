#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.signal import hilbert
from scipy.signal import convolve2d


def int_autocorr(x):
    # Function to compute the instantenous autocorrelation
    # John T. Semmlow, Biosignal and biomedical image processing:
    # MATLAB-based applications, Part 1
    # Output
    #    Rx instantaneous autocorrelation
    #  Input
    #    x signal

    N = x.size
    Rx = np.zeros((N, N), dtype='complex')  # % Initialize output
    # %
    # % Compute instantenous autocorrelation
    for ti in range(N - 1):  # 1:N % Increment over time
        taumax = min([ti, N - ti - 1, int(round(N / 2.) - 1)])
        tau = np.arange(-taumax, taumax + 1)
        Rx[tau - tau[0], ti] = x[ti + tau] * np.conj(x[ti - tau])

    return Rx


def cohen(x, fs, type_='WV'):
    """
    Function to compute several of Cohen's class of
    time-frequency distributions
    John T. Semmlow, Biosignal and biomedical image processing:
    MATLAB-based applications, Part 1

    Outputs
         CD Selected distribution
         f Frequency vector for plotting
         t Time vector for plotting
    Inputs
         x  Complex signal
         fs Sampling frequency
         type of distribution. Valid arguments are:
         'choi' (Choi-Williams), 'BJC' (Born-Jorden-Cohen);
         and 'R_M' (Rihaczek-Margenau) Default is Wigner-Ville
    """

    # Assing constants and check input
    # sigma = 1                    # Choi-Williams constant
    L = 30                       # Size of determining function

    N = x.size
    # Calculate time and frequency
    t = np.arange(0, N).astype(float) / fs
    f = np.arange(0, N).astype(float) * (fs / (2 * N))  # vectors for plotting

    # Compute instantenous autocorrelation: Eq. (7)
    CD = int_autocorr(x)

    # if type_[1] == 'c':            # Get appropriate determining
    #                                # function
    #     G = choi(sigma,L)          # Choi-Williams
    # elif type_[1] == 'B':
    #     G = BJC(L);                # Born-Jorden-Cohen
    # elif type_[1] == 'R':
    #     G = R_M(L);                # Rihaczek-Margenau
    # else:
    G = np.zeros((L, L), dtype=int)  # Default Wigner-Ville
    G[L // 2 - 1, L // 2 - 1] = 1

    #  Convolve determining function with instantenous
    #  autocorrelation
    CD = convolve2d(CD, G, "same")             # 2-D convolution

    return np.fft.fft(CD, axis=0), f, t