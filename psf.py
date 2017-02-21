#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
psf.py
------

'''

from __future__ import division, print_function, absolute_import, unicode_literals
import numpy as np
from scipy.special import erf
from scipy.misc import comb
import matplotlib.pyplot as pl

def IndefiniteExpIntegral(z, n):
  '''
  Integrates the function `z^n * exp(-z^2)` for a given integer value of `n`.

  '''
  
  if n == 0:
    return 0.5 * np.sqrt(np.pi) * erf(z)
  elif n == 1:
    return -0.5 * np.exp(-z ** 2)
  elif n == 2:
    return 0.25 * (np.sqrt(np.pi) * erf(z) - 2 * z * np.exp(-z ** 2))
  elif n == 3:
    return -0.5 * np.exp(-z ** 2) * (z ** 2 + 1)
  elif n == 4:
    return 3. / 8. * np.sqrt(np.pi) * erf(z) - 0.25 * np.exp(-z ** 2) * z * (2 * z ** 2 + 3)
  elif n == 5:
    return -0.5 * np.exp(-z ** 2) * (z ** 4 + 2 * z ** 2 + 2)
  else:
    raise ValueError("Invalid value for `n`.")
    
def Gaussian(x, x0, sigma):
  '''
  
  '''
  
  return 1. / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - x0) / sigma) ** 2)

def Polynomial(x, coeffs):
  '''
  
  '''
  
  return np.sum([c * x ** m for m, c in enumerate(coeffs)], axis = 0)

def ExpIntegral(n, x0, sigma):
  '''
  
  '''
  
  EI = IndefiniteExpIntegral((1 - x0) / (np.sqrt(2) * sigma), n) - IndefiniteExpIntegral(-x0 / (np.sqrt(2) * sigma), n)
  return 1. / np.sqrt(np.pi) * EI

def PolynomialBasisChange(coeffs, a, b):
  '''
  Performs the transformation f(x) --> f(z) where
  z = ax + b. `coeffs` are the polynomial coefficients
  in increasing order. Returns the coefficients in the
  new basis. Based on http://stackoverflow.com/a/142436.
  
  '''
  
  if b == 0:
    return np.array([c * a ** -n for n, c in enumerate(coeffs)])
  else:
    T = np.zeros((len(coeffs), len(coeffs)))
    for i in range(len(coeffs)):
      for j in range(len(coeffs)):
        T[i][j] = comb(j, i, exact = True) * a ** i * b ** (j - i)
    return np.linalg.solve(T, coeffs) 

def PRF(x0, sigma, amp, ipv):
  '''
  
  '''
  
  # Ensure inputs are arrays
  x0 = np.atleast_1d(x0)
  sigma = np.atleast_1d(sigma)
  amp = np.atleast_1d(amp)
  
  # Sum over the gaussians
  flux = 0.
  for x, s, a in zip(x0, sigma, amp):
  
    # Stretch the coefficients to a new basis, shifted by `x0` and scaled by `sigma`
    coeffs = PolynomialBasisChange(ipv, 1. / (np.sqrt(2) * s), -x / (np.sqrt(2) * s))

    # Integrate the polynomial times the gaussian
    flux += a * np.sum([c * ExpIntegral(n, x, s) for n, c in enumerate(coeffs) if c != 0.])
    
  return flux

def PlotPixel(x0 = 0.25, sigma = 0.5, amp = 1., ipv = [3 / 4., 1, -1]):
  '''
  PlotPixel(x0 = [0.2, 0.5, 0.8], sigma = [0.05, 0.2, 0.05], amp = [0.03, 0.5, 0.03], ipv = [0.991071, 0.5, -7.5, 14, -7])
  
  '''
  
  # Ensure inputs are arrays
  x0 = np.atleast_1d(x0)
  sigma = np.atleast_1d(sigma)
  amp = np.atleast_1d(amp)
  
  # The various curves
  x = np.linspace(0, 1, 100)
  psf = np.sum([amp[i] * Gaussian(x, x0[i], sigma[i]) for i in range(len(x0))], axis = 0)
  ipvf = Polynomial(x, ipv)
  prf = psf * ipvf

  fig, ax = pl.subplots(1)
  pl.plot(x, psf, color = 'k', ls = '--')
  pl.plot(x, ipvf, color = 'r', ls = '--')
  pl.plot(x, prf, color = 'k')
  pl.fill_between(x, 0, prf, color = 'k', alpha = 0.1)
  
  print("Approximate flux: %.5f" % np.trapz(prf, x))
  print("Exact flux:       %.5f" % PRF(x0, sigma, amp, ipv))
  pl.show()