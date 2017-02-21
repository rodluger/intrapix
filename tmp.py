from numpy.polynomial import hermite

def Hermite(z, n):
  '''
  
  '''

  expr = lambda m: (((-1) ** m) * ((2 * z) ** (n - 2 * m))) / (factorial(m, exact = True) * factorial(n - 2 * m, exact = True))
  return factorial(n, exact = True) * np.sum([expr(m) for m in np.arange(0, n // 2 + 1)])

def Hermite(z, n):
  '''
  HermiteH[n, z]
  
  '''
  
  coeffs = np.zeros(n + 1)
  coeffs[-1] = 1
  return hermite.Hermite(coeffs)(z)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
psf.py
------

'''

from __future__ import division, print_function, absolute_import, unicode_literals
import numpy as np
from scipy.special import hyp1f1, gamma, erf
from scipy.misc import comb
import matplotlib.pyplot as pl

HERMITE = np.array([[    1,     0,    -2,     0,    12,     0,  -120,     0,  1680,     0],
                    [    0,     2,     0,   -12,     0,   120,     0, -1680,     0, 30240],
                    [    0,     0,     4,     0,   -48,     0,   720,     0,-13440,     0],
                    [    0,     0,     0,     8,     0,  -160,     0,  3360,     0,-80640],
                    [    0,     0,     0,     0,    16,     0,  -480,     0, 13440,     0],
                    [    0,     0,     0,     0,     0,    32,     0, -1344,     0, 48384],
                    [    0,     0,     0,     0,     0,     0,    64,     0, -3584,     0],
                    [    0,     0,     0,     0,     0,     0,     0,   128,     0, -9216],
                    [    0,     0,     0,     0,     0,     0,     0,     0,   256,     0],
                    [    0,     0,     0,     0,     0,     0,     0,     0,     0,   512]])

def HermiteIndefiniteIntegral(z, n):
  '''
  Integrate[HermiteH[n, z]/E^z^2, z]

  '''
  
  if n == 0:
    return 0.5 * np.sqrt(np.pi) * erf(z)
  elif n == 1:
    return -0.5 * np.exp(-z ** 2)
  else:
    return 2 ** n * np.sqrt(np.pi) * (z * hyp1f1(n / 2. + 1 / 2., 3 / 2., -z ** 2) / gamma(1 / 2. - n / 2.) + \
                                      hyp1f1(n / 2., 1 / 2., -z ** 2) / ( n * gamma(-n / 2.)))

def Hermite(z, n):
  '''
  Returns the n^th order Hermite polynomial evaluated at `z`.
  
  '''
  
  assert n <= HERMITE.shape[0], "Hermite order is too large."
  z = np.atleast_1d(z)
  Z = np.array([z ** m for m in range(0, HERMITE.shape[0])])
  return np.dot(HERMITE[:,n], Z)

def HermiteCoeffs(coeffs):
  '''
  Converst a polynomial with coefficients `coeffs` to a sum of
  Hermite polynomials. Returns the coefficients of the Hermite polynomials.
  
  '''
  
  assert len(coeffs) <= HERMITE.shape[0], "Coefficient array exceeds maximum size."
  w = np.zeros(HERMITE.shape[0])
  w[:len(coeffs)] = coeffs
  return np.linalg.solve(HERMITE, w)

def Gaussian(x, x0, sigma):
  '''
  
  '''
  
  return 1. / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - x0) / sigma) ** 2)

def Polynomial(x, coeffs):
  '''
  
  '''
  
  return np.sum([c * x ** m for m, c in enumerate(coeffs)], axis = 0)

def HermiteIntegral(n, x0, sigma):
  '''
  
  '''
  
  HI = HermiteIndefiniteIntegral((1 - x0) / (np.sqrt(2) * sigma), n) - HermiteIndefiniteIntegral(-x0 / (np.sqrt(2) * sigma), n)
  return 1. / np.sqrt(np.pi) * HI

def PolynomialBasisChange(coeffs, a, b):
  '''
  Performs the transformation f(x) --> f(z) where
  z = ax + b. `coeffs` are the polynomial coefficients
  in increasing order. Returns the coefficients in the
  new basis. Based on http://stackoverflow.com/a/142436.
  
  '''
  
  T = np.zeros((len(coeffs), len(coeffs)))
  for i in range(len(coeffs)):
    for j in range(len(coeffs)):
      T[i][j] = comb(j, i, exact = True) * a ** i * b ** (j - i)
  return np.dot(T, coeffs) 

def PRF(x0, sigma, ipv_coeffs):
  '''
  
  '''
  
  # Stretch the coefficients to a new basis, shifted by `x0` and scaled by `sigma`
  ipv_coeffs = PolynomialBasisChange(ipv_coeffs, 1. / (np.sqrt(2) * sigma), -x0 / (np.sqrt(2) * sigma))
  
  # Compute the coefficients of the Hermite series for this polynomial
  hermite_coeffs = HermiteCoeffs(ipv_coeffs)
  
  # Integrate the polynomial times the gaussian
  return np.sum([c * HermiteIntegral(n, x0, sigma) for n, c in enumerate(hermite_coeffs) if c != 0.])

def PlotPixel(x0 = 0.5, sigma = 0.5, ipv_coeffs = [3 / 4., 1, -1]):
  '''
  
  '''
  
  x = np.linspace(0, 1, 100)
  psf = Gaussian(x, x0, sigma)
  ipv = Polynomial(x, ipv_coeffs)
  prf = psf * ipv
  
  flux_approx = np.trapz(prf, x)
  flux_exact = PRF(x0, sigma, ipv_coeffs)

  fig, ax = pl.subplots(1)
  pl.plot(x, psf, color = 'k', ls = '--')
  pl.plot(x, ipv, color = 'r', ls = '--')
  pl.plot(x, prf, color = 'k')
  pl.fill_between(x, 0, prf, color = 'k', alpha = 0.1)
  
  pl.show()

def PlotPixelResponse(sigma = 0.5, ipv_coeffs = [3 / 4., 1, -1]):
  '''
  
  '''
  
  flux_approx = np.zeros(100)
  flux_exact = np.zeros(100)
  x = np.linspace(0, 1, 100)
  
  for i, x0 in enumerate(x):
    psf = Gaussian(x, x0, sigma)
    ipv = Polynomial(x, ipv_coeffs)
    prf = psf * ipv
    flux_approx[i] = np.trapz(prf, x)
    flux_exact[i] = PRF(x0, sigma, ipv_coeffs)
  pl.plot(x, flux_approx, color = 'r', lw = 2, alpha = 0.3)
  pl.plot(x, flux_exact, color = 'k')
  pl.show()
  quit()
  
PlotPixelResponse(ipv_coeffs = [0., 1.])