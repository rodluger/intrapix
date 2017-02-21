#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
intrapix.py
-----------

'''

from __future__ import division, print_function, absolute_import, unicode_literals
import numpy as np
import matplotlib.pyplot as pl
import matplotlib.animation as animation

def Grid(nx, ny, res):
  '''
  
  '''
  
  x = np.linspace(0, nx, nx * res)
  y = np.linspace(0, ny, ny * res)
  X, Y = np.meshgrid(x, y)
  return X, Y

def Polynomial(x, coeffs):
  '''
  Returns a polynomial with coefficients `coeffs` evaluated at `x`
  
  '''
  
  return np.sum([c * x ** m for m, c in enumerate(coeffs)], axis = 0)

def Gauss2D(x, y, x0, y0, sx, sy, theta):
  '''
  https://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function
  
  '''
  
  a = ((np.cos(theta) ** 2) / (2 * sx ** 2)) + ((np.sin(theta) ** 2) / (2 * sy ** 2))
  b = -((np.sin(2 * theta)) / (4 * sx ** 2)) + ((np.sin(2 * theta)) / (4 * sy ** 2))
  c = ((np.sin(theta) ** 2) / (2 * sx ** 2)) + ((np.cos(theta) ** 2) / (2 * sy ** 2))
  return np.exp(-(a * (x - x0) ** 2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0) ** 2))

def GaussMixture2D(X, Y, x0, y0, sx, sy, amp, theta):
  '''
  A mixture of 2D Gaussians, each with mean (`x0`, `y0`),
  standard deviation (`sx`, `sy`), amplitude `amp` and 
  rotation angle `theta`, evaluated on the mesh `X`, `Y`
  
  '''
  
  # Ensure we have arrays
  x0 = np.atleast_1d(x0)
  y0 = np.atleast_1d(y0)
  sx = np.atleast_1d(sx)
  sy = np.atleast_1d(sy)
  amp = np.atleast_1d(amp)
  theta = np.atleast_1d(theta)
  assert len(x0) == len(y0) == len(sx) == len(sy) == len(amp) == len(theta), "Size mismatch!"
  nmix = len(x0)
  
  # Add the gaussians
  res = np.zeros_like(X)
  for i in range(nmix):
    res += Gauss2D(X, Y, x0[i], y0[i], sx[i], sy[i], theta[i])
  return res

def IPV(nx, ny, res, cx, cy):
  '''
  Intra-pixel variability function, given coefficient arrays `cx` and `cy`
  
  '''
  
  z = np.linspace(0, 1, res)
  ipv = np.ones((ny * res, nx * res))  
  for i in range(nx):
    ipv[:, i * res:(i + 1) * res] *= Polynomial(z, cx).reshape(1,-1)
  for j in range(ny):
    ipv[j * res:(j + 1) * res, :] *= Polynomial(z, cy).reshape(-1,1)
  return ipv

def PSV(nx, ny, res, eps):
  '''
  Inter-pixel sensitivity variation function, given amplitude `eps`
  
  '''
  
  psv = np.ones((ny * res, nx * res))
  for i in range(nx):
    for j in range(ny):
      psv[j * res:(j + 1) * res, i * res:(i + 1) * res] = 1 + eps * np.random.randn()
  return psv

def PSF(nx, ny, res, x0, y0, sx, sy, amp, theta):
  '''
  A PSF generated from the sum of any number of 2D Gaussians
  
  '''
  
  X, Y = Grid(nx, ny, res)
  return GaussMixture2D(X, Y, x0, y0, sx, sy, amp, theta)

def Pixelate(prf, nx, ny, res):
  '''
  Integrate the PSF over each of the pixels
  
  '''
  
  pix = np.empty((nx, ny))
  for i in range(nx):
    for j in range(ny):
      pix[i][j] = np.sum(prf[j * res:(j + 1) * res, i * res:(i + 1) * res])
  return pix

def Animate(nx = 8, ny = 8, res = 100, sx = 1.5, sy = 0.75, amp = 1.,
            theta = -0.3, cx = [0.75, 1, -1], cy = [0.75, 1, -1],
            eps = 0.1, dx = 0.25, dy = 0.25):
  '''
  Animate the high resolution PRF with a random walk across the detector
  
  '''
  
  # Construct the grids
  x0 = nx / 2. - 1
  y0 = ny / 2. - 1
  psv = PSV(nx, ny, res, eps)
  ipv = IPV(nx, ny, res, cx, cy)
  psf = PSF(nx, ny, res, x0, y0, sx, sy, amp, theta)
  prf = psv * ipv * psf

  # Plot
  fig, ax = pl.subplots(1, figsize = (5,5))
  image = ax.imshow(prf, extent = (0, nx, 0, ny), origin = 'lower')
  ax.set_xticks([])
  ax.set_yticks([])

  # Animate!
  def data_gen():
    x = x0
    y = y0
    while True:
      x += dx * np.random.randn()
      y += dy * np.random.randn()
      if y < 0:
        y = 0
      elif x < 0:
        x = 0
      yield x, y
      
  def run(data):
    x0, y0 = data
    psf = PSF(nx, ny, res, x0, y0, sx, sy, amp, theta)
    prf = psv * ipv * psf
    image.set_data(prf)
    return image,
    
  ani = animation.FuncAnimation(fig, run, data_gen, interval=10, repeat=False)
  pl.show()

if __name__ == '__main__':
  Animate()