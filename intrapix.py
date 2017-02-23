#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
intrapix.py
-----------

In the future:

  G = lambda y, x: Gauss2D(x, y, 5, 5, 0.2, 0.4, 0)
  print(dblquad(G, 0, 10, lambda x: 0, lambda x: 10))

'''

from __future__ import division, print_function, absolute_import, unicode_literals
import numpy as np
import matplotlib.pyplot as pl
import matplotlib.animation as animation
from scipy.integrate import dblquad

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
  norm = 1. / (2 * np.pi * sx * sy) # this is wrong (only for theta = 0)
  return norm * np.exp(-(a * (x - x0) ** 2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0) ** 2))

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
    res += amp[i] * Gauss2D(X, Y, x0[i], y0[i], sx[i], sy[i], theta[i])
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

def Zoom(arr, res):
  '''
  
  '''
  
  assert type(res) is int and res > 1, "Argument `res` must be a positive integer."
  ny, nx = np.shape(arr)
  zarr = np.ones((ny * res, nx * res))
  for i in range(nx):
    for j in range(ny):
      zarr[j * res:(j + 1) * res, i * res:(i + 1) * res] = arr[j,i]
  return zarr

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
  return pix.T

def PixelateQuad(nx, ny, x0, y0, sx, sy, amp, theta, cx, cy, psv):
  '''
  Not yet working...
  
  '''
  
  pix = np.empty((nx, ny))
  for i in range(nx):
    for j in range(ny):
      F = lambda y, x: Polynomial(x - np.floor(x), cx) * \
                       Polynomial(y - np.floor(y), cy) * \
                       psv[j][i] * \
                       Gauss2D(x, y, x0, y0, sx, sy, theta)
      pix[j][i] = dblquad(F, i, i+1, lambda x: j, lambda x: j+1)[0]  
  return pix
  
def Animate(nx = 8, ny = 8, res = 100, sx = 1.0, sy = 0.75, amp = 1.,
            theta = -0.3, cx = [0.75, 1, -1], cy = [0.75, 1, -1], eps = 0.1):
  '''
  Animate the high resolution PRF with a random walk across the detector
  
  '''
    
  # Load the motion vectors
  motion_vectors = np.load('data/motion_vectors.npz')
  x0 = motion_vectors['x0'][2140:] + nx / 2.
  y0 = motion_vectors['y0'][2140:] + ny / 2.
  cad = np.arange(0, len(x0))
  
  # Construct the grids
  psv = 1 + eps * np.random.randn(ny, nx)
  psvz = Zoom(psv, res)
  ipv = IPV(nx, ny, res, cx, cy)
  psf = PSF(nx, ny, res, x0[0], y0[0], sx, sy, amp, theta)
  prf = psvz * ipv * psf
  flx = Pixelate(prf, nx, ny, res)
  
  # Plot
  fig = pl.figure(figsize = (8,8))
  fig.subplots_adjust(top = 0.95, bottom = 0.1, left = 0.1, right = 0.95, wspace = 0.05, hspace = 0.1)
  axhi = pl.subplot2grid((4, 2), (0, 0), colspan = 1, rowspan = 2)
  axlo = pl.subplot2grid((4, 2), (0, 1), colspan = 1, rowspan = 2)  
  axx0 = pl.subplot2grid((4, 2), (2, 0), colspan = 2, rowspan = 1) 
  axy0 = pl.subplot2grid((4, 2), (3, 0), colspan = 2, rowspan = 1) 
  hires = axhi.imshow(prf, extent = (0, nx, 0, ny), origin = 'lower', aspect = 'auto')
  lores = axlo.imshow(flx, extent = (0, nx, 0, ny), origin = 'lower', aspect = 'auto')
  axx0.plot(cad, x0, 'k-', alpha = 0.75, lw = 1)
  axy0.plot(cad, y0, 'k-', alpha = 0.75, lw = 1)
  tracker1 = axx0.axvline(0, color = 'r')
  tracker2 = axy0.axvline(0, color = 'r')

  # Appearance
  for axis in [axhi, axlo]:
    axis.set_xticks([])
    axis.set_yticks([])
  for axis in [axx0, axy0]:
    axis.margins(0, None)
  axx0.set_xticklabels([])
  axx0.set_ylabel(r'$x_0$ (pixels)', fontsize = 12)
  axy0.set_ylabel(r'$y_0$ (pixels)', fontsize = 12)
  axy0.set_xlabel(r'Cadence', fontsize = 12)
  
  # Animate!
  def run(i):
    psf = PSF(nx, ny, res, x0[i], y0[i], sx, sy, amp, theta)
    prf = psvz * ipv * psf
    flx = Pixelate(prf, nx, ny, res)
    hires.set_data(prf)
    lores.set_data(flx)
    tracker1.set_xdata([i, i])
    tracker2.set_xdata([i, i])
    return hires, lores, tracker1, tracker2
    
  ani = animation.FuncAnimation(fig, run, frames=len(x0), interval=10, repeat=True)
  pl.show()

def LnLike(p, **kwargs):
  '''
  TODO!
  
  '''
  
  fpix = kwargs['fpix']
  ipv_order = kwargs['ipv_order']
  
  sx = p[0]
  sy = p[1]
  theta = p[2]
  eps = p[3]
  cx = p[4:4+ipv_order+1]
  cy = p[4+1+ipv_order:]
  
def RunMCMC():
  '''
  
  '''
  
  # General params
  nx = 8
  ny = 8
  res = 100
  
  # The "true" parameters
  cx = [0.75, 1, -1]
  cy = [0.75, 1, -1]
  theta = 0.3
  sx = 0.5
  sy = 0.75
  eps = 0.1
  amp = 1
  
  # Load the motion vectors
  motion_vectors = np.load('data/motion_vectors.npz')
  x0 = motion_vectors['x0'][3500:] + nx / 2.
  y0 = motion_vectors['y0'][3500:] + ny / 2.
  ncad = len(x0)
  
  # Construct the sensitivity map
  psv = 1 + eps * np.random.randn(ny, nx)
  ipv = IPV(nx, ny, res, cx, cy)
  
  # Compute the pixel fluxes
  fpix = np.empty((ncad, nx, ny))
  for i in range(ncad):
    psf = PSF(nx, ny, res, x0[i], y0[i], sx, sy, amp, theta)
    prf = Zoom(psv, res) * ipv * psf
    fpix[i] = Pixelate(prf, nx, ny, res)

  # Our initial guess
  p0 = np.concatenate([[np.abs(np.random.randn()), np.abs(np.random.randn()), 
                        2 * np.pi * np.random.rand(), 0.1 * np.random.randn()], 
                        np.random.randn(3), np.random.randn(3)])
  ipv_order = len(cx) - 1
  kwargs = dict(fpix = fpix, ipv_order = ipv_order)
  
  LnLike(p0, **kwargs)  
  

if __name__ == '__main__':
  
  Animate()