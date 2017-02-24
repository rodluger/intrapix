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
from matplotlib.ticker import FuncFormatter
from scipy.special import erf
from scipy.integrate import quad, dblquad
import timeit, builtins
from tqdm import tqdm
prange = lambda x: tqdm(range(x))

class GaussInt(object):
  '''
  Returns the definite integrals of x^n * exp(-ax^2 + bx + c) from 0 to 1.
  
  '''
  
  def __init__(self, a, b, c):
    '''
    
    '''
    
    self.a = a
    self.b = b
    self.c = c
    p = np.sqrt(self.a)
    q = self.b / (2 * p)
    self.GI0 = np.exp(q ** 2 + self.c) * np.sqrt(np.pi) * (erf(q) + erf(p - q)) / (2 * p)

  def __call__(self, n):
    '''
    
    '''
    
    if n == 0:
      return self.GI0
    elif n == 1:
      return (1 / (2 * self.a)) * (np.exp(self.c) * (1 - np.exp(self.b - self.a)) + self.b * self.GI0)
    elif n == 2:
      return (1 / (4 * self.a ** 2)) * (np.exp(self.c) * (self.b - (2 * self.a + self.b) * np.exp(self.b - self.a)) + (2 * self.a + self.b ** 2) * self.GI0)
    elif n == 3:
      return (1 / (8 * self.a ** 3)) * (np.exp(self.c) * (4 * self.a + self.b ** 2 - (4 * self.a ** 2 + 4 * self.a + 2 * self.a + self.b + self.b ** 2) * np.exp(self.b - self.a)) + self.b * (6 * self.a + self.b ** 2) * self.GI0)
    else:
      # TODO
      return 0.

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

def PolyGaussIntegrand1D(y, cx, cy, amp, x0, y0, sx, sy, rho):
  '''
  
  '''
  
  # Dimensions
  N = len(cy)
  K = len(x0)
  
  # Get the y IPV
  f = Polynomial(y, cy)
  
  # Our integrand is the expression f * g
  g = y * 0.
  
  # Loop over the components of the PSF
  for k in range(K):
  
    # Get the x Gaussian integrals
    a = 1 / (2 * (1 - rho[k] ** 2) * sx[k] ** 2)
    b = ((y - y0[k]) * rho[k] * sx[k] + x0[k] * sy[k]) / ((1 - rho[k] ** 2) * sx[k] ** 2 * sy[k])
    c = -(x0[k] ** 2 / sx[k] ** 2 + (y - y0[k]) ** 2 / sy[k] ** 2 + 2 * x0[k] * (y - y0[k]) * rho[k] / (sx[k] * sy[k])) / (2 * (1 - rho[k] ** 2))
    norm = (2 * np.pi * sx[k] * sy[k] * np.sqrt(1 - rho[k] ** 2))
    GI = GaussInt(a, b, c)
  
    # Loop over the orders of the x IPV
    for n in range(N):
      g += (amp[k] / norm) * cx[n] * GI(n)
  
  # We're done!
  return f * g

def Gauss2D(x, y, amp, x0, y0, sx, sy, rho):
  '''
  
  '''
  
  norm = (2 * np.pi * sx * sy * np.sqrt(1 - rho ** 2))
  return (amp / norm) * np.exp(-((x - x0) ** 2 / sx ** 2 + (y - y0) ** 2 / sy ** 2 - 2 * rho * (x - x0) * (y - y0) / (sx * sy)) / (2 * (1 - rho ** 2)))

def PolyGaussIntegrand2D(x, y, cx, cy, amp, x0, y0, sx, sy, rho):
  '''
  
  '''
  
  # Dimensions
  K = len(x0)
  
  # Get the IPV functions
  f = Polynomial(y, cy)
  g = Polynomial(x, cx)
  
  # Loop over the components of the PSF
  h = np.sum([Gauss2D(x, y, amp[k], x0[k], y0[k], sx[k], sy[k], rho[k]) for k in range(K)], axis = 0)

  # We're done!
  return f * g * h

def TestIntegration():
  '''
  
  '''
  
  # Define the params
  cx = np.random.randn(3); cx[0] = np.abs(cx[0])
  cy = np.random.randn(3); cy[0] = np.abs(cy[0])
  amp = [1.]
  x0 = np.random.randn(1)
  y0 = np.random.randn(1)
  sx = 0.5 + 0.1 * np.random.randn(1)
  sy = 0.5 + 0.1 * np.random.randn(1)
  rho = 2 * (np.random.rand(1) - 0.5)
  
  # Define our semi-analytic and numerical integrators
  fsem = lambda: PixelFlux(cx, cy, amp, x0, y0, sx, sy, rho, semi = True)
  fnum = lambda: PixelFlux(cx, cy, amp, x0, y0, sx, sy, rho, semi = False)
  
  # Time the calls to each function
  builtins.__dict__.update(locals())
  tsem = timeit.timeit('fsem()', number = 100) / 100.
  tnum = timeit.timeit('fnum()', number = 100) / 100.
  
  # Print
  print("Semi-analytic (%.1e s): %.9e" % (tsem, fsem()))
  print("Numerical     (%.1e s): %.9e" % (tnum, fnum()))
  print("Difference    (   %.1f x): %.9e" % (tnum/tsem, np.abs(1 - fnum()/fsem())))

def PixelFlux(cx, cy, amp, x0, y0, sx, sy, rho, semi = True, **kwargs):
  '''
  
  '''
  
  if semi:
    F = lambda y: PolyGaussIntegrand1D(y, cx, cy, amp, x0, y0, sx, sy, rho)
    res, err = quad(F, 0, 1, **kwargs)
  else:
    F = lambda y, x: PolyGaussIntegrand2D(x, y, cx, cy, amp, x0, y0, sx, sy, rho)
    res, err = dblquad(F, 0, 1, lambda x: 0, lambda x: 1, **kwargs)
  return res

def GenerateData(nx = 8, ny = 8, sx = 0.5, sy = 0.75, amp = 1.,
                 rho = 0.3, cx = [0.75, 1, -1], cy = [0.75, 1, -1],
                 motion_file = '201367065', data_file = '201367065', **kwargs):
  '''
  
  '''
  
  # Vectorize the PSF params
  rho = np.atleast_1d(rho)
  sx = np.atleast_1d(sx)
  sy = np.atleast_1d(sy)
  amp = np.atleast_1d(amp)
  
  # Load the motion vectors
  motion_vectors = np.load('data/motion/%s.npz' % motion_file)
  x0 = (motion_vectors['x0'] + nx / 2.).reshape(-1, 1)
  y0 = (motion_vectors['y0'] + ny / 2.).reshape(-1, 1)
  ncad = kwargs.pop('ncad', len(x0))
  
  # Loop over the time and pixel arrays
  fpix = np.zeros((ncad, nx, ny))
  for n in prange(ncad):
    for i in range(nx):
      for j in range(ny):
        fpix[n,j,i] = PixelFlux(cx, cy, amp, x0[n] - i, y0[n] - ny + j + 1, sx, sy, rho, **kwargs)

  # Save
  np.savez('data/flux/%s.npz' % data_file, fpix = fpix, nx = nx, ny = ny, sx = sx, sy = sy,
           amp = amp, rho = rho, cx = cx, cy = cy, motion_file = motion_file, ncad = ncad)

def PlotData(data_file = '201367065', res = 50, eps = 0.1, **kwargs):
  '''
  
  '''
  
  # Load
  data = np.load('data/flux/%s.npz' % data_file)
  fpix = data['fpix']
  ncad = data['ncad']
  cad = np.arange(ncad)
  nx, ny = fpix.shape[1], fpix.shape[2]
  sx = data['sx']
  sy = data['sy']
  amp = data['amp']
  rho = data['rho']
  cx = data['cx']
  cy = data['cy']
  motion_file = data['motion_file']
  motion_vectors = np.load('data/motion/%s.npz' % motion_file)
  x0 = (motion_vectors['x0'][:ncad] + nx / 2.).reshape(-1, 1)
  y0 = (motion_vectors['y0'][:ncad] + ny / 2.).reshape(-1, 1)
  
  # Inter-pixel variability
  psv = np.maximum(0, np.minimum(1, 1 + eps * np.random.randn(ny, nx)))
  psvz = Zoom(psv, res)
  fpix *= psv.reshape(-1, psv.shape[0], psv.shape[1])
  flux = np.sum(fpix, axis = (1,2))
  
  # Compute the high res model
  X, Y = Grid(nx, ny, res)
  ipv = IPV(nx, ny, res, cx, cy)
  fhires = psvz * ipv * np.sum([Gauss2D(X, Y, amp[k], x0[0][k], y0[0][k], sx[k], sy[k], rho[k]) for k in range(len(sx))], axis = 0)
  fhires = np.flipud(fhires)
  
  # Plot
  fig = pl.figure(figsize = (12, 8))
  fig.subplots_adjust(top = 0.95, bottom = 0.1, left = 0.075, right = 0.925, wspace = 0.05, hspace = 0.1)
  axccd = pl.subplot2grid((4, 3), (0, 0), colspan = 1, rowspan = 2)  
  axhi = pl.subplot2grid((4, 3), (0, 1), colspan = 1, rowspan = 2)
  axlo = pl.subplot2grid((4, 3), (0, 2), colspan = 1, rowspan = 2)  
  axm = pl.subplot2grid((4, 3), (2, 0), colspan = 2, rowspan = 1) 
  axf = pl.subplot2grid((4, 3), (3, 0), colspan = 2, rowspan = 1) 
  axc = pl.subplot2grid((4, 3), (2, 2), colspan = 1, rowspan = 2)
  axccd.imshow(np.flipud(psvz * ipv), extent = (0, nx, 0, ny), origin = 'lower', aspect = 'auto', cmap = pl.get_cmap('Greys_r'))
  hires = axhi.imshow(fhires, extent = (0, nx, 0, ny), origin = 'lower', aspect = 'auto')
  lores = axlo.imshow(fpix[0], extent = (0, nx, 0, ny), origin = 'lower', aspect = 'auto')
  axm.plot(cad, x0, 'k-', alpha = 0.75, lw = 1, label = 'X')
  axm.plot(cad, y0, 'r-', alpha = 0.75, lw = 1, label = 'Y')
  axf.plot(cad, flux, 'k.', alpha = 0.3, ms = 1)
  axc.plot(x0, flux, 'k.', alpha = 0.3, label = 'X')
  axc.plot(y0, flux, 'r.', alpha = 0.3, label = 'Y')
  tracker1 = axm.axvline(0, color = 'r')
  tracker2 = axf.axvline(0, color = 'r')

  # Appearance
  for axis in [axccd, axhi, axlo]:
    axis.set_xticks([])
    axis.set_yticks([])
  for axis in [axf, axm]:
    axis.margins(0, 0.2)
  axm.set_xticklabels([])
  axm.set_ylabel(r'Position', fontsize = 12)
  axf.set_ylabel(r'Flux', fontsize = 12)
  axf.set_xlabel(r'Cadence', fontsize = 12)
  axm.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p : '%.2f' % x))
  axf.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p : '%.2f' % x))
  axc.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p : '%.2f' % x))
  axm.legend(loc = 'upper left', fontsize = 10)
  axc.legend(loc = 'upper left', fontsize = 10)
  axc.set_ylabel(r'Position', fontsize = 12)
  axc.set_xlabel(r'Flux', fontsize = 12)
  axc.yaxis.tick_right()
  axc.yaxis.set_label_position("right")
  axccd.set_title('Detector Sensitivity')
  axhi.set_title('Sub-pixel Flux')
  axlo.set_title('Pixel Flux')
  
  # Animate!
  def run(i):
    fhires = psvz * ipv * np.sum([Gauss2D(X, Y, amp[k], x0[i][k], y0[i][k], sx[k], sy[k], rho[k]) for k in range(len(sx))], axis = 0)
    fhires = np.flipud(fhires)
    hires.set_data(fhires)
    lores.set_data(fpix[i])
    tracker1.set_xdata([i, i])
    tracker2.set_xdata([i, i])
    return hires, lores, tracker1, tracker2
    
  ani = animation.FuncAnimation(fig, run, frames=len(x0), interval = 1, repeat=True)
  pl.show()

if __name__ == '__main__':
  #GenerateData()
  PlotData()