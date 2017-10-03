from intrapix import PixelFlux
import pysyzygy as ps
import numpy as np
import matplotlib.pyplot as pl
from tqdm import tqdm

def GenerateData(nx = 7, ny = 7, sx = 0.5, sy = 0.75, amp = 1.,
                 x0 = 3.5, y0 = 3.5,
                 rho = 0.3, cx = [0.75, 1, -1], cy = [0.75, 1, -1],
                 noise = 0.0005, ncad = 100, **kwargs):
    '''

    '''
    
    # Time-variable focus
    sx *= (1 + 0.25 * np.cos(np.linspace(0, 4 * np.pi, ncad)))
    sy *= (1 + 0.25 * np.sin(np.linspace(0, 4 * np.pi, ncad)))
    rho *= (1 + 0.25 * np.sin(np.linspace(0, 2 * np.pi, ncad)))
    
    # Loop over the time and pixel arrays
    fpix = np.zeros((ncad, ny, nx))
    for n in tqdm(range(ncad)):
    
        # Add a bit of jitter
        dx = 0.085 * np.random.randn()
        dy = 0.085 * np.random.randn()
        
        for i in range(nx):
            for j in range(ny):
                
                # Compute the PSF
                fpix[n,j,i] = PixelFlux(cx, cy, [amp], [x0 - i + dx], 
                                        [y0 - ny + j + 1 + dy], [sx[n]], 
                                        [sy[n]], [rho[n]], **kwargs)
                
                # Add shot noise
                fpix[n,j,i] += noise * np.sqrt(fpix[n,j,i]) * np.random.randn()
                
    return fpix
    
# Get the raw pixel flux
calc = False
ncad = 3000
time = np.linspace(0, 13.7 * 2, ncad)
if calc:
    fpix = GenerateData(ncad = ncad)
    np.savez('fpix.npz', fpix = fpix)
else:
    fpix = np.load('fpix.npz')['fpix']
fpix[np.where(np.isnan(fpix))] = 0.
flux_notrn = np.sum(fpix, axis = (1,2))

# Add a transit model
trn = ps.Transit(t0 = 7., RpRs = 0.03, per = 12, rhos = 0.01)
transit = trn(time)
fpix *= transit.reshape(-1, 1, 1)

# Get the SAP flux
flux = np.sum(fpix, axis = (1,2))
med = np.nanmedian(flux)

# Regress with 2nd order PLD. Note that 
# we are cheating a bit since I'm regressing
# on the flux with *no transit* model.
# In reality, we would *jointly fit*
# the systematics and the transit, but
# this is a shortcut.
D = fpix.reshape(ncad, -1) / flux.reshape(-1, 1)
D = np.hstack((D, D * D))
A = np.dot(D.T, D)
B = np.dot(D.T, flux_notrn)
w = np.linalg.solve(A, B)
pld_model = np.dot(D, w)
pld_model -= np.nanmedian(pld_model)

# Plot
fig, ax = pl.subplots(2, figsize = (7, 4))
ax[0].plot(time, flux / med, 'k.', ms = 2, alpha = 0.3)
ax[0].plot(time, (pld_model + np.nanmedian(flux)) / med, 'r-', lw = 1, alpha = 0.2)
ax[1].plot(time, (flux - pld_model) / med, 'k.', ms = 2, alpha = 0.3)
ax[1].plot(time, transit, color = 'cornflowerblue', ls = '-', lw = 2, alpha = 1)

ax[1].set_xlabel('Time [days]', fontweight = 'bold')
ax[0].set_ylabel('Raw Flux', fontweight = 'bold')
ax[1].set_ylabel('Residuals', fontweight = 'bold')

fig.savefig('breathing.pdf', bbox_inches = 'tight')