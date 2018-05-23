import numpy as np
import os
import matplotlib.pyplot as plt
from importlib import reload

plt.ion()


npz = np.load('ttable_ot.npz')
lims, spacing, shape = npz['lims'], npz['spacing'], npz['shape']
tts = npz['tts']
stas = npz['stas']
stalocs = npz['stalocs']
evloc = npz['evloc']


sta_ix = 10

nx, ny, nz = shape
grid = tts[sta_ix].reshape(nz, ny, nx)

zix = 10

depth = lims[2][0] + zix * spacing
print('plot z=%.2fm slice of tt grid for station %s' % (depth, stas[sta_ix]))

# plot z slice for station 
zslice = grid[zix]

fig = plt.figure()
ax = plt.subplot()

xl, yl, zl = lims.reshape(3, 2)
extent = [xl[0], xl[1], yl[0], yl[1]]
im = ax.imshow(zslice, origin='lower', aspect='equal', extent=extent,
		 cmap='viridis', interpolation='none')
plt.colorbar(im)
plt.show()
