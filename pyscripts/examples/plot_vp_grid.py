import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab
import xutil
import xplot
# import os
# import pickle
# import subprocess
# import h5py
# import glob
# import gridutil as gut
# from importlib import reload


def plot_contours(g, ncont=10, vfmin=0.1, vfmax=0.1, ranges=None, vlims=None):
	mlab.clf()
	
	if vlims is None:
		gmin, gmax, p2p = xutil.MinMax(g)
		vmin, vmax = gmin + p2p * vfmin, gmax - vfmax * p2p
	else:
		vmin, vmax = vlims
	contours = list(np.linspace(vmin, vmax, ncont))

	src = mlab.pipeline.scalar_field(g)

	# mlab.pipeline.iso_surface(src, contours=contours, opacity=0.3, colormap='viridis', vmin=vmin, vmax=vmax)
	mlab.outline()
	mlab.axes(line_width=0.5, xlabel='Z', ylabel='Y', zlabel='X', ranges=ranges)
	mlab.pipeline.iso_surface(src, contours=contours[:-1], opacity=0.2, colormap='viridis', vmin=vmin, vmax=vmax)
	mlab.pipeline.iso_surface(src, contours=contours[-1:], opacity=0.8, colormap='viridis', vmin=vmin, vmax=vmax)
	mlab.colorbar(orientation='vertical')
	# print(mlab.view(), mlab.roll())


# utm_shift = np.array([650000, 4766000, 0])
data = np.load('dataJP.npz')
print(list(data.keys()))

grid, lims, spacing = data['grid'], data['lims'], data['spacing']
vels, loc_win, maxvals = data['vels'], data['loc_winners'], data['maxvals']
sta_locs, blast_loc = data['sta_locs'], data['blast_loc']
sigs = data['sigs']  # input waveforms

plt.ion()

xplot.sigs(sigs)


dist2blast = np.array([np.linalg.norm(blast_loc - lw) for lw in loc_win])


dmin = np.argmin(dist2blast)
print("Velocity at min dist: %.2f m/s" % vels[dmin])
print("Loc at min dist: ",  loc_win[dmin])


fig = plt.figure()
ax = plt.subplot(211)
ax.plot(vels, maxvals)
ax.set_ylabel('Value of gridmax')
ax.set_xlim([vels[0], vels[-1]])

ax = plt.subplot(212)
ax.plot(vels, dist2blast, color='green')
ax.axvline(vels[dmin], color='red', linestyle='--')
ax.set_ylabel('Distance to blast')
ax.set_xlabel('Velocity used to compute tts (m/s)')
ax.set_xlim([vels[0], vels[-1]])
fig.tight_layout()


zfo = [2, 1, 0]  # Grids saved with Z as slowest changin coordinate
ranges = list(lims[zfo].flatten() / 1000)


fig = mlab.figure(size=(1000, 901))

plot_contours(grid, ncont=20, vfmin=0.2, vfmax=0.02, ranges=ranges)
x, y, z = (sta_locs - lims[:, 0]).T / spacing
mlab.points3d(z, y, x, color=(0, 0, 1), scale_factor=2)

x, y, z = (blast_loc - lims[:, 0]).T / spacing
mlab.points3d(z, y, x, color=(1, 0, 0), scale_factor=2)


# mlab.view(azimuth=67.13, elevation=80.76, distance=223.213, focalpoint='auto', roll=-151.879)
# np.savez('dataJP.npz', grid=g, lims=lims, spacing=spacing, maxvals=maxes, vels=vels,
# 		 loc_winners=lmax.astype(np.float32), sta_locs=locs, blast_loc=blast, sigs=pp)
