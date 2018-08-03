""" utils."""

import numpy as np
# import math
# from scipy.fftpack import fft, ifft, fftfreq
# import os
# import pickle
import matplotlib.pyplot as plt
from mayavi import mlab
# import matplotlib.gridspec as gridspec
# import subprocess
# import h5py
# import glob
from xseis import xutil


def mlab_contour(g, ncont=10, vfmin=0.1, vfmax=0.1, ranges=None, vlims=None, cbar=True):
	
	if vlims is None:
		gmin, gmax, p2p = xutil.MinMax(g)
		vmin, vmax = gmin + p2p * vfmin, gmax - vfmax * p2p
	else:
		vmin, vmax = vlims
	contours = list(np.linspace(vmin, vmax, ncont))
	src = mlab.pipeline.scalar_field(g)
	# mlab.pipeline.iso_surface(src, contours=contours, opacity=0.3, colormap='viridis', vmin=vmin, vmax=vmax)
	mlab.outline()
	# mlab.axes(line_width=0.5, xlabel='Z', ylabel='Y', zlabel='X', ranges=ranges)
	mlab.axes(line_width=0.5, xlabel='X', ylabel='Y', zlabel='Z', ranges=ranges)
	mlab.pipeline.iso_surface(src, contours=contours[:-1], opacity=0.2, colormap='viridis', vmin=vmin, vmax=vmax)
	mlab.pipeline.iso_surface(src, contours=contours[-1:], opacity=0.8, colormap='viridis', vmin=vmin, vmax=vmax)
	if cbar:
		mlab.colorbar(orientation='vertical')
	return src


def power(output, gdef, stalocs=None, lmax=None):

	shape, origin, spacing = gdef[:3], gdef[3:6], gdef[6]
	grid = output.reshape(shape)

	lims = np.zeros((3, 2))
	lims[:, 0] = origin
	lims[:, 1] = origin + shape * spacing
	lims[0] -= lims[0, 0]
	lims[1] -= lims[1, 0]

	fig = mlab.figure(size=(1000, 901))
	ranges = list(lims.flatten())
	src = mlab_contour(grid, ncont=6, vfmin=0.1, vfmax=0.02, ranges=ranges)
	# x, y, z = (sloc - origin) / spacing
	if stalocs is not None:
		x, y, z = (stalocs - origin).T / spacing
		mlab.points3d(x, y, z, color=(1, 0, 0), scale_factor=1.0)

	if lmax is not None:
		x, y, z = (lmax - origin) / spacing
		mlab.points3d(x, y, z, color=(0, 0, 0), scale_factor=2.0)
