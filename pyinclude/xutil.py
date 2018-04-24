""" utils."""

import numpy as np
import math
# import os
# import pickle
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# import subprocess
# import h5py
# import glob
# import datetime
# from scipy.signal import sosfilt, zpk2sos, iirfilter
# from scipy.fftpack import fft, ifft, rfft, fftfreq
# import matplotlib.colors
# import matplotlib.cm as cm


def shift_locs(locs, unshift=False, vals=np.array([1.79236297e+05, 7.09943400e+06, 2.49199997e+02])):
	vals = np.array(vals)
	locs[:, 2] *= -1
	if unshift is True:
		return locs + vals
	else:
		return locs - vals


def shift_locs_ot(locs, unshift=False, vals=np.array([650000., 4766000., 0])):
	vals = np.array(vals)
	if unshift is True:
		return locs + vals
	else:
		return locs - vals


def normVec(v):
	return v / norm(v)


def norm(v):
	return math.sqrt(np.dot(v, v))


def angle_between_vecs(v1, v2):
	return math.acos(np.dot(v1, v2) / (norm(v1) * norm(v2)))


def dist_between(l1, l2):
	return norm(l1 - l2)
	# return np.linalg.norm(l1 - l2, axis=axis)


def build_PShSv_matrix(vec):
	"""Create orientations for channels to be rotated."""
	P = vec / norm(vec)
	SH = np.array([P[1], -P[0], 0])
	SV = np.cross(SH, P)
	return np.array([P, SH, SV])


def rotation_matrix(axis, theta):
	"""Return ccw rotation about the given axis by theta radians."""
	axis = np.asarray(axis)
	theta = np.asarray(theta)
	axis = axis / math.sqrt(np.dot(axis, axis))
	a = math.cos(theta / 2.0)
	b, c, d = -axis * math.sin(theta / 2.0)
	aa, bb, cc, dd = a * a, b * b, c * c, d * d
	bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
	return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
					 [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
					 [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def rotate_one(data, orients, uvec):
	"""Rotate channel data with respect to uvec."""
	contribs = np.sum(orients * uvec, axis=1)
	return np.sum(data * contribs[np.newaxis, :].T, axis=0)


def apply_rotation(vec, theta, axis, eps=1e-10):
	"""Apply rotation_matrix to all vectors."""
	for i in range(3):
		vec[i] = np.dot(rotation_matrix(axis, theta), vec[i])
	vec[np.abs(vec) < eps] = 0


def axes_from_orient(az, dip, roll):
	vec = np.array([[1, 0, 0], [0, 1, 0],  [0, 0, 1]]).astype(float)
	apply_rotation(vec, az, axis=vec[2])
	apply_rotation(vec, dip, axis=vec[1])
	apply_rotation(vec, roll, axis=vec[0])
	return vec


def MedAbsDev(grid):
	return np.median(np.abs(grid - np.median(grid)))


def MinMax(a):
	mn, mx = np.min(a), np.max(a)
	return mn, mx, mx - mn


def read_meta(fname):

	meta = np.loadtxt(fname)
	spacing = meta[6]
	lims = meta[:6].reshape(3, 2)
	shape = (np.diff(lims, axis=1).T[0] // spacing).astype(int)
	# xl, yl, zl = lims
	return lims, spacing, shape


def combine_grids(fles, shape):

	nx, ny, nz = shape

	nfle = len(fles)
	# shape = (nx, nx, nx)
	grids = np.zeros((nfle, nz, ny, nx), dtype=np.float32)

	for i, fn in enumerate(fles):
		print(fn)
		# xl, yl, zl = lims.reshape(3, 2)
		grid = np.load(fn).reshape(nz, ny, nx)
		grids[i] = grid

	return grids


def xyz_max(grid, lims, spacing):
	# thresh = np.std(grid) * nstd
	iwin = np.argmax(grid)

	pt = np.array(np.unravel_index(iwin, grid.shape))[::-1]
	pt = pt * spacing + lims[:, 0]
	return pt


def remap(x, out_min, out_max):
	in_min = np.min(x)
	in_max = np.max(x)
	return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
