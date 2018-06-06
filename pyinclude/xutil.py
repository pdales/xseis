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
from scipy.fftpack import fft, ifft, rfft, fftfreq


def roll_data(data, tts):
	droll = np.zeros_like(data)

	for i, sig in enumerate(data):
		droll[i] = np.roll(sig, -tts[i])
	return droll


def comb_channels(data, cmap):

	# groups = []
	# for grp in np.unique(cmap):
	# 	groups.append(np.where(cmap == grp)[0])

	groups = [np.where(sk == cmap)[0] for sk in np.unique(cmap)]
	dstack = np.zeros((len(groups), data.shape[1]))

	for i, grp in enumerate(groups):
		dstack[i] = np.sum(np.abs(data[grp]), axis=0)

	return dstack

	
def mlab_coords(locs, lims, spacing):
	return (locs - lims[:, 0]).T / spacing


def SearchClusters(data, dmin):

	inds = np.arange(data.shape[1])

	slocs = []
	vals = []
	for j, ix in enumerate(inds):
		print(j)
		lmax = data[:-1, ix, 1:]
		tmp = []
		for k, centroid in enumerate(lmax):
			diff = np.linalg.norm(lmax - centroid, axis=-1)
			tmp.append(np.where(diff < dmin)[0].size)

		imax = np.argmax(tmp)
		vals.append(tmp[imax])
		slocs.append(lmax[imax])

	vals = np.array(vals)
	slocs = np.array(slocs)

	return vals, slocs


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


def bandpass(data, band, sr, corners=4, zerophase=True):
	from scipy.signal import sosfilt, zpk2sos, iirfilter

	freqmin, freqmax = band
	fe = 0.5 * sr
	low = freqmin / fe
	high = freqmax / fe

	z, p, k = iirfilter(corners, [low, high], btype='band',
						ftype='butter', output='zpk')
	sos = zpk2sos(z, p, k)
	if zerophase:
		firstpass = sosfilt(sos, data)
		if len(data.shape) == 1:
			return sosfilt(sos, firstpass[::-1])[::-1]
		else:
			return np.fliplr(sosfilt(sos, np.fliplr(firstpass)))
	else:
		return sosfilt(sos, data)


def filter(data, btype, band, sr, corners=4, zerophase=True):
	from scipy.signal import sosfilt, zpk2sos, iirfilter
	# btype: lowpass, highpass, band

	fe = 0.5 * sr

	z, p, k = iirfilter(corners, band / fe, btype=btype,
						ftype='butter', output='zpk')
	sos = zpk2sos(z, p, k)
	if zerophase:
		firstpass = sosfilt(sos, data)
		if len(data.shape) == 1:
			return sosfilt(sos, firstpass[::-1])[::-1]
		else:
			return np.fliplr(sosfilt(sos, np.fliplr(firstpass)))
	else:
		return sosfilt(sos, data)


def norm2d(d):
	return d / np.max(np.abs(d), axis=1)[:, np.newaxis]


def cross_corr(sig1, sig2, norm=True, pad=False, phase_only=False, phat=False):
	"""Cross-correlate two signals."""
	pad_len = len(sig1)
	if pad is True:
		pad_len *= 2
		# pad_len = signal.next_pow_2(pad_len)

	sig1f = fft(sig1, pad_len)
	sig2f = fft(sig2, pad_len)

	if phase_only is True:
		ccf = np.exp(- 1j * np.angle(sig1f)) * np.exp(1j * np.angle(sig2f))
	else:
		ccf = np.conj(sig1f) * sig2f

	if phat:
		ccf = ccf / np.abs(ccf)

	cc = np.real(ifft(ccf))

	if norm:
		cc /= np.sqrt(energy(sig1) * energy(sig2))

	return np.roll(cc, len(cc) // 2)


def energy(sig, axis=None):
	return np.sum(sig ** 2, axis=axis)


def build_slice_inds(start, stop, wlen, stepsize=None):

	if stepsize is None:
		stepsize = wlen

	overlap = wlen - stepsize
	imin = np.arange(start, stop - overlap - 1, stepsize)
	imax = np.arange(start + wlen, stop + stepsize - 1, stepsize)

	return np.dstack((imin, imax))[0]
