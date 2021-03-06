""" utils."""

import numpy as np
import math
from scipy.fftpack import fft, ifft, fftfreq
import os
# import pickle
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# import subprocess
# import h5py
import glob
# import datetime
# from scipy.signal import sosfilt, zpk2sos, iirfilter
# from scipy.fftpack import fft, ifft, rfft, fftfreq
# import matplotlib.colors
# import matplotlib.cm as cm


def get_pt(index, shape, spacing, origin):
	nx, ny, nz = shape
	iz = index % nz
	iy = ((index - iz) // nz) % ny
	ix = index // (nz * ny)

	loc = np.array([ix, iy, iz]) * spacing + origin
	return loc


def imax_to_xyz_gdef(index, gdef):
	shape, origin, spacing = gdef[:3], gdef[3:6], float(gdef[6])
	nx, ny, nz = shape
	iz = index % nz
	iy = ((index - iz) // nz) % ny
	ix = index // (nz * ny)

	loc = np.array([ix, iy, iz], dtype=float) * spacing + origin
	return loc


def ttable_from_nll_grids(path, key="OT.P"):
	fles = np.sort(glob.glob(os.path.join(path, key + '*.time.buf')))
	hfles = np.sort(glob.glob(os.path.join(path, key + '*.time.hdr')))
	assert(len(fles) == len(hfles))
	stas = np.array([f.split('.')[-3].zfill(3) for f in fles], dtype='S4')
	isort = np.argsort(stas)
	fles = fles[isort]
	hfles = hfles[isort]
	names = stas[isort]

	vals = [read_nll_header(fle) for fle in hfles]
	sloc, shape, org, spacing = vals[0]
	slocs = np.array([v[0] for v in vals], dtype=np.float32)
	ngrid = np.product(shape)

	nsta = len(fles)
	tts = np.zeros((nsta, ngrid), dtype=np.float32)

	for i in range(nsta):
		gdata = np.fromfile(fles[i], dtype='f4')
		tts[i] = gdata

	gdef = np.concatenate((shape, org, [spacing])).astype(np.int32)

	ndict = {}
	for i, sk in enumerate(names):
		ndict[sk.decode('utf-8')] = i

	return tts, slocs, ndict, gdef


def read_nll_header(fle):
	# print(fle)
	dat = open(fle).read().split()
	shape = np.array(dat[:3], dtype=int)
	org = np.array(dat[3:6], dtype=np.float32) * 1000.
	spacing = (np.array(dat[6:9], dtype=np.float32) * 1000.)[0]
	sloc = np.array(dat[12:15], dtype=np.float32) * 1000.

	return sloc, shape, org, spacing


def g2mad(grid):
	# return max, median and median absolute dev
	med = np.median(grid)
	mad = np.median(np.abs(grid - med))
	# mad = max(mad, 1)
	out = (grid - med) / mad
	return out


def randomize_band(fsig, band, sr):

	fnew = fsig.copy()
	freqmin, freqmax = band
	samples = len(fnew)
	freqs = np.abs(fftfreq(samples, 1. / sr))
	idx = np.where(np.logical_and(freqs >= freqmin, freqs <= freqmax))[0]

	part = fnew[idx]
	amps = np.abs(part)
	rand = np.random.uniform(-np.pi, np.pi, len(part))
	angs = np.exp(1j * rand)
	fnew[idx] = amps * angs

	return fnew


def roll_data(data, tts):
	droll = np.zeros_like(data)

	for i, sig in enumerate(data):
		droll[i] = np.roll(sig, -tts[i])
	return droll


def comb_channels(data, cmap):
	
	groups = [np.where(sk == cmap)[0] for sk in np.unique(cmap)]
	dstack = np.zeros((len(groups), data.shape[1]))

	for i, grp in enumerate(groups):
		dstack[i] = np.mean(np.abs(data[grp]), axis=0)

	return dstack

	
def mlab_coords(locs, lims, spacing):
	return (locs - lims[:, 0]).T / spacing


def SearchClusters(data, dmin, zmin=0):

	inds = np.arange(data.shape[1])

	slocs = []
	vals = []
	for j, ix in enumerate(inds):
		print(j)
		lmax = data[:-1, ix, 1:]
		tmp = []
		for k, centroid in enumerate(lmax):
			if centroid[2] < zmin:
				tmp.append(0)
				continue
			diff = np.linalg.norm(lmax - centroid, axis=-1)
			tmp.append(np.where(diff < dmin)[0].size)

		imax = np.argmax(tmp)
		vals.append(tmp[imax])
		slocs.append(lmax[imax])

	vals = np.array(vals)
	slocs = np.array(slocs)

	return vals, slocs


def MeanDist(data):

	inds = np.arange(data.shape[1])
	vals = []
	for j, ix in enumerate(inds):
		print(j)
		lmax = data[:-1, ix, 1:]
		x, y, z = lmax.T
		err = np.std(x) + np.std(y) + np.std(z)
		vals.append(err)

	vals = np.array(vals) / 3

	return vals


def shift_locs(locs, unshift=False, vals=np.array([1.79236297e+05, 7.09943400e+06, 2.49199997e+02])):
	vals = np.array(vals)
	locs[:, 2] *= -1
	if unshift is True:
		return locs + vals
	else:
		return locs - vals


def shift_locs_ot(locs, unshift=False, vals=np.array([650000., 4766000., 0]), zdepth=1200.):
	vals = np.array(vals)
	lnew = locs.copy()

	if unshift is True:
		# lnew[:, 2] *= -1
		lnew.T[2] = (lnew.T[2] - zdepth) * -1
		return lnew + vals
	else:
		lnew.T[2] = zdepth - lnew.T[2]
		return lnew - vals

# def shift_locs_ot(locs, unshift=False, vals=np.array([650000., 4766000., 0])):
# 	vals = np.array(vals)
# 	if unshift is True:
# 		return locs + vals
# 	else:
# 		return locs - vals


def normVec(v):
	return v / norm(v)


def norm(v):
	return math.sqrt(np.dot(v, v))


def angle_between_vecs(v1, v2):
	return math.acos(np.dot(v1, v2) / (norm(v1) * norm(v2)))


def dist(l1, l2):
	return norm(l1 - l2)


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
		# print(fn)
		# xl, yl, zl = lims.reshape(3, 2)
		grid = np.load(fn).reshape(nz, ny, nx)
		grids[i] = grid

	return grids


def combine_grids_nll(fles, shape):

	nx, ny, nz = shape
	nfle = len(fles)
	grids = np.zeros((nfle, nx, ny, nz), dtype=np.float32)

	for i, fn in enumerate(fles):
		grids[i] = np.load(fn).reshape(shape)

	return grids


def xyz_max(grid, lims, spacing):
	# thresh = np.std(grid) * nstd
	iwin = np.argmax(grid)

	pt = np.array(np.unravel_index(iwin, grid.shape))[::-1]
	pt = pt * spacing + lims[:, 0]
	return pt


def xyz_index(loc, lims, spacing):
	# thresh = np.std(grid) * nstd
	shape = (np.diff(lims).T[0] / spacing).astype(int)
	nx, ny, nz = shape
	# x, y, z = loc
	inds = ((loc - lims[:, 0]) / spacing).astype(int)
	ix, iy, iz = inds
	return (iz * nx * ny) + (iy * nx) + ix


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


# def xcorr_freq(sig1f, sig2f):
# 	"""Cross-correlate two signals."""
	
# 	ccf = np.conj(sig1f) * sig2f

# 	if phat:
# 		ccf = ccf / np.abs(ccf)

# 	cc = np.real(ifft(ccf))

# 	if norm:
# 		cc /= np.sqrt(energy(sig1) * energy(sig2))

# 	return np.roll(cc, len(cc) // 2)


def energy(sig, axis=None):
	return np.sum(sig ** 2, axis=axis)


def energy_freq(fsig, axis=None):
	return np.sum(np.abs(fsig) ** 2, axis=axis) / fsig.shape[-1]


def build_slice_inds(start, stop, wlen, stepsize=None):

	if stepsize is None:
		stepsize = wlen

	overlap = wlen - stepsize
	imin = np.arange(start, stop - overlap - 1, stepsize)
	imax = np.arange(start + wlen, stop + stepsize - 1, stepsize)

	return np.dstack((imin, imax))[0]


def freq_window(cf, npts, sr):
	nfreq = int(npts // 2 + 1)
	fsr = npts / sr
	cf = np.array(cf, dtype=float)
	cx = (cf * fsr + 0.5).astype(int)

	win = np.zeros(nfreq, dtype=np.float32)
	win[:cx[0]] = 0
	win[cx[0]:cx[1]] = taper_cosine(cx[1] - cx[0])
	win[cx[1]:cx[2]] = 1
	win[cx[2]:cx[3]] = taper_cosine(cx[3] - cx[2])[::-1]
	win[cx[-1]:] = 0
	return win


def taper_cosine(wlen):
	return np.cos(np.linspace(np.pi / 2., np.pi, wlen)) ** 2


def phase(sig):
	return np.exp(1j * np.angle(sig))


def whiten2D(a, freqs, sr):
	wl = a.shape[1]
	win = freq_window(freqs, wl, sr)
	af = fft(a)
	for sx in range(a.shape[0]):
		whiten_freq(af[sx], win)
	a[:] = np.real(ifft(af))


def whiten(sig, win):
	"""Whiten signal, modified from MSNoise."""
	npts = len(sig)
	nfreq = int(npts // 2 + 1)

	assert(len(win) == nfreq)
	# fsr = npts / sr

	fsig = fft(sig)
	# hl = nfreq // 2

	half = fsig[: nfreq]
	half = win * phase(half)
	fsig[: nfreq] = half
	fsig[-nfreq + 1:] = half[1:].conjugate()[::-1]

	return np.real(ifft(fsig))


def whiten_freq(fsig, win):
	# npts = len(fsig)
	# nfreq = int(npts // 2 + 1)
	nfreq = int(len(fsig) // 2 + 1)
	assert(len(win) == nfreq)
	fsig[: nfreq] = win * phase(fsig[: nfreq])
	fsig[-nfreq + 1:] = fsig[1: nfreq].conjugate()[::-1]


def mirror_freqs(data):
	nfreq = int(data.shape[1] // 2 + 1)
	data[:, -nfreq + 1:] = np.fliplr(data[:, 1: nfreq].conjugate())


def taper_data(data, wlen):
	tap = taper_cosine(wlen)
	data[:wlen] *= tap
	data[-wlen:] *= tap[::-1]


def taper2d(data, wlen):
	tap = taper_cosine(wlen)
	for i in range(data.shape[0]):
		data[i][:wlen] *= tap
		data[i][-wlen:] *= tap[::-1]


def amax_cc(sig):
	return np.argmax(sig) - len(sig) // 2


def angle(a, b):
	# return np.arctan((a[1] - b[1]) / (a[0] - b[0]))
	return np.arctan2((a[1] - b[1]), (a[0] - b[0]))


def fftnoise(f):
	f = np.array(f, dtype='complex')
	Np = (len(f) - 1) // 2
	phases = np.random.rand(Np) * 2 * np.pi
	phases = np.cos(phases) + 1j * np.sin(phases)
	f[1:Np + 1] *= phases
	f[-1:-1 - Np:-1] = np.conj(f[1:Np + 1])
	return ifft(f).real


def band_noise(freqmin, freqmax, samples, sr):
	freqs = np.abs(fftfreq(samples, 1. / sr))
	f = np.zeros(samples)
	idx = np.where(np.logical_and(freqs >= freqmin, freqs <= freqmax))[0]
	f[idx] = 1
	return fftnoise(f)


def add_noise(data, band, sr, power):
	freqmin, freqmax = band
	samples = data.shape[1]
	for d in data:
		d += band_noise(freqmin, freqmax, samples, sr) * power


# def pairs_excluding_acoustic():
		# fd = rfft(sig)
	# # keys = np.arange(ds.shape[0])
	# # ckeys = np.array(list(itertools.combinations(keys, 2))).astype(np.int32)
	# ckeys = ck
	# scut = 700
	# avel = 345.
	# sloc = srcs[0]

	# dd0 = np.array([xutil.dist_between(l, sloc) for l in locs])
	# dd1 = np.diff(dd0[ckeys], axis=1)
	# ia = dd1 / avel * sr
	# # plt.hist(np.abs(ia), bins=100)
	# ik0 = np.where(np.abs(ia) < scut)[0]
	# sloc = srcs[1]
	# dd0 = np.array([xutil.dist_between(l, sloc) for l in locs])
	# dd1 = np.diff(dd0[ckeys], axis=1)
	# ia = dd1 / avel * sr
	# # plt.hist(np.abs(ia), bins=100)
	# ik1 = np.where(np.abs(ia) < scut)[0]
	# ik = np.unique(np.concatenate((ik0, ik1)))
	# ck2 = np.delete(ckeys, ik, axis=0)
	# xplot.stations(locs, ckeys=ck2[::500])
	# dd = np.linalg.norm(np.diff(locs[ck2], axis=1).reshape(-1, 3), axis=1)
	# # plt.hist(dd, bins=100)
	# cki = np.where(dd < 1000)[0]
