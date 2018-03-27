"""
Plotting
"""

# from matplotlib.colors import LogNorm
# from matplotlib.ticker import ScalarFormatter
import matplotlib.pyplot as plt
# from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from scipy import fftpack

from matplotlib.pyplot import rcParams
rcParams['figure.figsize'] = 10, 6


def v2color(vals):

	cnorm  = plt.Normalize(vmin=np.nanmin(vals), vmax=np.nanmax(vals))
	cmap = plt.cm.ScalarMappable(norm=cnorm, cmap=plt.get_cmap('viridis'))
	clrs = [cmap.to_rgba(v) for v in vals]
	return clrs


def plot_stations(locs, ckeys=None, vals=None, alpha=0.3, lstep=100):
	locs = locs[:, :2]
	x, y = locs.T
	plt.scatter(x, y, alpha=0.5, s=6)
	# x, y, z = locs[2900:3100].T
	for i in range(0, locs.shape[0], lstep):
		plt.text(x[i], y[i], i)

	if ckeys is not None:
		if vals is not None:
			clrs = v2color(vals)
			for i, ck in enumerate(ckeys):
				x, y = locs[ck].T
				plt.plot(x, y, alpha=alpha, color=clrs[i], linewidth=2)
		else:
			for ck in ckeys:
				x, y = locs[ck].T
				plt.plot(x, y, alpha=alpha, color='black')
	plt.axis('equal')
	plt.show()


def im_freq(d, sr, norm=False, xlims=None):

	fd = fftpack.rfft(d, axis=1)
	fd = np.abs(fd)

	if norm is True:
		fd /= np.max(fd, axis=1)[:, np.newaxis]

	n = fd.shape[1]
	freq = fftpack.rfftfreq(n, d=1. / sr)

	plt.imshow(fd, aspect='auto', extent=[freq[0], freq[-1], 0, fd.shape[0]], origin='lower', interpolation='none')
	if xlims is not None:
		plt.xlim(xlims)
	plt.show()


def im(d, norm=True, savedir=None, tkey='im_raw', cmap='viridis', aspect='auto', extent=None, locs=None, labels=None):

	fig = plt.figure(figsize=(14, 9), facecolor='white')
	# if times is not None:
	# 	extent = [times[0], times[-1], 0, d.shape[0]]

	if norm is True:
		dtmp = d / np.max(np.abs(d), axis=1)[:, np.newaxis]
	else:
		dtmp = d
	im = plt.imshow(dtmp, origin='lower', aspect=aspect, extent=extent,
			   cmap=cmap, interpolation='none')
	if extent is not None:
		plt.xlim(extent[:2])
		plt.ylim(extent[2:])
	if locs is not None:
		plt.scatter(locs[:, 0], locs[:, 1])
	plt.colorbar(im)
	if labels is not None:
		plt.xlabel(labels[0])
		plt.ylabel(labels[1])
	# manager = plt.get_current_fig_manager()
	# manager.resize(*manager.window.maxsize())
	plt.tight_layout()
	savefig(fig, savedir, tkey)


def freq_compare(sigs, sr, xlim=None):

	plt.subplot(211)
	for sig in sigs:
		plt.plot(sig)
	plt.xlabel('Time')
	plt.subplot(212)
	for sig in sigs:
		f = fftpack.fft(sig)
		freq = fftpack.fftfreq(len(f), d=1. / sr)
		# freq = np.fft.fftshift(freq)
		plt.plot(np.fft.fftshift(freq), np.fft.fftshift(np.abs(f)))
	if xlim is not None:
		plt.xlim(xlim)
	else:
		plt.xlim([0, sr / 2.])
	plt.xlabel('Freq (Hz)')
	plt.show()


def freq(sig, sr, xlim=None):

	plt.subplot(211)
	plt.plot(sig)
	plt.xlabel('Time')
	plt.subplot(212)
	f = fftpack.fft(sig)
	freq = fftpack.fftfreq(len(f), d=1. / sr)
	# freq = np.fft.fftshift(freq)

	plt.plot(np.fft.fftshift(freq), np.fft.fftshift(np.abs(f)))
	if xlim is not None:
		plt.xlim(xlim)
	else:
		plt.xlim([0, sr / 2.])
	plt.xlabel('Freq (Hz)')
	plt.show()


def sigs(d, spacing=10, labels=None, vlines=None):

	if vlines is not None:
		for v in vlines:
			plt.axvline(v, linestyle='--', color='red')

	std = np.std(d)
	shifts = np.arange(0, d.shape[0], 1) * spacing * std
	for i, sig in enumerate(d):
		plt.plot(sig + shifts[i])

	if labels is not None:
		for i, lbl in enumerate(labels):
			plt.text(0, shifts[i] + 2 * std, lbl, fontsize=15)

	plt.show()


def savefig(fig, savedir, tkey, dpi=100, facecolor='white', transparent=False):

	if savedir is not None:
		fname = tkey + '.png'
		fpath = os.path.join(savedir, fname)
		# plt.savefig(fpath, dpi=dpi, facecolor=facecolor, transparent=transparent, edgecolor='none')
		fig.savefig(fpath, dpi=dpi)
		plt.close()
	else:
		# fig.show()
		plt.show()

	# plt.close('all')


def set_axes_equal(ax):
	'''Make axes of 3D plot have equal scale so that spheres appear as spheres,
	cubes as cubes, etc..  This is one possible solution to Matplotlib's
	ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

	Input
	  ax: a matplotlib axis, e.g., as output from plt.gca().
	'''

	x_limits = ax.get_xlim3d()
	y_limits = ax.get_ylim3d()
	z_limits = ax.get_zlim3d()

	x_range = abs(x_limits[1] - x_limits[0])
	x_middle = np.mean(x_limits)
	y_range = abs(y_limits[1] - y_limits[0])
	y_middle = np.mean(y_limits)
	z_range = abs(z_limits[1] - z_limits[0])
	z_middle = np.mean(z_limits)

	# The plot bounding box is a sphere in the sense of the infinity
	# norm, hence I call half the max range the plot radius.
	plot_radius = 0.5 * max([x_range, y_range, z_range])

	ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
	ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
	ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
