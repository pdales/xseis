"""
Plotting
"""

# from matplotlib.colors import LogNorm
# from matplotlib.ticker import ScalarFormatter
import matplotlib.pyplot as plt
# from matplotlib.cm import ScalarMappable
# from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
# from scipy import fftpack
from mayavi import mlab
from datetime import datetime
from subprocess import call
import matplotlib.gridspec as gridspec
# from mpl_toolkits.axes_grid1 import make_axes_locatable
import xutil


def mlab_contour(g, ncont=10, vfmin=0.1, vfmax=0.1, ranges=None, vlims=None, cbar=True):
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
	if cbar:
		mlab.colorbar(orientation='vertical')
	return src
	# print(mlab.view(), mlab.roll())


def create_movie(dirpath, fps=5, crf=20):
	cwd = os.getcwd()
	os.chdir(dirpath)

	cmd = 'ffmpeg -framerate %d -i %%00d.png -c:v libx264 \
		   -profile:v high -crf %d -pix_fmt yuv420p output.mp4' % (fps, crf)

	call(cmd, shell=True)
	os.chdir(cwd)


def time_stamp():
	return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def mlab_patches(data, locs, pix, ix):

	mlab.clf()
	lkeep = locs[pix]
	maxes = data[:-1, ix, 0]
	locmax = data[:-1, ix, 1:]
	locsum = data[-1, ix, 1:]

	# fig = mlab.figure(size=(800, 751))
	# x, y, z = np.concatenate((locs[:100], locs[-100:])).T
	# x, y, z = np.concatenate((locs[:500], locs[-500:])).T
	# mlab.points3d(z, y, x, color=(0.2, 0.2, 0.2), scale_factor=40, mode='cone')
	# mlab.points3d(z, y, x, color=(0.2, 0.2, 0.2), scale_factor=40, mode='cone')
	l1 = locs[::50]
	l2 = l1.copy()
	l2[:, -1] += 2000
	x, y, z = np.concatenate((l1, l2)).T
	mlab.points3d(z, y, x, color=(0.2, 0.2, 0.2), scale_factor=40, opacity=0.1)
	mlab.outline()
	mlab.axes(line_width=0.5, xlabel='Z', ylabel='Y', zlabel='X')

	x, y, z = locmax.T
	mlab.points3d(z, y, x, maxes / np.max(maxes),
		 colormap='viridis', scale_factor=100, scale_mode='none')
	# mlab.points3d(z, y, x, maxes, colormap='viridis')

	x, y, z = locsum.T
	mlab.points3d(z, y, x, color=(1, 0, 0), scale_factor=100, mode='cube')
	for i, loc in enumerate(lkeep):
		x, y, z = np.stack((loc, locmax[i])).T
		mlab.plot3d(z, y, x, tube_radius=5, color=(0.5, 0.5, 0.5))
		# ln = mlab.plot3d(z, y, x, color=(1, 1, 1), line_width=10)
	return mlab


def add_imshow(ax, gdata, lims, vmin=None, vmax=None):

	xl, yl, zl = lims.reshape(3, 2)
	extent = [xl[0], xl[1], yl[0], yl[1]]

	im = ax.imshow(gdata, origin='lower', aspect='equal', extent=extent,
			 cmap='viridis', interpolation='none', vmin=vmin, vmax=vmax)
	return im


class PatchVecs:
	def __init__(self, data, locs, pix, inds, lims, msize=200, times=None, label=False, keep=False, lines=True):
		
		self.data = data
		self.locs = locs
		self.pix = pix
		self.maxvals = data[:, :, 0]
		self.maxlocs = data[:, :, 1:4]
		self.label = label
		self.keep = keep
		self.lines = lines

		self.inds = inds
		self.npair = len(inds)
		self.pos = 0
		self.msize = msize
		self.lims = lims
		self.xlim = lims[0]
		self.ylim = lims[1]

		if times is None:
			self.times = np.arange(data.shape[1])
		else:
			self.times = times

		self.dline = None
		self.dline2 = None
		self.screen_dir = "/home/phil/Pictures/glen/captures/"
		# self.cb = None
		# self.xs = []
		# self.ys = []
		self.fig = None
		self.ax = None
		self.ax2 = None
		self.points = []
		self.lines = []

		fig = mlab.figure(size=(800, 751))
		mlab.view(azimuth=-130, elevation=111, distance='auto', focalpoint='auto', roll=29)

	def start(self):
		self.fig_init()
		plt.show()

	@property
	def ix(self):
		return self.inds[self.pos]

	def refresh(self):
		[p.remove() for p in self.points]
		[ln.pop(0).remove() for ln in self.lines]

		# self.ax.clear()
		# self.refresh_ax()
		# self.dline, = self.ax.plot(self.xs, self.ys,
		# 	 color=self.color, marker='o', markersize=5,)
	def movie(self, fps=5):
		mydir = os.path.join(self.screen_dir, time_stamp())
		os.makedirs(mydir)

		self.fig_init()

		for i in range(self.npair):
			print(i)
			self.refresh()
			self.draw(self.ix)
			fpath = os.path.join(mydir, str(i) + '.png')
			self.fig.savefig(fpath, dpi=100)
			self.pos += 1

		create_movie(mydir, fps)

	def switch_trace(self, key):
		
		pos = self.pos

		if key == 'right':
			pos += 1
		elif key == 'left':
			pos -= 1
		if pos >= self.npair or pos < 0:
			print('pair number %d out of bounds' % self.pos)
		else:
			self.pos = pos
			self.refresh()
			self.draw(self.ix)

	def __call__(self, event):
		if event.key in ['left', 'right']:
			self.switch_trace(event.key)
		elif event.key == 'up':
			mlab_patches(self.data, self.locs, self.pix, self.ix)
		elif event.key == 'down':
			fpath = self.screen_dir + time_stamp() + '.png'
			self.fig.savefig(fpath, dpi=100)
			print("saved %s" % fpath)
		elif event.key == 'end':
			self.pos = int(event.xdata)
			self.refresh()
			self.draw(self.ix)
			
		self.dline.figure.canvas.draw()

	# def call2(self, event):
	# 	if event.key in ['left', 'right']:
	# 		pos = int(event.xdata)
	# 		if pos >= self.npair or pos < 0:
	# 			print('pair number %d out of bounds' % self.pos)
	# 		else:
	# 			self.pos = pos
	# 			self.refresh()
	# 			self.draw(self.ix)
	# 	self.dline.figure.canvas.draw()

	def fig_init(self):
		self.fig = plt.figure(figsize=(10, 9))
		gs = gridspec.GridSpec(5, 1)
		ax = plt.subplot(gs[0:4])

		ax.set_xlim(self.xlim)
		ax.set_ylim(self.ylim)
		self.ax = ax
		# self.refresh()
		x, y, z = self.locs.T
		self.dline = self.ax.scatter(x, y, s=5, alpha=0.1)
		plt.axis('equal')
		self.cid = self.dline.figure.canvas.mpl_connect('key_press_event', self)

		ax2 = plt.subplot(gs[4])
		self.dline2 = ax2.scatter(self.times, self.maxvals[-1], s=5)
		ax2.set_xlabel('Window #')
		ax2.set_ylabel('Max grid val')
		self.ax2 = ax2
		self.draw(self.ix)
		# self.cid2 = self.dline2.figure.canvas.mpl_connect('key_press_event', self.call2)

	def draw(self, ix, alpha=0.9):
		data = self.data
		lkeep = self.locs[self.pix]
		ax = self.ax
		lines = []
		points = []

		ax.set_title("ix: %d  (%d / %d)  tag: %d" % (ix, self.pos + 1, len(self.inds), self.times[ix]))
		mxs = data[:-1, ix, 0]
		lmax = data[:-1, ix, 1:]
		lsum = data[-1, ix, 1:]

		# if self.lines:
		for i, loc in enumerate(lkeep):
			x, y, z = np.stack((loc, lmax[i])).T
			lines.append(ax.plot(x, y, alpha=0.2, color='black'))
		x, y, z = lmax.T
		# points.append(plt.scatter(x, y, , color='black'))
		if self.msize is None:
			sc = ax.scatter(x, y, s=mxs[:-1] * 20, alpha=alpha, c=mxs, cmap='viridis')
		else:
			sc = ax.scatter(x, y, s=self.msize, alpha=alpha, c=mxs, cmap='viridis')
		if self.label:
			for i, loc in enumerate(lmax):
				x, y, z = loc
				points.append(ax.text(x, y, i))

		cb = self.fig.colorbar(sc, ax=ax, orientation='vertical')
		points.append(cb)
		points.append(sc)
		x, y, z = lsum
		if self.keep:
			ax.scatter(x, y, s=80, marker='x')
		else:
			points.append(ax.scatter(x, y, s=80, marker='x'))

		self.lines = lines
		self.points = points
		vl = self.ax2.axvline(self.times[ix], color='red', alpha=0.5)
		points.append(vl)
		mx = data[-1, ix, 0]
		points.append(self.ax2.scatter(self.times[ix], mx, c='red'))

