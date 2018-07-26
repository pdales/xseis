import numpy as np
import os
# import pickle
import matplotlib.pyplot as plt
# import subprocess
# import h5py
import glob
# import gridutil as gut
from importlib import reload
# import matplotlib.gridspec as gridspec
# from mayavi import mlab
import datetime
# from obspy.core.event import read_events
from obspy import read
# from obspy import Catalog
import itertools
import csig

# plt.ion()


def read_hdr(fle):
	print(fle)
	dat = open(fle).read().split()
	shape = np.array(dat[:3], dtype=int)
	org = np.array(dat[3:6], dtype=np.float32) * 1000.
	spacing = (np.array(dat[6:9], dtype=np.float32) * 1000.)[0]
	sloc = np.array(dat[12:15], dtype=np.float32) * 1000.

	return sloc, shape, org, spacing


mseed_fle = '20180523_185101_float.mseed'
syncdir = 'NLLOC_grids/'

fles = np.sort(glob.glob(syncdir + '*P*time.buf'))
hfles = np.sort(glob.glob(syncdir + '*P*time.hdr'))

stas = np.array([f.split('.')[-3].zfill(3) for f in fles], dtype='S4')
isort = np.argsort(stas)
fles = fles[isort]
hfles = hfles[isort]
names = stas[isort]

vals = [read_hdr(fle) for fle in hfles]
sloc, shape, org, spacing = vals[0]
locs = np.array([v[0] for v in vals])
ngrid = np.product(shape)

nsta = len(fles)
tts_p = np.zeros((nsta, ngrid), dtype=np.float32)

for i in range(nsta):
	gdata = np.fromfile(fles[i], dtype='f4')
	tts_p[i] = gdata

gdef = np.concatenate((shape, org, [spacing])).astype(np.int32)

sdict = {}
for i, sk in enumerate(names):
	sdict[sk.decode('utf-8')] = i


st = read(mseed_fle)
st.sort(keys=['starttime'])
t0 = st[0].stats.starttime
for tr in st:
	tr.stats.station = tr.stats.station.zfill(3)
st.sort()

names = np.array([tr.stats.station for tr in st])
kvalid = np.unique(names)
sd = {}
for i, sk in enumerate(kvalid):
	sd[sk] = i

chanmap = []
for chan in names:
	chanmap.append(sd[chan])
chanmap = np.array(chanmap, dtype=np.uint16)

#########################################################

ksta = []
for i, sk in enumerate(kvalid):
	ksta.append(sdict[sk])
slocs = locs[ksta]


sr = st[0].stats.sampling_rate
nsig = len(st)
npts = 6000
a = np.zeros((nsig, npts), dtype=np.float32)
for i, tr in enumerate(st):
	i0 = int((tr.stats.starttime - t0) * sr + 0.5)
	sig = tr.data
	slen = min(len(sig), npts)
	a[i, i0: i0 + slen] = sig[:slen]

# xplot.sigs(a)

dsr = 3000.
tts = (tts_p[ksta] * dsr).astype(np.uint16)
outbuf = np.zeros(5, dtype=np.float32)

csig.search_py(a, slocs, chanmap, gdef, tts, outbuf)

epoch_ref = datetime.datetime(1970, 1, 1)
# t0 = datetime(2015, 6, 26, 13, 30)
# t0 = datetime(2015, 6, 28, 0, 0)
epoch = (t0.datetime - epoch_ref).total_seconds() * 1e6
outbuf[4] = (outbuf[4] / dsr * 1e6) + epoch

# def search_py(np.ndarray[np.float32_t, ndim=2] data,
# 			  np.ndarray[np.float32_t, ndim=2] stalocs,
# 			  np.ndarray[np.uint16_t, ndim=1] chan_map,
# 			  np.ndarray[int, ndim=1] tmeta,
# 			  np.ndarray[np.uint16_t, ndim=2] ttable,
# 			  np.ndarray[np.float32_t, ndim=1] outbuf,
# 			   ):
