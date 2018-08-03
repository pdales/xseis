""" utils."""

import numpy as np
# import math
# from scipy.fftpack import fft, ifft, fftfreq
# import os
# import pickle
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# import subprocess
# import h5py
# import glob
# from xseis import xutil


def stream_to_buffer(st, t0, npts):

	sr = st[0].stats.sampling_rate
	nsig = len(st)
	npts = 6000
	data = np.zeros((nsig, npts), dtype=np.float32)
	for i, tr in enumerate(st):
		i0 = int((tr.stats.starttime - t0) * sr + 0.5)
		sig = tr.data
		slen = min(len(sig), npts)
		data[i, i0: i0 + slen] = sig[:slen]
	return data

