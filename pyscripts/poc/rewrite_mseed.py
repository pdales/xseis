import numpy as np
import os
# import pickle
# import subprocess
# import h5py
# import gridutil as gut
# import matplotlib.gridspec as gridspec
# from mayavi import mlab
# from obspy.core.event import read_events
from obspy import read
# from obspy import Catalog

# plt.ion()

fle = '20180523_185101.mseed'
fle2 = 'as_float.mseed'
st = read(fle)

for tr in st:
	tr.stats.station = tr.stats.station.zfill(3)
	tr.data = tr.data.astype(np.float32)
	tr.stats.mseed.encoding = "FLOAT32"

st.write(fle2, format='MSEED', reclen=4096)
