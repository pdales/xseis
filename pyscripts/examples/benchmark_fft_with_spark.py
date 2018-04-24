"""
Benchmarking Spark by ffting a collection of signals
"""

from pyspark import SparkConf, SparkContext
import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack

# sc = SparkContext("local[4]", "test")
sc = SparkContext("local[1]", "test")

nsig, npts = 384, 6000
# nsig, npts = 1000, 6000
data = np.random.rand(nsig * npts).reshape(nsig, npts).astype(np.float32)

# Benchmarks to compare our Spark fft with
# Scipy rfft performance comparable to fftw3 with no patience
# fft 2d array at once
%timeit fftpack.rfft(data)
# 11.3 ms ± 34.7 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

# fft row-by-row, better comparison to Spark
%timeit [fftpack.rfft(sig) for sig in data]
# 13.7 ms ± 37.1 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)


# parallize the keys 
keys = np.arange(data.shape[0])
rdd = sc.parallelize(keys)

# First attempt: using broadcast so data not 
# copied to each map function
data_bc = sc.broadcast(data)
func = lambda k: fftpack.rfft(data_bc.value[k])
%timeit output = rdd.map(func).collect()
# 133 ms ± 17.1 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# Second attempt: trying to get this as fast as possible
# by not returning anything 
%timeit rdd.foreach(func)
# 84.1 ms ± 3.51 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)


keys = np.arange(data.shape[0])
rdd = sc.parallelize(keys)

# Naive attempt: not broadcasting
# data_bc = sc.broadcast(data)
rdd = sc.parallelize(data)
func = lambda sig: fftpack.rfft(sig)
%timeit output = rdd.map(func).collect()
# 147 ms ± 8.65 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

