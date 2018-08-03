from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os
os.environ["CXX"] = "g++-7"


setup(ext_modules=cythonize(Extension(
           "clib",                                # the extension name
           sources=["clib.pyx"],  # the Cython, cpp source
           language="c++",
           extra_compile_args=["-std=c++17", "-lfftw3f", "-lfftw3f_threads", "-lm", "-O3", "-pthread", "-march=native", "-ffast-math", "-lm", "-fopenmp"],
           extra_link_args=["-fopenmp"],
           libraries=['fftw3', 'fftw3f', 'fftw3l', 'fftw3_threads', 'fftw3f_threads', 'fftw3l_threads'],        
           include_dirs=[np.get_include(), 'include'],
      )))