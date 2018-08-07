# distutils: language = c++
# Cython interface file for wrapping the object
cimport numpy as np
from libcpp.vector cimport vector
from libcpp cimport bool
from libc.stdint cimport uint32_t, int64_t

# cdef extern from "xseis/structures.h" namespace "structures":
cdef extern from "xseis/structures.h":
	ctypedef struct Arr_f "Array2D<float>":
		float *data_
		size_t nrow_
		size_t ncol_		
	ctypedef struct Arr_ui16 "Array2D<uint16_t>":
		unsigned short *data_
		size_t nrow_
		size_t ncol_
	ctypedef struct Vec_f "Vector<float>":
		float *data_
		size_t size_
	ctypedef struct Vec_ui16 "Vector<uint16_t>":
		unsigned short *data_
		size_t size_


def fCorrSearchDec2X(np.ndarray[np.float32_t, ndim=2] data,
					sr,
					np.ndarray[np.float32_t, ndim=2] stalocs,
					np.ndarray[np.uint16_t, ndim=1] chanmap,
					np.ndarray[np.uint16_t, ndim=2] ttable,
					np.ndarray[np.uint32_t, ndim=1] outbuf,
					np.ndarray[np.float32_t, ndim=1] output,
					nthreads
			   ):

	return CorrSearchDec2X(Arr_f(&data[0, 0], data.shape[0], data.shape[1]),
					sr,
					Arr_f(&stalocs[0, 0], stalocs.shape[0], stalocs.shape[1]),
					Vec_ui16(&chanmap[0], chanmap.shape[0]),					
					Arr_ui16(&ttable[0, 0], ttable.shape[0], ttable.shape[1]),
					&outbuf[0],
					Vec_f(&output[0], output.shape[0]),
					nthreads
					)



def fSearch(np.ndarray[np.float32_t, ndim=2] data,
			  np.ndarray[np.float32_t, ndim=2] stalocs,
			  np.ndarray[np.uint16_t, ndim=1] chanmap,
			  np.ndarray[int, ndim=1] tmeta,
			  np.ndarray[np.uint16_t, ndim=2] ttable,
			  np.ndarray[np.float32_t, ndim=1] outbuf,
			   ):

	return Search(Arr_f(&data[0, 0], data.shape[0], data.shape[1]),
					Arr_f(&stalocs[0, 0], stalocs.shape[0], stalocs.shape[1]),
					Vec_ui16(&chanmap[0], chanmap.shape[0]),
					&tmeta[0],
					Arr_ui16(&ttable[0, 0], ttable.shape[0], ttable.shape[1]),
					&outbuf[0],
					)


cdef extern from "xseis/interloc.h" namespace "interloc":

	void Search(Arr_f& data, Arr_f& stalocs, Vec_ui16& chanmap, int* tmeta, Arr_ui16& ttable, float *outbuf)

	void CorrSearchDec2X(Arr_f& data, float sr, Arr_f& stalocs, Vec_ui16& chanmap, Arr_ui16& ttable, uint32_t *outbuf, Vec_f& output, uint32_t nthreads)


