# distutils: language = c++
# Cython interface file for wrapping the object
# import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp cimport bool



# void search(float *rdat_p, uint32_t nchan, uint32_t npts, float* stalocs_p, uint16_t* chanmap_p, int* tmeta, uint16_t* ttable_p, float *outbuf, uint32_t nsta, uint32_t ngrid) 


# cdef extern from "../include/xseis/iloc.h" namespace "iloc":
cdef extern from "../include/xseis/iloc.h":
# cdef extern from "xseis/iloc.h" namespace "iloc":

	# void search(Arr_f data, Arr_f stalocs, Vec_ui16 chan_map, int* tmeta, Arr_ui16 ttable, float *outbuf)
	void search(float *rdat_p, unsigned nchan, unsigned npts, float* stalocs_p, unsigned nsta, unsigned short* chanmap_p, int* tmeta, unsigned short* ttable_p, unsigned ngrid, float *outbuf)

	# void search2(int *rdat_p, unsigned nchan, unsigned npts)

def search_py(np.ndarray[np.float32_t, ndim=2] data,
			  np.ndarray[np.float32_t, ndim=2] stalocs,
			  np.ndarray[np.uint16_t, ndim=1] chan_map,
			  np.ndarray[int, ndim=1] tmeta,
			  np.ndarray[np.uint16_t, ndim=2] ttable,
			  np.ndarray[np.float32_t, ndim=1] outbuf,
			   ):

	# return search2(&tmeta[0], data.shape[0], data.shape[1])
		

	return search(&data[0, 0], data.shape[0], data.shape[1],
					&stalocs[0, 0], stalocs.shape[0],
					&chan_map[0],
					&tmeta[0],
					&ttable[0, 0], ttable.shape[1],
					&outbuf[0]
					)



# cdef extern from "xseis/structures.h" namespace "structures":
# 	ctypedef struct Arr_f "structures::Array2D<float>":
# 		float *data
# 		size_t nrow
# 		size_t ncol
# 	ctypedef struct Arr_ui16 "structures::Array2D<uint16_t>":
# 		unsigned short *data
# 		size_t nrow
# 		size_t ncol
# 	ctypedef struct Vec_f "structures::Vector<float>":
# 		float *data
# 		size_t size
# 	ctypedef struct Vec_ui16 "structures::Vector<uint16_t>":
# 		unsigned short *data
# 		size_t size


# search(Array2D<float> rdat, Array2D<float> stalocs, Vector<uint16_t> chan_map, int* tmeta, Array2D<uint16_t> ttable, float *outbuf)

# cdef extern from "xseis/iloc.h" namespace "iloc":

# 	void search(Arr_f data, Arr_f stalocs, Vec_ui16 chan_map, int* tmeta, Arr_ui16 ttable, float *outbuf)


# def search_py(np.ndarray[np.float32_t, ndim=2] data,
# 			  np.ndarray[np.float32_t, ndim=2] stalocs,
# 			  np.ndarray[np.uint16_t, ndim=1] chan_map,
# 			  np.ndarray[np.int, ndim=1] tmeta,
# 			  np.ndarray[np.uint16_t, ndim=2] ttable,
# 			  np.ndarray[np.float32_t, ndim=1] outbuf,
# 			   ):

# 	return search(Arr_f(&data[0, 0], data.shape[0], data.shape[1]),
# 					Arr_f(&stalocs[0, 0], stalocs.shape[0], stalocs.shape[1]),
# 					Vec_ui16(&chan_map[0], chan_map.shape[0]),
# 					&tmeta[0],
# 					Arr_ui16(&ttable[0, 0], ttable.shape[0], ttable.shape[1]),
# 					&outbuf[0],
# 					)


# def beampower_homo_py(np.ndarray[np.float32_t, ndim=2] points,
# 							np.ndarray[np.float32_t, ndim=1] out,
# 							np.ndarray[np.float32_t, ndim=2] data,
# 							np.ndarray[np.float32_t, ndim=1] tts,
# 							np.ndarray[np.float32_t, ndim=2] sta_locs,
# 							np.ndarray[int, ndim=1] ix_keep,
# 							velocity, src_time, sr):

# 	return beampower_homo(Arr_f(&points[0, 0], points.shape[0], points.shape[1]),
# 						Vec_f(&out[0], out.shape[0]),
# 						Arr_f(&data[0, 0], data.shape[0], data.shape[1]),
# 						Vec_f(&tts[0], tts.shape[0]),
# 						Arr_f(&sta_locs[0, 0], sta_locs.shape[0], sta_locs.shape[1]),
# 						Vec_i(&ix_keep[0], ix_keep.shape[0]),
# 						velocity,
# 						src_time,
# 						sr)

# def slant_stack_py(np.ndarray[np.float32_t, ndim=2] data, np.ndarray[np.float32_t, ndim=1] tts, np.ndarray[int, ndim=1] ix_keep, src_time, sr):

# 	return slant_stack(&data[0, 0], data.shape[1], &tts[0], &ix_keep[0], ix_keep.shape[0], src_time, sr)
