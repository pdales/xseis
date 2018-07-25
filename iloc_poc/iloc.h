// #include <fftw3.h>
#include <cstring>
// #include <cstddef>

// #include <omp.h>
// #include <fftw3.h>

// #include "xseis/utils.h"
// #include "xseis/structures.h"
// #include "xseis/process.h"
// #include "xseis/beamform.h"
// #include "xseis/npy.h"
// #include "xseis/keygen.h"
// #include "xseis/mseed.h"

// #include "xseis/h5wrap.h"
// #include "xseis/raytrace.h"
// #include "xseis/FftHandler.h"


void searchMSEED(char* fbuf, size_t nbytes,
				char* stas, size_t nsta,
				float* sta_locs, int* tmeta,
				float *tt_ptr, float *outbuf); 

