// #include <fftw3.h>
#include <cstring>
#include <omp.h>

#include "xseis/utils.h"
#include "xseis/structures.h"
#include "xseis/process.h"
#include "xseis/beamform.h"
#include "xseis/FftHandler.h"
#include "xseis/h5wrap.h"
// #include "xseis/npy.h"
// #include <iostream>
// #include <string>

int main(void) 
{	
	uint nthreads = 4;
	uint nsig = 384;
	// uint wlen = 8192;
	uint wlen = 6000;

	auto data = Array2D<float>(nsig, wlen);

	auto keys = Vector<uint16_t>(nsig);
	keys.arange(0, nsig, 1);
	auto ckeys = beamform::unique_pairs(keys);

	utils::PrintArraySize(data, "data");
	utils::PrintArraySize(ckeys, "ckeys");

	// auto fh = FftHandler(FFTW_ESTIMATE, nthreads);
	// auto fh = FftHandler(FFTW_MEASURE, nthreads);
	auto fh = FftHandler(FFTW_PATIENT, nthreads);

	auto clock = Clock();
	clock.start();

	auto fdata = fh.plan_fwd(data, wlen);
	clock.log("plan fwd FFT");
	// fh.plan_inv(fdata, data);

	// auto pdata = fh.plan_inv(fdata);

	auto fdata_cc = Array2D<fftwf_complex>({ckeys.nrow_, fdata.ncol_});
	auto data_cc = fh.plan_inv_cc(fdata_cc);
	clock.log("plan inv FFT ccs");

	// utils::PrintArraySize(data, "data");
	// utils::PrintArraySize(fdata_cc, "fdata_cc");
	// utils::PrintArraySize(data_cc, "data_cc");

	utils::FillRandFloat(data, -1, 1);
	// utils::print(data);

	clock.start();	
	fh.exec_fwd();
	clock.log("fft");

	#pragma omp parallel num_threads(nthreads)
	{
		process::XCorrPairs(fdata, ckeys, fdata_cc);
	}

	clock.log("xcorr");
	fh.exec_inv_cc();
	clock.log("ifft ccfs");
	clock.print();

	return 0;
}
