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
	uint32_t nthreads = 4;
	omp_set_num_threads(nthreads);	

	uint nsig = 384;
	uint wlen = 6000;
	float sr = 6000;

	auto data = Array2D<float>(nsig, wlen);

	auto keys = Vector<uint16_t>(nsig);
	keys.arange(0, nsig, 1);
	auto ckeys = beamform::unique_pairs(keys);

	utils::PrintArraySize(data, "data");
	utils::PrintArraySize(ckeys, "ckeys");

	// auto fh = FftHandler(FFTW_ESTIMATE, nthreads);
	auto fh = FftHandler(FFTW_MEASURE, nthreads);
	// auto fh = FftHandler(FFTW_PATIENT, nthreads);

	auto clock = Clock();
	clock.start();

	auto fdata = fh.plan_fwd(data, wlen);
	clock.log("plan fwd FFT");
	// fh.plan_inv(fdata, data);

	// auto pdata = fh.plan_inv(fdata);

	auto fdata_cc = Array2D<fftwf_complex>({ckeys.nrow_, fdata.ncol_});
	auto data_cc = fh.plan_inv_cc(fdata_cc);
	clock.log("plan inv FFT ccs");

	unsigned nfreq = fdata.ncol_;
	std::vector<float> corner_freqs {50, 60, 500, 550};
	auto filter = process::BuildFreqFilter(corner_freqs, nfreq, sr);	
	auto Whiten = [=, &filter] (float (*fsig)[2], unsigned nfreq){
			process::ApplyFreqFilterReplace(fsig, nfreq, filter);};
	// float energy = filter.energy() * 2;

	auto Taper = [=] (float *sig, int npts){process::taper(sig, npts, 50);};

	auto vshift = process::BuildPhaseShiftVec(nfreq, wlen / 2);
	auto PhaseShift = [=, &vshift] (float (*fsig)[2], uint nfreq){
				process::Convolve(fsig, vshift.data(), nfreq);};

	// utils::PrintArraySize(data, "data");
	// utils::PrintArraySize(fdata_cc, "fdata_cc");
	// utils::PrintArraySize(data_cc, "data_cc");

	utils::FillRandFloat(data, -1, 1);
	// utils::print(data);

	clock.start();	
	fh.exec_fwd();


	#pragma omp parallel
	{
		process::ApplyFuncToRows(data, &Taper);

		#pragma omp single
		{
			clock.log("taper");
			fh.exec_fwd();
			clock.log("fft");
		}

		process::ApplyFuncToRows(fdata, &Whiten);

		#pragma omp single
		{
			clock.log("whiten");
		}

		process::XCorrPairs(fdata, ckeys, fdata_cc);

		#pragma omp single
		{
			clock.log("xcorr");
		}

		process::ApplyFuncToRows(fdata_cc, &PhaseShift);

		#pragma omp single
		{
			clock.log("roll ccs");
		}

		#pragma omp single
		{ 		
			fh.exec_inv_cc();
			clock.log("ifft cc");

		}
	}

	clock.print();

	return 0;
}
