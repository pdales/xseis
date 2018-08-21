/*
* @Author: Philippe Dales
* @Date:   2018-07-26 14:26:23
* @Last Modified by:   Philippe Dales
* @Last Modified time: 2018-07-26 14:26:23
*/
#ifndef INTERLOC_H
#define INTERLOC_H

#include <cstring>
#include <omp.h>
#include <fftw3.h>
#include <cstddef>

#include "xseis/structures.h"
#include "xseis/process.h"
#include "xseis/beamform.h"
#include "xseis/keygen.h"
#include "xseis/npy.h"


namespace interloc {


void CorrSearchDec2XBoth(Array2D<float>& rawdat, float sr, Array2D<float>& stalocs, Vector<uint16_t>& chanmap, Array2D<uint16_t>& ttable1, Array2D<uint16_t>& ttable2, uint32_t* outbuf, Vector<float>& grid, uint32_t nthreads, std::string& logdir, int debug) 
{	

	// some processing params hard-coded for now	
	std::vector<float> cfreqs {40, 45, 300, 350}; // whiten corner freqs
	uint32_t tap_len = 50; // nsamp to taper raw signals
	float min_dist = 300; // min interstation pair distance
	float max_dist = 1500; // max interstation pair distance
	uint32_t cc_smooth_len = 10; // nsamp to smooth ccfs

	omp_set_num_threads(nthreads);

	// std::string resdir(OUTDIR);
	auto clock = Clock();

	uint32_t nsta = ttable1.nrow_;
	uint32_t ngrid = ttable1.ncol_;
	uint32_t nchan = rawdat.nrow_;
	
	// Create data buffers ////////////////////////////////////////////////////		
	size_t wlr = rawdat.ncol_;  // wlen of raw data
	size_t flr = wlr / 2 + 1; // nfreq of raw data
	// wlen of 2x decimated and zero padded aligned data wlr=(wlr / dec) * 2
	size_t wl = utils::PadToBytes<float>(wlr);
	size_t fl = wl / 2 + 1; // nfreq

	auto dat = Array2D<float>(nchan, wl); // decimated and zero padded data
	auto fdat = Array2D<fftwf_complex>(nchan, utils::PadToBytes<fftwf_complex>(fl));
	auto tmp = Vector<float>(wl);
	auto ftmp = Vector<fftwf_complex>(fl);

	// int patience = FFTW_MEASURE;
	int patience = FFTW_ESTIMATE;
	fftwf_plan plan_fwd = fftwf_plan_dft_r2c_1d(wl, &tmp[0], &ftmp[0], patience);
	fftwf_plan plan_inv = fftwf_plan_dft_c2r_1d(wl, &ftmp[0], &tmp[0], patience);
	// fftw_export_wisdom_to_filename(const char *filename)
	clock.log("plan fft");

	// utils::PrintArraySize(rawdat, "raw data");
	// utils::PrintArraySize(dat, "data");
	// utils::PrintArraySize(ttable, "ttable");

	// Pre-process ////////////////////////////////////////////////////////////
	// std::vector<float> cfreqs {20, 25, 240, 250};
	auto filter = process::BuildFreqFilter(cfreqs, fl, sr);	
	float energy = filter.energy() * 2;

	// copy, fft, whiten, ifft, taper, copy each 2nd value to dat, fft
	// This step doesnt gain much from parallelization with this data size
	// #pragma omp parallel for
	for(size_t i = 0; i < rawdat.nrow_; ++i) {
		process::Fill(tmp, 0);
		std::copy(rawdat.row(i), rawdat.row(i) + wlr, &tmp[0]); // gaurantee alignment for fftw
		fftwf_execute_dft_r2c(plan_fwd, &tmp[0], &ftmp[0]);
		process::ApplyFreqFilterReplace(&ftmp[0], fl, filter); // apply whiten
		fftwf_execute_dft_c2r(plan_inv, &ftmp[0], &tmp[0]);
		process::taper(&tmp[0], wl, tap_len);

		float* drow = dat.row(i);
		process::Fill(drow, wl, 0);
		for(size_t j = 0; j < wl / 2; ++j) 	drow[j] = tmp[j * 2] / wl; //dec and scale
		fftwf_execute_dft_r2c(plan_fwd, drow, fdat.row(i));
	}
	clock.log("fft");
	// npy::Save(resdir + "d1.npy", dat);

	// Correlate //////////////////////////////////////////////////////////////
	// group similar channels and create cc pairs
	auto keys = utils::arange<uint16_t>(0, nsta);
	auto groups = keygen::GroupChannels(keys, chanmap);	
	auto spairs = keygen::DistFilt(keys, stalocs, min_dist, max_dist);

	size_t npair = spairs.nrow_;
	auto ccdat = Array2D<float>(npair, wl); // buffer for correlations
	assert((uintptr_t) ccdat.row(1) % MEM_ALIGNMENT == 0);
	assert(npair > 10);

	// values to roll ccfs for zero lag in middle (conv in freq domain)
	auto vshift = process::BuildPhaseShiftVec(fl, wl / 2);
	
	clock.start();

	// compute abs-valued cross-correlations (1 ccf per valid station pair)
	#pragma omp parallel
	{
		auto tmp = malloc_cache_align<float>(wl);
		auto ftmp = malloc_cache_align<fftwf_complex>(fl);

		#pragma omp for
		for(size_t i = 0; i < npair; ++i) {
			uint16_t *pair = spairs.row(i);
			float *csig = ccdat.row(i);
			std::fill(csig, csig + wl, 0);

			// sums absolute valued ccfs of all interstation channel pairs
			uint32_t nstack = 0;		
			for(auto&& k0 : groups[pair[0]]) {
				for(auto&& k1 : groups[pair[1]]) {

					process::XCorr(fdat.row(k0), fdat.row(k1), ftmp, fl);
					process::Convolve(&vshift[0], ftmp, fl);
					fftwf_execute_dft_c2r(plan_inv, ftmp, tmp);
					for(size_t j=0; j < wl; ++j) csig[j] += std::abs(tmp[j]);
					nstack++;
				}
			}
			process::Multiply(csig, wl, 1.0 / (nstack * energy)); // normalize
			process::EMA_NoAbs(csig, wl, cc_smooth_len, true); // EMA smoothing
		}
		free(tmp);
		free(ftmp);
	}
	clock.log("xcorr");


	auto out1 = Vector<float>(ngrid);
	beamform::InterLocBlocks(ccdat, spairs, ttable1, out1);
	// auto imax = process::argmax(grid);
	std::cout << "max1: " << out1[process::argmax(out1)] << '\n';

	auto out2 = Vector<float>(ngrid);
	beamform::InterLocBlocks(ccdat, spairs, ttable2, out2);
	std::cout << "max2: " << out2[process::argmax(out2)] << '\n';

	clock.log("search");
	assert(grid.size_ == out1.size_);

	for(size_t i = 0; i < ngrid; ++i) {
		grid[i] = (out1[i] + out2[i]) / 2;
	}

	if(debug == 2) {
		npy::Save(logdir + "grid.npy", grid);		
	}
	

	auto wtt1 = ttable1.copy_col(process::argmax(grid));
	auto wtt2 = ttable2.copy_col(process::argmax(grid));
	// npy::Save(resdir + "tts_win.npy", wtt);
	clock.log("wtt");

	auto stack = Vector<float>(wl);
	auto rolled = Array2D<float>(groups.size(), wl);
	rolled.fill(0);
	// auto k = Vector<float>(wl);
	stack.fill(0);
	float* sig = &tmp[0];

	for(size_t i = 0; i < groups.size(); ++i) {
		int roll1 = static_cast<int>(wtt1[i]);
		int roll2 = static_cast<int>(wtt2[i]);

		for(auto&& k : groups[i]) {
			process::ExpMovingAverage(dat.row(k), wl, 10, true);

			process::Copy(dat.row(k), wl, sig);
			process::Roll(sig, wl, roll1);
			process::Accumulate(sig, &stack[0], wl);
			process::Accumulate(sig, rolled.row(i), wl);

			process::Copy(dat.row(k), wl, sig);
			process::Roll(sig, wl, roll2);
			process::Accumulate(sig, &stack[0], wl);
			process::Accumulate(sig, rolled.row(i), wl);			
		}
		// process::Copy(&stack[0], wl, rolled.row(i));
	}

	clock.log("roll for ot");

	if(debug == 2) {
		npy::Save(logdir + "roll.npy", rolled);		
		npy::Save(logdir + "stack.npy", stack);		
	}

	auto imax = process::argmax(grid);
	// utils::PrintVec(lmax);

	outbuf[0] = grid[imax] * 10000;
	outbuf[1] = imax;
	outbuf[2] = process::argmax(stack) * 2; // ot ix for original sr


	clock.print();

	// npy::Save(resdir + "d2.npy", dat);
	// std::cout << "epoch: " << epoch << '\n';
	// return 0;

}


void CorrSearchDec2XBoth(float* rawdat_p, uint32_t nchan, uint32_t npts, float sr, float* stalocs_p, uint32_t nsta, uint16_t* chanmap_p, uint16_t* ttable1_ptr, uint16_t* ttable2_ptr, uint32_t ngrid, uint32_t* outbuf, float* grid_p, uint32_t nthreads, std::string& logdir, int debug) 
{

	auto rawdat = Array2D<float>(rawdat_p, nchan, npts, false);
	auto stalocs = Array2D<float>(stalocs_p, nsta, 3, false);
	auto chanmap = Vector<uint16_t>(chanmap_p, nchan, false);
	auto ttable1 = Array2D<uint16_t>(ttable1_ptr, nsta, ngrid, false);
	auto ttable2 = Array2D<uint16_t>(ttable2_ptr, nsta, ngrid, false);
	auto grid = Vector<float>(grid_p, ngrid, false);
	CorrSearchDec2XBoth(rawdat, sr, stalocs, chanmap, ttable1, ttable2, outbuf, grid, nthreads, logdir, debug);
	// CorrSearchDec2XBoth(rawdat, sr, stalocs, chanmap, ttable1, ttable2, outbuf, grid, nthreads, logdir);
	// std::cout << "logdir: " << logdir << '\n';

}




void CorrSearchDec2X(Array2D<float>& rawdat, float sr, Array2D<float>& stalocs, Vector<uint16_t>& chanmap, Array2D<uint16_t>& ttable, uint32_t* outbuf, Vector<float>& grid, uint32_t nthreads) 
{	
	omp_set_num_threads(nthreads);

	// std::string resdir(OUTDIR);
	auto clock = Clock();

	uint32_t nsta = ttable.nrow_;
	uint32_t ngrid = ttable.ncol_;
	uint32_t nchan = rawdat.nrow_;
	
	// Create data buffers ////////////////////////////////////////////////////		
	size_t wlr = rawdat.ncol_;  // wlen of raw data
	size_t flr = wlr / 2 + 1; // nfreq of raw data
	// wlen of 2x decimated and zero padded aligned data wlr=(wlr / dec) * 2
	size_t wl = utils::PadToBytes<float>(wlr);
	size_t fl = wl / 2 + 1; // nfreq

	auto dat = Array2D<float>(nchan, wl); // decimated and zero padded data
	auto fdat = Array2D<fftwf_complex>(nchan, utils::PadToBytes<fftwf_complex>(fl));
	auto tmp = Vector<float>(wl);
	auto ftmp = Vector<fftwf_complex>(fl);

	// int patience = FFTW_MEASURE;
	int patience = FFTW_ESTIMATE;
	fftwf_plan plan_fwd = fftwf_plan_dft_r2c_1d(wl, &tmp[0], &ftmp[0], patience);
	fftwf_plan plan_inv = fftwf_plan_dft_c2r_1d(wl, &ftmp[0], &tmp[0], patience);
	// fftw_export_wisdom_to_filename(const char *filename)
	clock.log("plan fft");

	// utils::PrintArraySize(rawdat, "raw data");
	// utils::PrintArraySize(dat, "data");
	// utils::PrintArraySize(ttable, "ttable");

	// Pre-process ////////////////////////////////////////////////////////////
	std::vector<float> cfreqs {40, 45, 300, 350};
	// std::vector<float> cfreqs {20, 25, 240, 250};
	auto filter = process::BuildFreqFilter(cfreqs, fl, sr);	
	float energy = filter.energy() * 2;

	// copy, fft, whiten, ifft, taper, copy each 2nd value to dat, fft	
	// #pragma omp parallel for
	for(size_t i = 0; i < rawdat.nrow_; ++i) {
		process::Fill(tmp, 0);
		std::copy(rawdat.row(i), rawdat.row(i) + wlr, &tmp[0]); // gaurantee alignment for fftw
		fftwf_execute_dft_r2c(plan_fwd, &tmp[0], &ftmp[0]);
		process::ApplyFreqFilterReplace(&ftmp[0], fl, filter); // whiten
		fftwf_execute_dft_c2r(plan_inv, &ftmp[0], &tmp[0]);
		process::taper(&tmp[0], wl, 50);

		float* drow = dat.row(i);
		process::Fill(drow, wl, 0);
		for(size_t j = 0; j < wl / 2; ++j) 	drow[j] = tmp[j * 2] / wl; //decimate
		fftwf_execute_dft_r2c(plan_fwd, drow, fdat.row(i));
	}
	clock.log("fft");
	// npy::Save(resdir + "d1.npy", dat);

	// Correlate //////////////////////////////////////////////////////////////
	auto keys = utils::arange<uint16_t>(0, nsta);
	auto groups = keygen::GroupChannels(keys, chanmap);
	
	float min_dist = 300;
	float max_dist = 1500;
	auto spairs = keygen::DistFilt(keys, stalocs, min_dist, max_dist);
	size_t npair = spairs.nrow_;
	// npy::Save(resdir + "ck.npy", spairs);

	auto vshift = process::BuildPhaseShiftVec(fl, wl / 2);	
	auto ccdat = Array2D<float>(npair, wl);
	assert((uintptr_t) ccdat.row(1) % MEM_ALIGNMENT == 0);
	// utils::PrintArraySize(ccdat, "ccdat");
	
	clock.start();

	#pragma omp parallel
	{
		auto tmp = malloc_cache_align<float>(wl);
		auto ftmp = malloc_cache_align<fftwf_complex>(fl);

		#pragma omp for
		for(size_t i = 0; i < npair; ++i) {
			uint16_t *pair = spairs.row(i);
			float *csig = ccdat.row(i);
			std::fill(csig, csig + wl, 0);

			uint32_t nstack = 0;		
			for(auto&& k0 : groups[pair[0]]) {
				for(auto&& k1 : groups[pair[1]]) {
					process::XCorr(fdat.row(k0), fdat.row(k1), ftmp, fl);
					process::Convolve(&vshift[0], ftmp, fl);
					fftwf_execute_dft_c2r(plan_inv, ftmp, tmp);
					for(size_t j=0; j < wl; ++j) csig[j] += std::abs(tmp[j]);
					nstack++;
				}
			}
			process::Multiply(csig, wl, 1.0 / (nstack * energy));
			process::EMA_NoAbs(csig, wl, 10, true);
		}
		free(tmp);
		free(ftmp);
	}
	clock.log("xcorr");


	auto output = Vector<float>(ngrid);
	beamform::InterLocBlocks(ccdat, spairs, ttable, output);	
	clock.log("search");

	// auto mad = beamform::out2MAD(output);
	// npy::Save(resdir + utils::ZeroPadInt(0) + ".gdat", output);
	// auto stats = beamform::VAMax(output);
	// utils::PrintMaxAndLoc(stats[0], gridlocs.row(stats[1]));
	// auto stats = beamform::MaxAndLoc(output, gridlocs);

	// std::cout << "amax: " << process::argmax(output) << '\n';
	// std::cout << "max: " << process::max(output) << '\n';
	// utils::PrintMaxAndLoc(stats);

	// std::string fout = resdir + "latest.npy";
	// npy::Save(fout, output);

	auto wtt = ttable.copy_col(process::argmax(output));
	// npy::Save(resdir + "tts_win.npy", wtt);

	auto stack = Vector<float>(wl);
	stack.fill(0);

	for(size_t i = 0; i < groups.size(); ++i) {
		int rollby = static_cast<int>(wtt[i]);
		// std::cout << "rollby: " << rollby << '\n';			
		for(auto&& k : groups[i]) {
			float *sig = dat.row(k);
			process::ExpMovingAverage(sig, wl, 10, true);
			process::Roll(sig, wl, rollby);
			process::Accumulate(sig, &stack[0], wl);
		}
	}

	clock.log("roll for ot");

	auto imax = process::argmax(output);
	// utils::PrintVec(lmax);

	outbuf[0] = output[imax] * 10000;
	outbuf[1] = imax;
	outbuf[2] = process::argmax(stack) * 2; // ot ix for original sr

	process::Copy(&output[0], ngrid, &grid[0]);

	clock.print();

	// npy::Save(resdir + "d2.npy", dat);
	// std::cout << "epoch: " << epoch << '\n';
	// return 0;

}


void CorrSearchDec2X(float* rawdat_p, uint32_t nchan, uint32_t npts, float sr, float* stalocs_p, uint32_t nsta, uint16_t* chanmap_p, uint16_t* ttable_p, uint32_t ngrid, uint32_t* outbuf, float* grid_p, uint32_t nthreads) 
{

	auto rawdat = Array2D<float>(rawdat_p, nchan, npts, false);
	auto stalocs = Array2D<float>(stalocs_p, nsta, 3, false);
	auto chanmap = Vector<uint16_t>(chanmap_p, nchan, false);
	auto ttable = Array2D<uint16_t>(ttable_p, nsta, ngrid, false);
	auto grid = Vector<float>(grid_p, ngrid, false);
	CorrSearchDec2X(rawdat, sr, stalocs, chanmap, ttable, outbuf, grid, nthreads);

}



}


#endif
