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
#include "xseis/npy.h"
#include "xseis/keygen.h"


namespace interloc {

void CorrSearchDec2X(Array2D<float>& rdat, float sr, Array2D<float>& stalocs, Vector<uint16_t>& chanmap, Array2D<uint16_t>& ttable, uint32_t* outbuf, Vector<float>& grid, uint32_t nthreads=4) 
{	

	omp_set_num_threads(nthreads);

	// std::cout << "rdat owns: " << rdat.owns_ << '\n';
	// std::string resdir(OUTDIR);
	auto clock = Clock();

	uint32_t nsta = ttable.nrow_;
	uint32_t ngrid = ttable.ncol_;
	uint32_t nchan = rdat.nrow_;

	auto keys = utils::arange<uint16_t>(0, nsta);
	auto groups = keygen::GroupChannels(keys, chanmap);
	
	// Create data buffers ////////////////////////////////////////////////////	
	
	// size_t nchan = rdat.nrow_;
	size_t wlr = rdat.ncol_;  // wlen of raw data
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

	utils::PrintArraySize(rdat, "raw data");
	utils::PrintArraySize(dat, "data");
	utils::PrintArraySize(ttable, "ttable");

	clock.log("plan fft");

	// Pre-process ////////////////////////////////////////////////////////////
	std::vector<float> cfreqs {40, 45, 300, 350};
	// std::vector<float> cfreqs {20, 25, 240, 250};
	auto filter = process::BuildFreqFilter(cfreqs, fl, sr);	
	float energy = filter.energy() * 2;

	// copy, fft, whiten, ifft, taper, copy each 2nd value to dat, fft	
	// #pragma omp parallel for
	for(size_t i = 0; i < rdat.nrow_; ++i) {
		process::Fill(tmp, 0);
		std::copy(rdat.row(i), rdat.row(i) + wlr, &tmp[0]);
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
// outbuf[3] = 0;
// outbuf[4] = ((ot * sr) / 1000000. + 0.5) + epoch;
// outbuf[4] = 0;

// // npy::Save(resdir + std::to_string(epoch) + ".gdat", output);
// std::vector<float> stats;
// for(size_t i = 0; i < 5; ++i) stats.push_back(outbuf[i]);
// // utils::AppendToFile(resdir + "output.log", stats);

// printf("maxval x y z origin_time\n");
// utils::PrintVec(stats);


// // void search(float *rdat_p, uint32_t nchan, uint32_t npts, float* stalocs_p, uint32_t nsta, uint16_t* chanmap_p, int* tmeta, uint16_t* ttable_p, uint32_t ngrid, float *outbuf) 

void Search(Array2D<float>& rdat, Array2D<float>& stalocs, Vector<uint16_t>& chanmap, int* tmeta, Array2D<uint16_t>& ttable, float *outbuf) 
{	

	std::cout << "rdat owns: " << rdat.owns_ << '\n';
	// std::string resdir(OUTDIR);
	auto clock = Clock();

	utils::print(rdat);


	uint32_t nsta = stalocs.nrow_;
	uint32_t nchan = rdat.nrow_;
	uint32_t fixlen = rdat.ncol_;
	uint32_t ngrid = ttable.ncol_;

	// for now sampling rate hardcoded
	float const sr_raw = 6000;
	float const sr = 3000;
	// uint32_t const fixlen = 6000;

	int* shape = &tmeta[0];
	int* origin = &tmeta[3];
	int spacing = tmeta[6];

	// size_t ngrid = shape[0] * shape[1] * shape[2];
	// printf("shape: (%d,%d,%d) og: (%d,%d,%d) spacing: %d m\n", shape[0], shape[1],
	// 			 shape[2], origin[0], origin[1], origin[2], spacing);
	
	auto keys = utils::arange<uint16_t>(0, nsta);
	auto groups = keygen::GroupChannels(keys, chanmap);

	
	// Create data buffers ////////////////////////////////////////////////////	
	
	// size_t nchan = rdat.nrow_;
	size_t wlr = rdat.ncol_;  // wlen of raw data
	size_t flr = wlr / 2 + 1;
	// wlen of decimated and zero padded aligned data wlr=(wlr / dec) * 2
	size_t wl = utils::PadToBytes<float>(wlr);
	size_t fl = wl / 2 + 1; // complex len

	auto dat = Array2D<float>(nchan, wl); // decimated and zero padded
	auto fdat = Array2D<fftwf_complex>(nchan, utils::PadToBytes<fftwf_complex>(fl));
	auto tmp = Vector<float>(wl);
	auto ftmp = Vector<fftwf_complex>(fl);

	// int patience = FFTW_MEASURE;
	int patience = FFTW_ESTIMATE;
	fftwf_plan plan_fwd = fftwf_plan_dft_r2c_1d(wl, &tmp[0], &ftmp[0], patience);
	fftwf_plan plan_inv = fftwf_plan_dft_c2r_1d(wl, &ftmp[0], &tmp[0], patience);
	// fftw_export_wisdom_to_filename(const char *filename)

	utils::PrintArraySize(rdat, "raw data");
	utils::PrintArraySize(dat, "data");

	// clock.log("plan fft");

	// Pre-process ////////////////////////////////////////////////////////////
	std::vector<float> cfreqs {40, 45, 300, 350};
	// std::vector<float> cfreqs {20, 25, 240, 250};
	auto filter = process::BuildFreqFilter(cfreqs, fl, sr_raw);	
	float energy = filter.energy() * 2;

	// copy, fft, whiten, ifft, taper, copy each 2nd value to dat, fft	
	// #pragma omp parallel for
	for(size_t i = 0; i < rdat.nrow_; ++i) {
		process::Fill(tmp, 0);
		std::copy(rdat.row(i), rdat.row(i) + wlr, &tmp[0]);
		fftwf_execute_dft_r2c(plan_fwd, &tmp[0], &ftmp[0]);
		process::ApplyFreqFilterReplace(&ftmp[0], fl, filter); // whiten
		fftwf_execute_dft_c2r(plan_inv, &ftmp[0], &tmp[0]);
		process::taper(&tmp[0], wl, 50);

		float* drow = dat.row(i);
		process::Fill(drow, wl, 0);
		for(size_t j = 0; j < wl / 2; ++j) 	drow[j] = tmp[j * 2] / wl; //decimate
		fftwf_execute_dft_r2c(plan_fwd, drow, fdat.row(i));
	}
	// clock.log("fft");
	// npy::Save(resdir + "d1.npy", dat);
	printf("preproc\n");

	// Correlate //////////////////////////////////////////////////////////////
	float min_dist = 300;
	float max_dist = 1500;
	auto spairs = keygen::DistFilt(keys, stalocs, min_dist, max_dist);
	size_t npair = spairs.nrow_;
	// npy::Save(resdir + "ck.npy", spairs);

	auto vshift = process::BuildPhaseShiftVec(fl, wl / 2);	
	auto ccdat = Array2D<float>(npair, wl);
	assert((uintptr_t) ccdat.row(1) % MEM_ALIGNMENT == 0);
	// utils::PrintArraySize(ccdat, "ccdat");
	
	// clock.start();

	#pragma omp parallel
	{
		auto tmp = malloc_cache_align<float>(wl);
		auto ftmp = malloc_cache_align<fftwf_complex>(fl);

		#pragma omp for
		for(size_t i = 0; i < npair; ++i) {
			uint16_t *pair = spairs.row(i);
			float *csig = ccdat.row(i);
			std::fill(csig, csig + wl, 0);

			uint nstack = 0;		
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
	// clock.log("xcorr");

	printf("xcorr\n");

	auto output = Vector<float>(ngrid);
	beamform::InterLocBlocks(ccdat, spairs, ttable, output);	
	clock.log("search");

	printf("search\n");

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

	auto imax = process::argmax(output);
	auto lmax = process::get_point(imax, tmeta);
	// utils::PrintVec(lmax);
	size_t ot = process::argmax(stack);

	outbuf[0] = output[imax];
	outbuf[1] = lmax[0];
	outbuf[2] = lmax[1];
	outbuf[3] = lmax[2];
	// outbuf[4] = ((ot * sr) / 1000000. + 0.5) + epoch;
	outbuf[4] = ot;
	
	// npy::Save(resdir + std::to_string(epoch) + ".gdat", output);
	std::vector<float> stats;
	for(size_t i = 0; i < 5; ++i) stats.push_back(outbuf[i]);
	// utils::AppendToFile(resdir + "output.log", stats);
	
	printf("maxval x y z origin_time\n");
	utils::PrintVec(stats);

	// npy::Save(resdir + "d2.npy", dat);
	// std::cout << "epoch: " << epoch << '\n';
	// return 0;

}


}


#endif
