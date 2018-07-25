#ifndef ILOC_H
#define ILOC_H
// #include <fftw3.h>
#include <cstring>
#include <omp.h>
#include <fftw3.h>
#include <cstddef>

#include "xseis/utils.h"
#include "xseis/structures.h"
#include "xseis/process.h"
#include "xseis/beamform.h"
// #include "xseis/h5wrap.h"
#include "xseis/npy.h"
#include "xseis/keygen.h"
#include "xseis/mseed.h"
// #include "xseis/raytrace.h"
// #include "xseis/FftHandler.h"


void searchMSEED(char* fbuf, size_t nbytes,
				char* stas, uint32_t nsta,
				float* sta_locs, int* tmeta,
				float *tt_ptr, float *outbuf) 
{	
	uint32_t nthreads = 4;
	omp_set_num_threads(nthreads);
	float sr_raw = 6000;
	float sr = 3000;
	uint32_t fixlen = 6000;

	auto clock = Clock();

	int* shape = &tmeta[0];
	int* origin = &tmeta[3];
	int spacing = tmeta[6];

	std::cout << "spacing: " << spacing << '\n';

	size_t ngrid = shape[0] * shape[1] * shape[2];

	std::cout << "nx: " << shape[0] << '\n';
	std::cout << "ny: " << shape[1] << '\n';
	std::cout << "nz: " << shape[2] << '\n';

	auto tts_all = Array2D<float>(tt_ptr, nsta, ngrid, false);
	auto slocs_all = Array2D<float>(sta_locs, nsta, 3, false);
	// auto sraw = Vector<char[4]>(stas, nsta, false);

	// std::vector<std::string> names;
	// for(size_t i = 0; i < nsta; i++) names.push_back(std::string(&stas[i * 4]));

	std::map<std::string, uint32_t> sta_all;
	for(uint32_t i = 0; i < nsta; ++i) sta_all[std::string(&stas[i * 4])] = i;

	// get mseed data as contig buffer and fill chanmap
	std::map<std::string, size_t> chanmap;
	size_t epoch;
	Array2D<float> rdat = mseed::ToDataFixed(fbuf, nbytes, chanmap, epoch, 6000);
	// clock.log("mseed to contig");
	// npy::Save(resdir + "data.npy", rdat);
	// return 0;

	std::map<std::string, std::vector<uint16_t>> stagroups;
	for(auto& x : chanmap) stagroups[x.first.substr(0, x.first.size() - 2)].push_back(x.second);

	// indexes for matchin stations
	std::vector<uint16_t> validkeys;
	for(auto&& x : stagroups) validkeys.push_back(sta_all[x.first]);
	// clock.log("sgroups,save");
	// utils::PrintVec(validkeys);

	auto stalocs = Array2D<float>(validkeys.size(), 3);

	for(size_t i = 0; i < validkeys.size(); ++i) {
		auto ptr = slocs_all.row(validkeys[i]);
		std::copy(ptr, ptr + 3, stalocs.row(i));		
	}
	auto keys = utils::arange<uint16_t>(0, stalocs.nrow_);
	VVui16 groups;
	for(auto&& x : stagroups) groups.push_back(x.second);
	// for(auto& x : groups) utils::PrintVec(x);
	// clock.log("load glocs, tts");

	auto ttable = Array2D<uint16_t>(validkeys.size(), ngrid);
	for(size_t i = 0; i < validkeys.size(); ++i) {
		auto ptr_in = tts_all.row(validkeys[i]);
		auto ptr_out = ttable.row(i);
		for(size_t j = 0; j < ttable.ncol_; ++j) {
			ptr_out[j] = static_cast<uint16_t>(ptr_in[j] * sr + 0.5);
		}
	}

	// clock.log("convert ttable");		
	// Create data buffers ////////////////////////////////////////////////////	
	
	size_t nchan = rdat.nrow_;
	size_t wlr = rdat.ncol_;  // wlen of raw data
	size_t flr = wlr / 2 + 1;
	// wlen of decimated and zero padded aligned data wlr=(wlr / dec) * 2
	size_t wl = utils::PadToBytes<float>(wlr);
	size_t fl = wl / 2 + 1; // complex len

	// auto rdat = Array2D<float>(nchan, wlr); // raw data
	auto dat = Array2D<float>(nchan, wl); // decimated and zero padded
	auto fdat = Array2D<fftwf_complex>(nchan, utils::PadToBytes<fftwf_complex>(fl));
	auto tmp = Vector<float>(wl);
	auto ftmp = Vector<fftwf_complex>(fl);

	// clock.log(" ");

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
		process::ApplyFreqFilterReplace(&ftmp[0], fl, filter);
		fftwf_execute_dft_c2r(plan_inv, &ftmp[0], &tmp[0]);
		process::taper(&tmp[0], wl, 50);

		float* drow = dat.row(i);
		process::Fill(drow, wl, 0);
		for(size_t j = 0; j < wl / 2; ++j) 	drow[j] = tmp[j * 2] / wl;
		fftwf_execute_dft_r2c(plan_fwd, drow, fdat.row(i));
	}
	// clock.log("fft");
	// npy::Save(resdir + "d1.npy", dat);

	// Correlate //////////////////////////////////////////////////////////////

	float min_dist = 300;
	float max_dist = 1500;
	auto spairs = keygen::DistFilt(keys, stalocs, min_dist, max_dist);
	size_t npair = spairs.nrow_;
	// clock.log("sta pairs");
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
	clock.log("xcorr");


	auto output = Vector<float>(ngrid);
	beamform::InterLocBlocks(ccdat, spairs, ttable, output);	
	clock.log("search");

	// auto mad = beamform::out2MAD(output);
	// npy::Save(resdir + utils::ZeroPadInt(0) + ".gdat", output);

	// auto stats = beamform::VAMax(mad);
	// utils::AppendToFile(resdir + utils::ZeroPadInt(pix) + ".sdat", stats);
	// utils::PrintMaxAndLoc(stats[0], gridlocs.row(stats[1]));
	// auto stats = beamform::MaxAndLoc(output, gridlocs);

	std::cout << "amax: " << process::argmax(output) << '\n';
	std::cout << "max: " << process::max(output) << '\n';

	outbuf[0] = process::max(output);

	// utils::PrintMaxAndLoc(stats);

	// std::string fout = resdir + "latest.npy";
	// npy::Save(fout, output);

	// auto wtt = ttable.copy_col(process::argmax(output));
	// npy::Save(resdir + "tts_win.npy", wtt);

	// for(size_t i = 0; i < groups.size(); ++i) {
	// 	int rollby = static_cast<int>(wtt[i]);
	// 	// std::cout << "rollby: " << rollby << '\n';			
	// 	for(auto&& k : groups[i]) {
	// 		float *sig = dat.row(k);
	// 		process::ExpMovingAverage(sig, wl, 10, true);
	// 		process::Roll(sig, wl, rollby);
	// 	}
	// }


	// npy::Save(resdir + "d2.npy", dat);
	// std::cout << "epoch: " << epoch << '\n';
	// return 0;

}
#endif

