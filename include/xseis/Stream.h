#ifndef STREAM_H
#define STREAM_H

#include <iostream>
#include <fftw3.h>
#include "xseis/process.h"
#include "xseis/beamform.h"

class Stream {
	public:
		float *data = nullptr;
		int nsig, npts;
		float sr;
		int nthreads;

        float *data_cc = nullptr;
		int plan_flag;

		float (*fdata)[2] = nullptr;
		int nfreq;
		float fsr;

        // fftwf_complex *fdata = nullptr;
        // fftwf_complex *fdata = nullptr;
        fftwf_complex *fdata_cc = nullptr;
        fftwf_plan plan1, plan2, plan3;

		int *ckeys = nullptr;
		int ncc;


		Stream(float *data, int nsig, int npts, float sr, int nthreads, int patience)
			: data(data), nsig(nsig), npts(npts), sr(sr), nthreads(nthreads) {
			fsr = npts / sr;

			nfreq = npts / 2 + 1;

			if (patience==0){plan_flag = FFTW_ESTIMATE;}
			else if (patience==1){plan_flag = FFTW_MEASURE;}
			else if (patience==2){plan_flag = FFTW_PATIENT;}
			else{printf("Patience must be 0, 1 or 2 \n");}

			fftwf_init_threads();
			fftwf_plan_with_nthreads(nthreads);
			fftwf_set_timelimit(30);
		}

		// ~Stream()
		// {
		// 	fftwf_destroy_plan(plan1);
		// 	fftwf_destroy_plan(plan2);
		// 	fftwf_destroy_plan(plan3);
		// 	fftwf_cleanup_threads();
		// 	fftwf_cleanup();
		// }

		// BEAMFORMING WRAPPER

		// void tt_homo(float *src_loc, float *sta_locs, int nsta, float velocity, float *traveltimes){
		// 	beamform::tt_homo(src_loc, sta_locs, nsta, velocity, traveltimes);
		// }

		//  SIGNAL PROCESSING

		void multiply(float factor){
			// auto fbind = std::bind(process::multiply, _1, _2, factor);
			auto fbind = [factor] (float *sig, int npts2){process::multiply(sig, npts2, factor);};	
			// process::map_signals(data, nsig, npts, &fbind);
			process::map_signals_parallel(data, nsig, npts, &fbind, nthreads);
		}

		void taper(float factor){
			auto fbind = std::bind(process::taper, _1, _2, factor);
			// auto fbind = [] (float *sig, int npts){process::taper(sig, npts, factor);};	
			process::map_signals_parallel(data, nsig, npts, &fbind, nthreads);
		}

		void norm_one_bit(){
			auto fbind = std::bind(process::norm_one_bit, _1, _2);
			// auto fbind = [] (float *sig, int npts){process::norm_one_bit(sig, npts);};	
			process::map_signals_parallel(data, nsig, npts, &fbind, nthreads);
		}

		void absolute(){
			auto fbind = std::bind(process::absolute, _1, _2);
			process::map_signals_parallel(data, nsig, npts, &fbind, nthreads);
		}

		void norm_energy(){
			auto fbind = std::bind(process::norm_energy, _1, _2);
			process::map_signals_parallel(fdata, nsig, npts, &fbind, nthreads);
		}

		void fft(){fftwf_execute(plan1);}
		void ifft(){fftwf_execute(plan2);}
		void ifft_cc(){fftwf_execute(plan3);}

		void whiten(float fmin, float fmax, int len_taper) {
			auto fbind = std::bind(process::whiten, _1, _2, fsr, fmin, fmax, len_taper);
			// auto fbind = [] (float *sig, int npts){
			// 	process::whiten(sig, npts, fsr, fmin, fmax, len_taper);
			// };	

			process::map_signals_parallel(fdata, nsig, nfreq, &fbind, nthreads);
		}

		void roll_cc(){
			auto fbind = std::bind(process::roll, _1, _2, npts / 2);
			process::map_signals_parallel(data_cc, ncc, npts, &fbind, nthreads);
		}

		void plan_fft()	{
			// Out of place real forward and inverse FFT plan
			// Note that output is scaled by npts as in FFTW3 docs
			printf("Planning fft/ifft.. ");
			clock_t tStart = clock();

			fdata = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * nfreq * nsig);
			// fdata = (fftwf_complex*) data;

			plan1 = fftwf_plan_many_dft_r2c(1, &npts, nsig, data, NULL, 1, npts, fdata, NULL, 1, nfreq, plan_flag);
			plan2 = fftwf_plan_many_dft_c2r(1, &npts, nsig, fdata, NULL, 1, nfreq, data, NULL, 1, npts, plan_flag);

			printf("Completed in %.3f seconds\n", (float)(clock() - tStart)/CLOCKS_PER_SEC);
		}

		// void plan_fft()	{
		// 	// Out of place real forward and inverse FFT plan
		// 	// Note that output is scaled by npts as in FFTW3 docs
		// 	printf("Planning fft/ifft.. ");
		// 	clock_t tStart = clock();

		// 	fdata = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * nfreq * nsig);
		// 	// fdata = (fftwf_complex*) data;

		// 	plan1 = fftwf_plan_many_dft_r2c(1, &npts, nsig, data, NULL, 1, npts, fdata, NULL, 1, nfreq, plan_flag);
		// 	plan2 = fftwf_plan_many_dft_c2r(1, &npts, nsig, fdata, NULL, 1, nfreq, data, NULL, 1, npts, plan_flag);

		// 	printf("Completed in %.3f seconds\n", (float)(clock() - tStart)/CLOCKS_PER_SEC);
		// }


		void plan_cc(float *arg_data_cc, int arg_ncc)
		{
			printf("Planning CC fft/ifft.. ");
			clock_t tStart = clock();

			ncc = arg_ncc;
			data_cc = arg_data_cc;

			fdata_cc = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * nfreq * ncc);
			// fdata_cc = float (*fdata)[2]

			plan3 = fftwf_plan_many_dft_c2r(1, &npts, ncc, fdata_cc, NULL, 1, nfreq, data_cc, NULL, 1, npts, plan_flag);

			printf("Completed in %.3f seconds\n", (float)(clock() - tStart)/CLOCKS_PER_SEC);
		}

		void run_cc(int *ckeys)
		{
			printf("nsig= %d nfreq = %d ncc = %d \n", nsig, nfreq, ncc);
			process::correlate_all_parallel(fdata, nsig, nfreq, fdata_cc, ckeys, ncc,  nthreads);
			// process::correlate_all(fdata, nsig, nfreq, fdata_cc, ckeys, ncc);
		}

		// void init_beamform()
		// {
		// 	tt_homo(float *src_loc, float *sta_locs, int nsta,
		// 		 float velocity, float *traveltimes)
		// }


};
#endif
