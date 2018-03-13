#ifndef FftHandler_H
#define FftHandler_H

#include <iostream>
#include <fftw3.h>
#include "xseis/structures.h"

class FftHandler {
	public:

		int patience_;
		const int nthread_;
		fftwf_plan plan_fwd_ = NULL;
		fftwf_plan plan_fwd2_ = NULL;
		fftwf_plan plan_inv_= NULL;	
		fftwf_plan plan_inv_cc_= NULL;	

		// describes real input array
		float* inf32_;
		// int nsig_, npts_;
		uint32_t nsig_, npts_;

		// describes plan_fwd_ output
		fftwf_complex* outf32_;
		uint32_t nfreq_;

		// describes plan_inv_ output
		fftwf_complex* ini32_;
		float* outi32_;
		uint32_t wlen_;

		// fftwf_complex* outf32_cc;
		int nf[1], ni[1], idistf, idisti, howmanyf, howmanyi, odistf, odisti;
		const int *inembed = NULL;
		const int *onembed = NULL;
		const int rank = 1;    // Computing multiple 1D transforms
		const int istride = 1; // Distance between two elements in same input column
		const int ostride = 1; // Distance between two elements in same outpt column

		
		FftHandler(int patience, int nthread): patience_(patience), nthread_(nthread)
		{
			fftwf_init_threads();
			fftwf_plan_with_nthreads(nthread_);			
			fftwf_set_timelimit(30); //time limit for patient flag
		}

		~FftHandler()
		{
			// delete outf32_;
			// delete ini32_;
			// delete outi32_;
			fftwf_destroy_plan(plan_fwd_);
			fftwf_destroy_plan(plan_inv_);
			fftwf_cleanup_threads();
			fftwf_cleanup();
		}

		Array2D<fftwf_complex> plan_fwd(Array2D<float>& data, uint32_t wlen)
		{
			nsig_ = data.nrow_;
			npts_ = data.ncol_;
			wlen_ = wlen;
			nfreq_ = wlen_ / 2 + 1;

			inf32_ = data.data_;

			outf32_ = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * nsig_ * nfreq_);
			auto arr_outf32 = Array2D<fftwf_complex>(outf32_, nsig_, nfreq_);
			// auto arr_outf32 = Array2D<fftwf_complex>({nsig_, nfreq_});
			// outf32_ = arr_outf32.data_;

			// ini32_ = outf32_;
			// outi32_ = new float[nsig_ * wlen_];

			nf[0]    = wlen_;        // 1D real transform length
			howmanyf = nsig_;   // Number of transforms
			idistf = npts_;   // Distance between start of k'th input
			odistf = nfreq_;     // Distance between start of k'th output 
			
			plan_fwd_ = fftwf_plan_many_dft_r2c(rank, nf, howmanyf,
													inf32_, inembed,
													istride, idistf,
													outf32_, onembed,
													ostride, odistf,
													patience_);
			return arr_outf32;

		}


		void plan_fwd2(Array2D<float>& data, Array2D<fftwf_complex>& arr_outf32)
		{
			nsig_ = data.nrow_;
			npts_ = data.ncol_;
			wlen_ = data.ncol_;
			nfreq_ = wlen_ / 2 + 1;

			inf32_ = data.data_;

			// outf32_ = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * nsig_ * nfreq_);
			// auto arr_outf32 = Array2D<fftwf_complex>(outf32_, nsig_, nfreq_);
			// auto arr_outf32 = Array2D<fftwf_complex>({nsig_, nfreq_});
			outf32_ = arr_outf32.data_;

			// ini32_ = outf32_;
			// outi32_ = new float[nsig_ * wlen_];

			nf[0]    = wlen_;        // 1D real transform length
			howmanyf = nsig_;   // Number of transforms
			idistf = npts_;   // Distance between start of k'th input
			odistf = nfreq_;     // Distance between start of k'th output 
			
			plan_fwd2_ = fftwf_plan_many_dft_r2c(rank, nf, howmanyf,
													inf32_, inembed,
													istride, idistf,
													outf32_, onembed,
													ostride, odistf,
													patience_);

		}

		Array2D<float> plan_inv(Array2D<fftwf_complex>& data)
		{
			nsig_ = data.nrow_;
			nfreq_ = data.ncol_;
			
			npts_ = (nfreq_ - 1) * 2;

			ini32_ = data.data_;
			auto arr_outi32 = Array2D<float>({nsig_, npts_});
			outi32_ = arr_outi32.data_;

			ni[0]    = npts_;        // Length of time-domain data to inverse (cclen)
			howmanyi = nsig_; // Number of inverse transforms
			idisti = nfreq_;     // Distance between start of k'th input
			odisti = npts_;   // Distance between start of k'th output 

			
			plan_inv_ = fftwf_plan_many_dft_c2r(rank, ni, howmanyi,
													ini32_, inembed,
													istride, idisti,
													outi32_, onembed,
													ostride, odisti,
													patience_);

			return arr_outi32;
		}

		void plan_inv(Array2D<fftwf_complex>& fdata, Array2D<float>& arr_outi32)
		{
			nsig_ = fdata.nrow_;
			nfreq_ = fdata.ncol_;
			
			npts_ = (nfreq_ - 1) * 2;

			ini32_ = fdata.data_;
			// auto arr_outi32 = Array2D<float>({nsig_, npts_});
			outi32_ = arr_outi32.data_;

			ni[0]    = npts_;        // Length of time-domain data to inverse (cclen)
			howmanyi = nsig_; // Number of inverse transforms
			idisti = nfreq_;     // Distance between start of k'th input
			odisti = npts_;   // Distance between start of k'th output 

			
			plan_inv_ = fftwf_plan_many_dft_c2r(rank, ni, howmanyi,
													ini32_, inembed,
													istride, idisti,
													outi32_, onembed,
													ostride, odisti,
													patience_);

		}


		Array2D<float> plan_inv_cc(Array2D<fftwf_complex>& data)
		{
			nsig_ = data.nrow_;
			nfreq_ = data.ncol_;
			
			npts_ = (nfreq_ - 1) * 2;

			ini32_ = data.data_;
			auto arr_outi32 = Array2D<float>({nsig_, npts_});
			outi32_ = arr_outi32.data_;

			ni[0]    = npts_;        // Length of time-domain data to inverse (cclen)
			howmanyi = nsig_; // Number of inverse transforms
			idisti = nfreq_;     // Distance between start of k'th input
			odisti = npts_;   // Distance between start of k'th output 
						
			plan_inv_cc_ = fftwf_plan_many_dft_c2r(rank, ni, howmanyi,
													ini32_, inembed,
													istride, idisti,
													outi32_, onembed,
													ostride, odisti,
													patience_);

			return arr_outi32;
		}


		void exec_fwd()
		{			
			fftwf_execute(plan_fwd_);
		}

		void exec_fwd2()
		{			
			fftwf_execute(plan_fwd2_);
		}

		void exec_fwd(uint32_t offset)
		{			
			fftwf_execute_dft_r2c(plan_fwd_, inf32_ + offset, outf32_);
		}

		void exec_inv()
		{			
			fftwf_execute(plan_inv_);
		}

		void exec_inv_cc()
		{			
			fftwf_execute(plan_inv_cc_);

		}


		// // void plan_strided(structures::Array2D<float>& data, int wlen_arg)
		// void plan_strided(float* data, uint32_t shape[2], int wlen_arg)
		// {
		// 	nsig_ = shape[0];
		// 	npts_ = shape[1];
		// 	wlen_ = wlen_arg;
		// 	nfreq_ = wlen_ / 2 + 1;

		// 	inf32_ = data;
		// 	// inf32_shape = shape;

		// 	// allocate memory, will move this out later
		// 	outf32_ = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * nsig_ * nfreq_);
		// 	ini32_ = outf32_;
		// 	outi32_ = new float[nsig_ * wlen_];

		// 	nf[0]    = wlen_;        // 1D real transform length
		// 	howmanyf = nsig_;   // Number of transforms
		// 	idistf = npts_;   // Distance between start of k'th input
		// 	odistf = nfreq_;     // Distance between start of k'th output 
		// 	ni[0]    = wlen_;        // Length of time-domain data to inverse (cclen)
		// 	howmanyi = nsig_; // Number of inverse transforms
		// 	idisti = nfreq_;     // Distance between start of k'th input
		// 	odisti = wlen_;   // Distance between start of k'th output 

			
		// 	plan_fwd_ = fftwf_plan_many_dft_r2c(rank, nf, howmanyf,
		// 											inf32_, inembed,
		// 											istride, idistf,
		// 											outf32_, onembed,
		// 											ostride, odistf,
		// 											patience_);

		// 	plan_inv_ = fftwf_plan_many_dft_c2r(rank, ni, howmanyi,
		// 											ini32_, inembed,
		// 											istride, idisti,
		// 											outi32_, onembed,
		// 											ostride, odisti,
		// 											patience_);

		// }


		// void plan_cc(float *arg_data_cc, int arg_ncc)
		// {
		// 	printf("Planning CC fft/ifft.. ");
		// 	clock_t tStart = clock();

		// 	ncc = arg_ncc;
		// 	data_cc = arg_data_cc;

		// 	fdata_cc = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * nfreq_ * ncc);
		// 	// fdata_cc = float (*fdata)[2]

		// 	plan3 = fftwf_plan_many_dft_c2r(1, &npts_, ncc, fdata_cc, NULL, 1, nfreq_, data_cc, NULL, 1, npts_, patience_);

		// 	printf("Completed in %.3f seconds\n", (float)(clock() - tStart)/CLOCKS_PER_SEC);
		// }

		// void run_cc(int *ckeys)
		// {
		// 	printf("nsig_= %d nfreq_ = %d ncc = %d \n", nsig_, nfreq_, ncc);
		// 	process::correlate_all_parallel(fdata, nsig_, nfreq_, fdata_cc, ckeys, ncc,  nthreads);
		// 	// process::correlate_all(fdata, nsig_, nfreq_, fdata_cc, ckeys, ncc);
		// }


};
#endif
