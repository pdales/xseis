/*
Signal processing functions.

Dont notice performance difference using omp simd aligned.
Seems that c++ vectorizes by default when optimizations on.
*/

#ifndef PROCESS_H
#define PROCESS_H

#define _USE_MATH_DEFINES
#include <cmath>

#include <iostream>
#include <vector>
#include <array>
#include <algorithm>
#include <fftw3.h>
// #include <queue>
// #include <random>
#include <thread>
#include <functional>
#include <chrono>
#include <omp.h>

#include "xseis/structures.h"

const uint MEM_ALIGNMENT = 16;

// typedef std::pair<float, std::array<float, 3> > vpair;
// typedef std::priority_queue<vpair, std::vector<vpair>, std::greater<vpair>> fe_queue;

namespace process {


// Cross-correlate complex signals, cc(f) = s1(f) x s2*(f)
#pragma omp declare simd aligned(sig1, sig2, out:MEM_ALIGNMENT)
void XCorr(fftwf_complex* sig1, fftwf_complex* sig2, fftwf_complex* out, uint32_t nfreq)
{
	#pragma omp simd aligned(sig1, sig2, out:MEM_ALIGNMENT)
	for (uint32_t i = 0; i < nfreq; ++i){
		out[i][0] = sig1[i][0] * sig2[i][0] + sig1[i][1] * sig2[i][1];
		out[i][1] = sig1[i][0] * sig2[i][1] - sig1[i][1] * sig2[i][0];
	}
}

// Cross-correlate signal pairs of fdata and output to fdata_cc
void XCorrPairs(Array2D<fftwf_complex>& fdata, Array2D<uint16_t>& pairs, Array2D<fftwf_complex>& fdata_cc)
{	
	uint32_t nfreq = fdata.ncol_;

	#pragma omp for
	for (size_t i = 0; i < pairs.nrow_; ++i)
	{
		XCorr(fdata.row(pairs(i, 0)), fdata.row(pairs(i, 1)),
			 fdata_cc.row(i), nfreq);
	}

}


#pragma omp declare simd aligned(data:MEM_ALIGNMENT)
template <typename T, typename F>
void ApplyFuncToRows(T *__restrict__ data, size_t nsig, size_t npts, F* func){
	// Generic map function

	#pragma omp for simd aligned(data:MEM_ALIGNMENT)
	for (size_t i = 0; i < nsig; i++)
	{
		(*func)(data + (i * npts), npts);
	}
}

template <typename T, typename F>
void ApplyFuncToRows(Array2D<T>& data, F* func){
	ApplyFuncToRows(data.data_, data.nrow_, data.ncol_, func);	
}



Vector<float> BuildFreqFilter(std::vector<float>& corner_freqs, uint nfreq, float sr)
{

	float fsr = (nfreq * 2 - 1) / sr;
	printf("nfreq: %u, FSR: %.4f\n", nfreq, fsr);

	std::vector<uint32_t> cx;
	for(auto&& cf : corner_freqs) {
		cx.push_back(static_cast<uint32_t>(cf * fsr + 0.5));
		// printf("cf/fsr %.2f, %.5f\n", cf, fsr);
	}
	printf("filt corner indexes \n");
	for(auto&& c : cx) {
		// printf("cx/ cast: %.3f, %u\n", cx, (uint32_t)cx);
		printf("--%u--", c);
	}
	printf("\n");

	// whiten corners:  cutmin--porte1---porte2--cutmax
	auto filter = Vector<float>(nfreq);
	filter.fill(0);

	// int wlen = porte1 - cutmin;
	float cosm_left = M_PI / (2. * (cx[1] - cx[0]));
	// left hand taper
	for (uint i = cx[0]; i < cx[1]; ++i) {
		filter[i] = std::pow(std::cos((cx[1] - (i + 1) ) * cosm_left), 2.0);
	}

	// setin middle freqs amp = 1
	for (uint i = cx[1]; i < cx[2]; ++i) {
		filter[i] = 1;
	}

	float cosm_right = M_PI / (2. * (cx[3] - cx[2]));

	// right hand taper
	for (uint i = cx[2]; i < cx[3]; ++i) {
		filter[i] = std::pow(std::cos((i - cx[2]) * cosm_right), 2.0);
	}

	return filter;	

}

void ApplyFreqFilterReplace(float (*fdata)[2], uint nfreq, Vector<float>& filter)
{
	float angle;

	for (uint i = 0; i < filter.size_; ++i)
	{
		if(filter[i] == 0) {
			fdata[i][0] = 0;
			fdata[i][1] = 0;
		}
		else {
			angle = std::atan2(fdata[i][1], fdata[i][0]);
			fdata[i][0] = filter[i] * std::cos(angle);
			fdata[i][1] = filter[i] * std::sin(angle);
		}		
	}

}

void ApplyFreqFilterMultiply(float (*fdata)[2], uint nfreq, Vector<float>& filter)
{
	float angle;

	for (uint i = 0; i < filter.size_; ++i)
	{
		if(filter[i] == 0) {
			fdata[i][0] = 0;
			fdata[i][1] = 0;
		}
		else {
			angle = std::atan2(fdata[i][1], fdata[i][0]);
			fdata[i][0] *= filter[i] * std::cos(angle);
			fdata[i][1] *= filter[i] * std::sin(angle);
		}		
	}
}


void square_signal(float *sig, size_t npts)
{
	for (size_t i = 0; i < npts; ++i) {
		sig[i] = sig[i] * sig[i];
	}
}

void root_signal(float *sig, size_t npts)
{
	for (size_t i = 0; i < npts; ++i) {
		sig[i] = std::sqrt(sig[i]);
	}
}



void multiply(float *sig, size_t npts, float factor){
	for (size_t i = 0; i < npts; ++i){
		sig[i] *= factor;
	}
}

void clip(float *sig, size_t npts, float thresh){
	for (size_t i = 0; i < npts; ++i){
		if (sig[i] > thresh){sig[i] = thresh;}
		else if (sig[i] < -thresh){sig[i] = -thresh;}
	}
}

void demean(float *sig, size_t npts)
{
	float mean = 0;
	for (size_t i = 0; i < npts; ++i){
		mean += sig[i];
	}

	mean /= npts;

	for (size_t i = 0; i < npts; ++i){
		sig[i] -= mean;
	}
}

void norm_one_bit(float *sig, size_t npts)
{
	for (size_t i = 0; i < npts; ++i){
		sig[i] = (sig[i] > 0) - (sig[i] < 0);
	}
}

void norm_one_or_zero(float *sig, size_t npts)
{
	for (size_t i = 0; i < npts; ++i){
		if(sig[i] <= 0) {
			sig[i] = 0;
		}
		else{
			sig[i] = 1;
		}
	}
}


void ExpMovingAverage(float *sig, size_t npts, uint wlen, bool both_ways=false)
{
	float alpha = 2 / (static_cast<float>(wlen) + 1);
	float beta = 1 - alpha;
	
	sig[0] = std::abs(sig[0]);

	for (size_t i = 1; i < npts; ++i){
		sig[i] = alpha * std::abs(sig[i]) + beta * sig[i - 1];
	}

	if(both_ways == true) {
		for (long i = npts - 2; i >= 0; --i){
		sig[i] = alpha * std::abs(sig[i]) + beta * sig[i + 1];
	}

	}
}


// void ExpMovingAverageSquare(float *sig, size_t npts, uint wlen)
// {
// 	float alpha = 2 / (static_cast<float>(wlen) + 1);
// 	float beta = 1 - alpha;
	
// 	sig[0] = sig[0] * sig[0];

// 	for (size_t i = 1; i < npts; ++i){
// 		sig[i] = alpha * sig[i] * sig[i] + beta * sig[i - 1];
// 	}
// }

		// esig[i] = alpha * esig[i] + (1 - alpha) * esig[i - 1]


// template<typename T>
// bool abs_compare(T a, T b)
bool abs_compare(float a, float b)
{
	return (std::abs(a) < std::abs(b));
}

void norm_max_abs(float *sig, size_t npts)
{
	float max = *std::max_element(sig, sig + npts, abs_compare);

	if (max != 0){
		for (size_t i = 0; i < npts; ++i){
			sig[i] /= max;
		}
	}
}

void absolute(float *sig, size_t npts)
{
	for (size_t i = 0; i < npts; ++i){
		sig[i] = std::abs(sig[i]);
	}
}


void roll(float* sig, size_t npts, int nroll)
{
	std::rotate(sig, sig + nroll, sig + npts);
}


void taper(float *sig, size_t npts, uint len_taper)
{
	float factor = (2 * M_PI) / ((len_taper * 2) - 1);
	float *sig_end = sig + npts - len_taper;

	for (size_t i = 0; i < len_taper; ++i) {
		sig[i] *= 0.5 - 0.5 * std::cos(i * factor);
	}
	for (size_t i = 0; i < len_taper; ++i) {
		sig_end[i] *= 0.5 - 0.5 * std::cos((i + len_taper) * factor);
	}
}



template<typename T>
float standard_deviation(T *data, size_t size) {

	float mean = 0;
	for(size_t i = 0; i < size; ++i) {
		mean += data[i];
	}
	mean /= size;

	float var = 0;
	for(size_t i = 0; i < size; ++i) {
		var += (data[i] - mean) * (data[i] - mean);
	}

	var /= size;
	return std::sqrt(var);
}




// void norm_energy(float (*sig)[2], int npts)
// {
// 	int nfreq = npts / 2 + 1;
// 	float energy = 0;

// 	for (int i = 0; i < nfreq; ++i) {
// 		energy += (sig[i][0] * sig[i][0] + sig[i][1] * sig[i][1]);
// 	}
// 	// printf("energy = %.5f \n", energy);
// 	// printf("nfreq = %d \n", nfreq);

// 	for (int i = 0; i < nfreq; ++i) {
// 		sig[i][0] /= energy;
// 		sig[i][1] /= energy;
// 	}
// }

void SlidingWinMax(float *sig, size_t npts, size_t wlen)
{	
	// Sliding window max abs val smoothin (horribly slow)
	
	absolute(sig, npts);

	if (wlen % 2 == 0){wlen += 1;}
	size_t hlen = wlen / 2 + 1;

	float buf[wlen];
	size_t buf_idx = 0;

	// Fill buffer with last WLEN vals of sig	
	std::copy(&sig[npts - wlen], &sig[npts], buf);


	// Handle edge case with index wrapin via mod function
	for (size_t i = npts - hlen; i < npts + hlen; ++i) {

		sig[i % npts] = *std::max_element(buf, buf + wlen);
		buf[buf_idx] = sig[(i + hlen) % npts];
		buf_idx = (buf_idx + 1) % wlen;
	}
	// handle non-edge case
	for (size_t i = hlen; i < npts - hlen; ++i) {

		sig[i] = *std::max_element(buf, buf + wlen);
		buf[buf_idx] = sig[i + hlen];
		buf_idx = (buf_idx + 1) % wlen;
	}

}


// void SlidingAverage(float *sig, size_t npts, size_t wlen)
// {
// 	if (wlen % 2 == 0){wlen += 1;}

// 	float buf[wlen];
// 	float curr_sum = 0;

// 	size_t buf_idx = 0;
// 	size_t hlen = wlen / 2 + 1;	

// 	// Do initial buffer fill
// 	for (size_t i = npts - wlen; i < npts; ++i) {
// 		buf[i] = sig[i];
// 		curr_sum += buf[i];
// 	}

// 	float wlen_f = static_cast<float>(wlen);
// 	float buf_in = sig[wlen];	
// 	size_t iwrap = 0;
// 	// Handle edge case with wrapping
// 	for (size_t i = npts - hlen; i < npts + hlen; ++i) {
// 		// iwrap = ;
// 		sig[i % npts] = curr_sum / wlen_f;

// 		buf_in = sig[(i + hlen) % npts];
// 		buf[buf_idx] = sig[(i - 1 + hlen) % npts];
// 		buf_idx = (buf_idx + 1) % wlen;

// 		curr_sum += buf_in - buf[buf_idx];

// 		// buf[i] = sig[iwrap];
// 		// curr_sum += sig[iwrap];
// 	}

// 		// 	if (i < half_len || i >= sig_len - half_len){
// 		// 	sig[i] = curr_sum / win_len;
// 		// 	continue;
// 		// }
// 		// curr_sum += buf_in - buf[buf_idx];
// 		// sig[i] = curr_sum / win_len;

// 		// buf_in = sig[i + half_len];
// 		// buf[buf_idx] = sig[i - 1 + half_len];
// 		// buf_idx = (buf_idx + 1) % win_len;

// 	// compute in place sliding avg using ring buffer
// 	for (size_t i = 0; i < npts; ++i) {
		
// 		curr_sum += buf_in - buf[buf_idx];
// 		sig[i] = curr_sum / wlen;

// 		buf_in = sig[i + hlen];
// 		buf[buf_idx] = sig[i - 1 + hlen];
// 		buf_idx = (buf_idx + 1) % wlen;
// 	}
// }

// void SlidingRMS(float *sig, size_t npts, size_t wlen)
// {
// 	square_signal(sig, npts);
// 	SlidingAverage(sig, npts, wlen);
// 	root_signal(sig, npts);
// }



// void correlate_all_parallel(float (*fdata)[2], int nsig, int nfreq, float (*fdata_cc)[2], uint32_t *pairs, int npairs, int nthreads)
// {
// 	std::vector<std::thread> pool;

// 	int start, end, npairs_chunk;
// 	float (*ptr_out)[2] = nullptr;
// 	uint32_t *ptr_pairs = nullptr;

// 	for (int i = 0; i < nthreads; i++){

// 		start = floor(i * npairs / nthreads);
// 		end = floor((i + 1) * npairs / nthreads);
// 		ptr_out = fdata_cc + (start * nfreq);
// 		ptr_pairs = pairs + (start * 2);

// 		npairs_chunk = end - start;
// 		// printf("range [%d: %d] npairs: %d \n", start, end, npairs_chunk);
// 		// printf("startval = %.2f \n", ptr_out[0][0]);

// 		pool.push_back(std::thread(correlate_all, fdata, nsig, nfreq, ptr_out,
// 									 ptr_pairs, npairs_chunk));
// 	}
// 	for (std::thread& t : pool){
// 		t.join();
// 	}
// }


// void correlate_all(float (*fdata)[2], int nsig, int nfreq, float (*fdata_cc)[2],
// 				 uint32_t *pairs, int npairs)
// {
// 	int key1, key2;
// 	float (*sig1)[2] = nullptr;
// 	float (*sig2)[2] = nullptr;
// 	float (*sigcc)[2] = nullptr;

// 	#pragma omp for
// 	for (int i = 0; i < npairs; ++i)
// 	{
// 		key1 = pairs[i * 2];
// 		key2 = pairs[i * 2 + 1];

// 		sig1 = fdata + (key1 * nfreq);
// 		sig2 = fdata + (key2 * nfreq);
// 		sigcc = fdata_cc + (i * nfreq);

// 		XCorr(sig1, sig2, sigcc, nfreq);
// 	}
// }


// void XCorrPairs(Array2D<fftwf_complex>& fdata, Array2D<uint32_t>& pairs, Array2D<fftwf_complex>& fdata_cc)
// {	
// 	uint32_t nfreq = fdata.ncol_;

// 	#pragma omp for
// 	for (uint32_t i = 0; i < pairs.nrow_; ++i)
// 	{
// 		XCorr(fdata.row(pairs(i, 0)), fdata.row(pairs(i, 1)),
// 			 fdata_cc.row(i), nfreq);
// 	}
// }


// void WhitenSpectrum(float (*fdata)[2], int nfreq, float sr, std::vector<float> filt_shape)
// {
// 	float fmin = filt_shape[0];
// 	float fmax = filt_shape[1];
// 	float len_taper_ratio = filt_shape[2];
// 	// uint len_taper = len_taper_ratio * nfreq;
// 	int len_taper = len_taper_ratio * nfreq;
// 	// printf("ntaper %d\n", len_taper);

// 	float fsr = (nfreq * 2 - 1) / sr;
// 	// whiten corners:  cutmin--porte1---porte2--cutmax
// 	int porte1 = fsr * fmin;
// 	int porte2 = fsr * fmax;

// 	int cutmin = std::max(porte1 - len_taper, 1);
// 	int cutmax = std::min(porte2 + len_taper, nfreq);
// 	float angle;
// 	float amp;

// 	// printf("%.8f fHz  %d npts_taper \n", fsr, npts_taper);
// 	// printf("%d %d %d %d \n", cutmin, porte1, porte2, cutmax);
// 	int wlen = porte1 - cutmin;
// 	float cosm = M_PI / (2. * wlen);

// 	// whiten signal from cutmin to cutmax
// 	for (int i = 0; i < cutmin; ++i) {
// 		fdata[i][0] = 0.0;
// 		fdata[i][1] = 0.0;
// 	}

// 	// left hand taper
// 	for (int i = cutmin; i < porte1; ++i) {
// 		amp = std::pow(std::cos((porte1 - (i + 1) ) * cosm), 2.0);
// 		angle = std::atan2(fdata[i][1], fdata[i][0]);
// 		fdata[i][0] = amp * std::cos(angle);
// 		fdata[i][1] = amp * std::sin(angle);
// 	}

// 	// setin middle freqs amp = 1
// 	for (int i = porte1; i < porte2; ++i) {
// 		angle = std::atan2(fdata[i][1], fdata[i][0]);
// 		fdata[i][0] = std::cos(angle);
// 		fdata[i][1] = std::sin(angle);
// 	}

// 	wlen = cutmax - porte2;
// 	cosm = M_PI / (2. * wlen);

// 	// right hand taper
// 	for (int i = porte2; i < cutmax; ++i) {
// 		amp = std::pow(std::cos((i - porte2) * cosm), 2.0);
// 		angle = std::atan2(fdata[i][1], fdata[i][0]);
// 		fdata[i][0] = amp * std::cos(angle);
// 		fdata[i][1] = amp * std::sin(angle);
// 	}
	
// 	for (int i = cutmax; i < nfreq; ++i) {
// 		fdata[i][0] = 0.0;
// 		fdata[i][1] = 0.0;
// 	}
// }


// template <typename T, typename F>
// void map_signals(T* data, int nsig, int npts, F* func){
// 	for (int i = 0; i < nsig; i++){
// 		(*func)(data + (i * npts), npts);
// 	}
// }


// template <typename T, typename F>
// void apply_to_signals(Array2D<T>& arr, F* func, uint32_t nthreads)
// {	
// 	uint32_t nsig = arr.nrow_;	
// 	uint32_t npts = arr.ncol_;	
// 	if (nthreads <= 1) {map_signals(arr.data_, nsig, npts, func);}

// 	else {
// 		std::vector<std::thread> pool;
// 		int start, end, nsig_chunk;
// 		T* dstart = nullptr;

// 		for (int i = 0; i < nthreads; i++){

// 			start = floor(i * nsig / nthreads);
// 			end = floor((i + 1) * nsig / nthreads);
// 			dstart = arr.data_ + (start * npts);
// 			nsig_chunk = end - start;
// 			// printf("%d %d\n", start, end);
// 			pool.push_back(std::thread(&map_signals<T, F>, dstart, nsig_chunk, npts, func));
// 		}

// 		for(auto& thread : pool) thread.join();
// 	}
// }



// template <typename T1, typename T2>
// void map_signals_parallel(T1* data, int nsig, int npts, T2* func, int nthreads)
// {
// 	if (nthreads <= 1) {map_signals(data, nsig, npts, func);}

// 	else {
// 		std::vector<std::thread> pool;
// 		int start, end, nsig_chunk;
// 		T1* dstart = nullptr;

// 		for (int i = 0; i < nthreads; i++){

// 			start = floor(i * nsig / nthreads);
// 			end = floor((i + 1) * nsig / nthreads);
// 			dstart = data + (start * npts);
// 			nsig_chunk = end - start;
// 			// printf("%d %d\n", start, end);
// 			pool.push_back(std::thread(&map_signals<T1, T2>, dstart, nsig_chunk, npts, func));
// 		}

// 		for(auto& thread : pool) thread.join();
// 	}
// }



// template <typename T1, typename T2>
// void map_signals_pool(T1* data, int nsig, int npts, T2* func, std::vector<std::thread> pool)
// {

// 	int start, end, nsig_chunk;
// 	T1* dstart = nullptr;
// 	uint int nthreads = pool.size();

// 	for (int i = 0; i < nthreads; i++){
// 	// for(std::vector<int>::uint32_type i = 0; i < pool.size(); i++) {

// 		start = floor(i * nsig / nthreads);
// 		end = floor((i + 1) * nsig / nthreads);
// 		dstart = data + (start * npts);
// 		nsig_chunk = end - start;
// 		// printf("%d %d\n", start, end);
// 		pool.push_back(std::thread(&map_signals<T1, T2>, dstart, nsig_chunk, npts, func));
// 	}

// 	for(auto& thread : pool) thread.join();

// }

// void WhitenSpectrum(float (*fdata)[2], int nfreq, float fsr, float fmin, float fmax, int len_taper)


// float energy_freq_domain(float (*sig)[2], int nfreq)
// {
// 	float energy = 0;

// 	for (int i = 0; i < nfreq; ++i) {
// 		energy += sig[i][0] * sig[i][0] + sig[i][1] * sig[i][1];
// 	}
// 	// multiply by 2 for symmetry
// 	return energy * 2;
// }



}

#endif
