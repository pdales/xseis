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
// #include <omp.h>

#include "xseis/structures.h"

// typedef std::pair<float, std::array<float, 3> > vpair;
// typedef std::priority_queue<vpair, std::vector<vpair>, std::greater<vpair>> fe_queue;

namespace process {


// template<typename T>
// T max(T* begin, T* end) {
// 	return *std::max_element(begin, end);
// }

// template<typename T>
// T min(T* begin, T* end) {
// 	return *std::min_element(begin, end);
// }

// template<typename T>
// size_t argmax(T* begin, T* end) {
// 	return std::distance(begin, std::max_element(begin, end));
// }


template<typename Container>
float max(Container& data) {
	return *std::max_element(data.begin(), data.end());
}

template<typename Container>
float min(Container& data) {
	return *std::min_element(data.begin(), data.end());
}

// template<typename T>
// T min(T* begin, T* end) {
// 	return *std::min_element(begin, end);
// }


template<typename Container>
size_t argmax(Container& data) {
	return std::distance(data.begin(), std::max_element(data.begin(), data.end()));
}


inline float AngleBetweenPoints(float* a, float*b) 
{
	return std::atan((a[1] - b[1]) / (a[0] - b[0]));
	// return std::atan2(a[1] - b[1], a[0] - b[0]);
}

inline float DistCartesian(float* a, float* b)
{	
	float dx = a[0] - b[0];
	float dy = a[1] - b[1];
	float dz = a[2] - b[2];
	return std::sqrt(dx * dx + dy * dy + dz * dz);
}

inline float DistCartesian2D(float* a, float* b)
{	
	float dx = a[0] - b[0];
	float dy = a[1] - b[1];
	return std::sqrt(dx * dx + dy * dy);
}


float DistDiff(float* a, float* b, float* c) {	
	return DistCartesian(a, c) - DistCartesian(b, c);
}

uint mod_floor(int a, int n) {
	return ((a % n) + n) % n;
}


Vector<fftwf_complex> BuildPhaseShiftVec(size_t const nfreq, int const nshift) {
	
	auto v = Vector<fftwf_complex>(nfreq);
	// std::vector<fftwf_complex> v(nfreq);
	float const fstep = 0.5 / (nfreq - 1);
	float const factor = nshift * 2 * M_PI * fstep;

	for(size_t i = 0; i < nfreq; ++i) {
		v[i][0] = std::cos(i * factor);
		v[i][1] = std::sin(i * factor);			
	}

	return v;
}


// Mutiply sig1 by sig2 (x + yi)(u + vi) = (xu-yv) + (xv+yu)i
// x + yi = s1[0] + s1[1]i
// u + vi = s2[0] + s2[1]i
#pragma omp declare simd aligned(sig1, sig2:MEM_ALIGNMENT)
void Convolve(fftwf_complex const* const sig2, fftwf_complex* const sig1, uint32_t const nfreq)
{
	float tmp;
	#pragma omp simd aligned(sig1, sig2:MEM_ALIGNMENT)
	for (uint32_t i = 0; i < nfreq; ++i){
		tmp = sig1[i][0] * sig2[i][0] - sig1[i][1] * sig2[i][1];
		sig1[i][1] = sig1[i][0] * sig2[i][1] + sig1[i][1] * sig2[i][0];
		sig1[i][0] = tmp;
	}
}

#pragma omp declare simd aligned(sig1, sig2, out:MEM_ALIGNMENT)
inline void Convolve(fftwf_complex const* const sig1, fftwf_complex const* const sig2,
		   fftwf_complex* const out, uint32_t const nfreq)
{
	#pragma omp simd aligned(sig1, sig2, out:MEM_ALIGNMENT)
	for (uint32_t i = 0; i < nfreq; ++i){
		out[i][0] = sig1[i][0] * sig2[i][0] - sig1[i][1] * sig2[i][1];
		out[i][1] = sig1[i][0] * sig2[i][1] + sig1[i][1] * sig2[i][0];
	}
}


#pragma omp declare simd aligned(data, stack:MEM_ALIGNMENT)
inline void Accumulate(fftwf_complex const* const data, fftwf_complex* const stack,
						 uint32_t const npts)
{		
	#pragma omp simd aligned(data, stack:MEM_ALIGNMENT)
	for(uint32_t i = 0; i < npts; ++i) {
		stack[i][0] += data[i][0];
		stack[i][1] += data[i][1];
	}
}

#pragma omp declare simd aligned(data, stack:MEM_ALIGNMENT)
inline void Accumulate(float const* const data, float* const stack,
						 uint32_t const npts)
{		
	#pragma omp simd aligned(data, stack:MEM_ALIGNMENT)
	for(uint32_t i = 0; i < npts; ++i) {
		stack[i] += data[i];
	}
}

#pragma omp declare simd aligned(sig:MEM_ALIGNMENT)
void Whiten(fftwf_complex* const sig, uint32_t const npts)
{		
	#pragma omp simd aligned(sig:MEM_ALIGNMENT)
	for(uint32_t i = 0; i < npts; ++i) {
		float abs = std::sqrt(sig[i][0] * sig[i][0] + sig[i][1] * sig[i][1]);
		sig[i][0] /= abs;
		sig[i][1] /= abs;
	}
}

#pragma omp declare simd aligned(sig, out:MEM_ALIGNMENT)
void Absolute(fftwf_complex const* const sig, float* out, uint32_t const npts)
{		
	#pragma omp simd aligned(sig, out:MEM_ALIGNMENT)
	for(uint32_t i = 0; i < npts; ++i) {
		out[i] = std::sqrt(sig[i][0] * sig[i][0] + sig[i][1] * sig[i][1]);
	}
}

#pragma omp declare simd aligned(sig:MEM_ALIGNMENT)
void Absolute(float* sig, uint32_t const npts)
{		
	#pragma omp simd aligned(sig:MEM_ALIGNMENT)
	for(uint32_t i = 0; i < npts; ++i) {
		sig[i] = std::abs(sig[i]);
	}
}

// #pragma omp declare simd aligned(sig1, sig2:MEM_ALIGNMENT)
// void Convolve(fftwf_complex* sig1, fftwf_complex* sig2, uint32_t const nfreq)
// {
// 	float tmp;
// 	#pragma omp simd aligned(sig1, sig2:MEM_ALIGNMENT)
// 	for (uint32_t i = 0; i < nfreq; ++i){
// 		tmp = sig1[i][0] * sig2[i][0] - sig1[i][1] * sig2[i][1];
// 		sig1[i][1] = sig1[i][0] * sig2[i][1] + sig1[i][1] * sig2[i][0];
// 		sig1[i][0] = tmp;
// 	}
// }



// Cross-correlate complex signals, cc(f) = s1(f) x s2*(f)
#pragma omp declare simd aligned(sig1, sig2, out:MEM_ALIGNMENT)
void XCorr(fftwf_complex const* const sig1, fftwf_complex const* const sig2,
		   fftwf_complex* const out, uint32_t const nfreq)
{
	#pragma omp simd aligned(sig1, sig2, out:MEM_ALIGNMENT)
	for (uint32_t i = 0; i < nfreq; ++i){
		out[i][0] = sig1[i][0] * sig2[i][0] + sig1[i][1] * sig2[i][1];
		out[i][1] = sig1[i][0] * sig2[i][1] - sig1[i][1] * sig2[i][0];
	}
}

#pragma omp declare simd aligned(sig1, sig2:MEM_ALIGNMENT)
float DotProductEnergy(float const* const sig1, float const* const sig2, uint32_t const npts)
{
	float result = 0;
	#pragma omp simd aligned(sig1, sig2:MEM_ALIGNMENT)
	for (uint32_t i = 0; i < npts; ++i){
		// result += sig1[0] * sig2[0];		
		result += (sig1[0] * sig2[0]) * (sig1[0] * sig2[0]);		
	}
	return result;
}

#pragma omp declare simd aligned(sig1, sig2:MEM_ALIGNMENT)
float DotProduct(float const* const sig1, float const* const sig2, uint32_t const npts)
{
	float result = 0;
	#pragma omp simd aligned(sig1, sig2:MEM_ALIGNMENT)
	for (uint32_t i = 0; i < npts; ++i){
		result += sig1[0] * sig2[0];		
	}
	return result;
}


// // Cross-correlate signal pairs of fdata and output to fdata_cc
// void XCorrPairs(Array2D<fftwf_complex>& fdata, Array2D<uint16_t>& ckeys, Array2D<fftwf_complex>& fdata_cc)
// {	
// 	uint32_t nfreq = fdata.ncol_;

// 	#pragma omp for
// 	for (size_t i = 0; i < ckeys.nrow_; ++i)
// 	{
// 		// std::cout << "npair: " << i << '\n';
// 		XCorr(fdata.row(ckeys(i, 0)), fdata.row(ckeys(i, 1)),
// 			 fdata_cc.row(i), nfreq);
// 	}

// }


// #pragma omp declare simd aligned(data:MEM_ALIGNMENT)
template <typename T, typename F>
void ApplyFuncToRows(T *__restrict__ data, size_t nsig, size_t npts, F* func){
	// Generic map function

	// #pragma omp for simd aligned(data:MEM_ALIGNMENT)
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

void ApplyFreqFilterReplace(float (*fdata)[2], uint const nfreq, Vector<float>& filter)
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


float rms_energy(float *sig, size_t npts)
{
// np.sqrt(np.mean(data ** 2, axis=axis))

	float square_sum = 0;
	for (size_t i = 0; i < npts; ++i){
		square_sum += sig[i] * sig[i];
	}

	return std::sqrt(square_sum / npts);
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

void EMA_NoAbs(float *sig, size_t npts, uint wlen, bool both_ways=false)
{
	float alpha = 2 / (static_cast<float>(wlen) + 1);
	float beta = 1 - alpha;

	for (size_t i = 1; i < npts; ++i){
		sig[i] = alpha * sig[i] + beta * sig[i - 1];
	}

	if(both_ways == true) {
		for (long i = npts - 2; i >= 0; --i){
			sig[i] = alpha * sig[i] + beta * sig[i + 1];
		}
	}
}

float median(float *sig, size_t npts)
{
    size_t half = npts / 2;
    std::nth_element(sig, sig + half, sig + npts);
    return sig[half];
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

void zero_around_max(float *sig, size_t npts, size_t wlen)
{	
	// size_t amax = std::distance(sig, std::max_element(sig, sig + npts));	
	// size_t hlen = wlen / 2;

	// size_t cutmin = std::max(amax - hlen, (size_t) 0);
	// size_t cutmax = std::min(amax + hlen, (size_t) npts);

	// for(size_t i = cutmin; i < cutmax; ++i) {
	// 	sig[i] = 0;
	// }

	long amax = std::distance(sig, std::max_element(sig, sig + npts));	
	long hlen = wlen / 2;

	long cutmin = amax - hlen;
	long cutmax = amax + hlen;

	if(cutmin >= 0 && cutmax <= npts) {
		for(size_t i = cutmin; i < cutmax; ++i) {
			sig[i] = 0;
		}	
	}
	else if (cutmin < 0){
		for(size_t i = npts + cutmin; i < npts; ++i) {
			sig[i] = 0;
		}
		for(size_t i = 0; i < cutmax; ++i) {
			sig[i] = 0;
		}
	}
	else if (cutmax > npts){
		for(size_t i = cutmin; i < npts; ++i) {
			sig[i] = 0;
		}
		for(size_t i = 0; i < cutmax - npts; ++i) {
			sig[i] = 0;
		}
	}
}

void absolute(float *sig, size_t npts)
{
	for (size_t i = 0; i < npts; ++i){
		sig[i] = std::abs(sig[i]);
	}
}


void Roll(float* sig, size_t npts, long nroll)
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

	return std::sqrt(var / size);
}


template<typename T>
float mean(T *data, size_t size) {

	float mean = 0;
	for(size_t i = 0; i < size; ++i) {
		mean += data[i];
	}
	mean /= size;
	return mean;	
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


void Multiply(float *sig, size_t npts, float val){
	for (size_t i = 0; i < npts; ++i){
		sig[i] *= val;
	}
}

void Multiply(fftwf_complex* data, size_t npts, float val)
{		
	for(size_t i = 0; i < npts; ++i) {
		data[i][0] *= val;
		data[i][1] *= val;
	}
}


template<typename Container>
void Multiply(Container& data, float val) {
	Multiply(data.data_, data.size_, val);
}




void Fill(fftwf_complex* data, size_t npts, float val)
{		
	for(size_t i = 0; i < npts; ++i) {
		data[i][0] = val;
		data[i][1] = val;
	}
}

void Fill(float* data, size_t npts, float val)
{		
	for(size_t i = 0; i < npts; ++i) {
		data[i] = val;		
	}
}



void Fill(Vector<fftwf_complex>& data, float val)
{		
	for(size_t i = 0; i < data.size_; ++i) {
		data[i][0] = val;
		data[i][1] = val;
	}
}

void Fill(Vector<float>& data, float val)
{		
	for(size_t i = 0; i < data.size_; ++i) {
		data[i] = val;		
	}
}

void Copy(fftwf_complex const *in, size_t npts, fftwf_complex *out)
{		
	std::copy(&(in)[0][0], &(in + npts)[0][0], &out[0][0]);
}

void Copy(float const *in, size_t npts, float *out)
{		
	std::copy(in, in + npts, out);
}



void Subtract(fftwf_complex const *data, fftwf_complex *data_mod, size_t npts)
{		
	for(size_t i = 0; i < npts; ++i) {
		data_mod[i][0] -= data[i][0];
		data_mod[i][1] -= data[i][1];
	}
}


#pragma omp declare simd aligned(sig:MEM_ALIGNMENT)
float Energy(const fftwf_complex *sig, uint32_t const nfreq)
{
	// E = 1/N sum(|x(f)**2|)
	float tmp = 0;
	#pragma omp simd aligned(sig:MEM_ALIGNMENT)
	for (uint32_t i = 0; i < nfreq; ++i){
		tmp += sig[i][0] * sig[i][0] + sig[i][1] * sig[i][1];
	}
	return tmp / static_cast<float>(nfreq);
}

// Cross-correlate complex signals, cc(f) = s1(f) x s2*(f)
#pragma omp declare simd aligned(sig1, sig2:MEM_ALIGNMENT)
float XCorrEnergy(fftwf_complex const *sig1, fftwf_complex const *sig2, uint32_t const nfreq)
{
	float a, b;
	float sum = 0;
	#pragma omp simd aligned(sig1, sig2:MEM_ALIGNMENT)
	for (uint32_t i = 0; i < nfreq; ++i){
		a = (sig1[i][0] * sig2[i][0]) + (sig1[i][1] * sig2[i][1]);
		b = (sig1[i][0] * sig2[i][1]) - (sig1[i][1] * sig2[i][0]);
		sum += (a * a) + (b * b);

		// a = sig1[i][0] * sig2[i][0] + sig1[i][1] * sig2[i][1];
		// b = sig1[i][0] * sig2[i][1] - sig1[i][1] * sig2[i][0];
		// sum += (sig1[i][0] * sig2[i][0] + sig1[i][1] * sig2[i][1]) * (sig1[i][0] * sig2[i][0] + sig1[i][1] * sig2[i][1]) + (sig1[i][0] * sig2[i][1] - sig1[i][1] * sig2[i][0]) * (sig1[i][0] * sig2[i][1] - sig1[i][1] * sig2[i][0]);
	}

	return sum;
}


// def get_pt(index, shape, spacing, origin):
// 	nx, ny, nz = shape
// 	# nx, ny, nz = spacing
// 	iz = index % nz
// 	iy = ((index - iz) / nz) % ny
// 	ix = index / (nz * ny)

// 	loc = np.array([ix, iy, iz], dtype=np.float32) * spacing + origin
// 	return loc

std::vector<float> get_point(size_t index, int spacing, int* origin, int* shape){
	
	int nx = shape[0];			
	int ny = shape[1];			
	int nz = shape[2];
	int iz = index % nz;
	int iy = ((index - iz) / nz) % ny;
	int ix = index / (nz * ny);

	std::vector<float> v(3);
	v[0] = ix * spacing + origin[0];
	v[1] = iy * spacing + origin[1];
	v[2] = iz * spacing + origin[2];
	return v;
}




}

#endif
