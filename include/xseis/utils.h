#ifndef UTILS_H
#define UTILS_H
#include <iostream>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <typeinfo>
#include <random>
#include <type_traits>
#include <functional>
#include <map>

#include "xseis/structures.h"

namespace utils {

template<typename T>
size_t PadToBytes(const size_t size, const uint32_t nbytes=CACHE_LINE)
{    
	const uint32_t paddingElements = nbytes / sizeof(T);    
	const uint32_t mod = size % paddingElements;
	uint32_t ipad;
	
	mod == 0 ? ipad = 0 : ipad = paddingElements - mod;
	return size + ipad;	
}

std::vector<size_t> WinsAlignedF32(size_t npts, size_t wlen, float overlap) 
{
	std::vector<size_t> wix;

	float dxf = (1 - overlap) * wlen;
	assert(dxf > 1);

	size_t dx = PadToBytes<float>(dxf);

	std::cout << "actual_overlap: " <<  1.0 - (float) dx / wlen << "%\n";
	size_t ix = 0;
	while(ix < (npts - wlen)) {
		wix.push_back(ix);
		ix += dx;
	}
	return wix;				
}


std::vector<size_t> OverlappingWindows(size_t npts, size_t wlen, float overlap) 
{
	std::vector<size_t> wix;
	size_t ix = 0;
	while(ix < (npts - wlen)) {
		wix.push_back(ix);
		ix += (1 - overlap) * wlen;
	}
	return wix;				
}


template<typename T>
std::vector<T> linspace(T start, T stop, size_t size){

		std::vector<T> vec;
		vec.reserve(size);

		float step = (stop - start) / static_cast<float>(size);
		// std::cout << "step: " << step << '\n';

		for (size_t i = 0; i < size; ++i) {
			// vec[i] = start + step * static_cast<float>(i);
			vec.push_back(start + step * static_cast<float>(i));
		}

		return vec;
	}

	
template<typename T>
std::vector<T> arange(T start, T stop, T step = 1) {
	std::vector<T> values;
	values.reserve((stop - start) / step);
	
	for (T value = start; value < stop; value += step)
		values.push_back(value);
	return values;
}


// Set first N elements of array with randomly chosen elements
template<class BidiIter >
void RandomShuffle(BidiIter begin, BidiIter end, size_t num_random) {
	size_t left = std::distance(begin, end);
	while (num_random--) {
		BidiIter r = begin;
		std::advance(r, rand()%left);
		std::swap(*begin, *r);
		++begin;
		--left;
	}
}

template<typename T, typename UInt>
Array2D<T> ShuffleRows(Array2D<T>& data, Vector<UInt>& keys) {
	
	auto kc = keys.copy();
	std::srand(std::time(0));
	std::random_shuffle(kc.begin(), kc.end());

	auto dnew = Array2D<T>(data.nrow_, data.ncol_);

	float *drow = nullptr;
	for(unsigned i = 0; i < keys.size_; ++i) {
		drow = data.row(kc[i]);
		std::copy(drow, drow + data.ncol_, dnew.row(i));		
	}

	return dnew;

}


template<typename T, typename UInt>
void ShuffleRows(Array2D<T>& data, Vector<UInt>& keys, Array2D<T>& dshuff) {
	
	auto kc = keys.copy();
	std::srand(std::time(0));
	std::random_shuffle(kc.begin(), kc.end());

	float *drow = nullptr;
	for(unsigned i = 0; i < keys.size_; ++i) {
		drow = data.row(kc[i]);
		std::copy(drow, drow + data.ncol_, dshuff.row(i));		
	}

}


template<typename T>
void CopyArrayData(Array2D<T>& a, Array2D<T>& b) {	
	std::copy(a.begin(), a.end(), b.begin());
}


template<typename T>
void FillRandInt(Vector<T>& d, T min, T max)
{
	std::mt19937::result_type seed = time(0);
	auto rand = std::bind(std::uniform_int_distribution<T>(min, max - 1), std::mt19937(seed));
	for(size_t i = 0; i < d.size_; ++i) {
		d[i] = rand();
	}
}

template<typename T>
void FillRandFloat(T& d, float min, float max, uint rseed=1)
{
	// std::mt19937::result_type seed = time(0);
	std::mt19937::result_type seed = rseed;
	auto rand = std::bind(std::uniform_real_distribution<float>(min, max), std::mt19937(seed));
	for(size_t i = 0; i < d.size_; ++i) {
		d[i] = rand();
	}
}


// std::mt19937 get_prng() {
// 	std::random_device r;
// 	std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
// 	return std::mt19937(seed);
// }

// 	std::uniform_real_distribution<float> drange_x(limits[0], limits[3]);
// 	std::uniform_real_distribution<float> drange_y(limits[1], limits[4]);
// 	std::uniform_real_distribution<float> drange_z(limits[2], limits[5]);

// 	std::array<float, 3> curr_coord;
// 	curr_coord = {drange_x(e2), drange_y(e2), drange_z(e2)};



std::vector<uint32_t> gen_equal_parts(uint32_t ntot, uint32_t nthread) 
{
	uint32_t chunk = ntot / nthread;
	uint32_t remaining = ntot % nthread;

	std::vector<uint32_t> psizes(nthread, chunk);
	for (int i = 0; i < remaining; ++i) {psizes[i] += 1;}

	std::vector<uint32_t> partitions(nthread);
	partitions.push_back(0);
	std::partial_sum(psizes.begin(), psizes.end(), partitions.begin() + 1);
	return partitions;
}


void CopyArrayDataOffsetKeys(Array2D<float>& data_in, Vector<uint32_t>& keeprows, uint32_t offset, Array2D<float>& data_out)
{
	float *start_out;
	float *start_in;
	for (uint32_t i = 0; i < keeprows.size_; ++i)
	{	
		start_in = data_in.row(keeprows[i]) + offset;
		start_out = data_out.row(i);
		for (uint32_t j = 0; j < data_out.ncol_; ++j) {
			start_out[j] = start_in[j];
		}
	}
}

void CopyArrayDataOffset(Array2D<float>& data_in, uint32_t offset, Array2D<float>& data_out, size_t wlen=0)
{
	if (wlen == 0) wlen = data_out.ncol_;
	
	for (size_t i = 0; i < data_in.nrow_; ++i)
	{	
		float *start_in = data_in.row(i) + offset;
		float *start_out = data_out.row(i);
		
		for (size_t j = 0; j < wlen; ++j) {
			start_out[j] = start_in[j];
		}
	}
}


// void CopyArrayDataOffsetZpad(Array2D<float>& data_in, size_t offset, size_t wlen, Array2D<float>& data_out)
// {
// 	float *start_out;
// 	float *start_in;
// 	for (size_t i = 0; i < data_in.nrow_; ++i)
// 	{	
// 		start_in = data_in.row(i) + offset;
// 		start_out = data_out.row(i);
// 		for (size_t j = 0; j < wlen; ++j) {
// 			start_out[j] = start_in[j];
// 		}
// 	}
// }

// void copy_slice(Array2D<float>& data_in, Vector<uint32_t>& keeprows, uint32_t offset, Array2D<float>& data_out)
// {
// 	uint32_t cix;
// 	float *start;
// 	float *end;
// 	for (uint32_t i = 0; i < keeprows.size_; ++i)
// 	{	
// 		cix = keeprows[i];
// 		start = data_in.row(cix) + offset;
// 		end = data_in.row(cix) + offset + data_out.ncol_;;
// 		std::copy(start, end, data_out.row(i));
// 	}
// }

template <typename T>
void print(std::vector<T> &vec){	
	printf("\n[");
	std::cout.precision(4);
	int trunc = 100;

	for (int i = 0; i < vec.size(); ++i) {
		std::cout << vec[i] << " ";
		if (i > trunc) {printf("..\n"); break;}		
		// printf("%.1f ", float(data[i]));
	}
	printf("]\n");
}


template <typename T>
void print(Vector<T> &vec){	
	printf("\n[");
	std::cout.precision(4);
	int trunc = 100;

	for (int i = 0; i < vec.size_; ++i) {
		std::cout << vec[i] << " ";
		if (i > trunc) {printf("..\n"); break;}		
		// printf("%.1f ", float(data[i]));
	}
	printf("]\n");
}

template <typename T>
void print(Array2D<T> &arr){
	std::cout.precision(4);

	int wraplen = 0;
	int trunc = 20;

	for (unsigned i = 0; i < arr.nrow_; ++i)
	{	
		printf("[");		
		if (i > trunc) {printf("..\n"); break;}

		for (unsigned j = 0; j < arr.ncol_; ++j)
		{
			if (j > trunc) {printf("..\n"); break;}
			std::cout << arr(i, j) << " ";
		}
		printf("]\n");		
	}
	printf("shape: (%u, %u)]\n", arr.nrow_, arr.ncol_);
}


template <typename T>
void Write(std::string fname, Array2D<T>& arr){
	// std::cout.precision(10);
	std::ofstream myfile (fname);

	for (int i = 0; i < arr.nrow_; ++i) {	
		for (int j = 0; j < arr.ncol_; ++j) {
			myfile << arr(i, j) << " ";
		}
		myfile << "\n";
	}
	myfile.close();	
}

template <typename T>
void WritePowerGrid(std::string fname, Vector<T>& pwr, Grid& grid){

	std::ofstream myfile (fname);

	for (int i = 0; i < grid.lims.size(); ++i)
	{
		myfile << grid.lims[i] << " ";		
	}
	myfile << "\n";
	myfile << grid.nz << " " << grid.ny << " " << grid.nx << "\n";

	for (size_t i = 0; i < pwr.size_; ++i) {	
		myfile << pwr[i] << " ";
	}
	myfile.close();	
}


template <typename T>
void Write(std::string fname, Vector<T>& arr){
	// std::cout.precision(10);
	std::ofstream myfile (fname);

	for (int i = 0; i < arr.size_; ++i) {	
		myfile << arr[i] << "\n";
	}
	myfile.close();	
}

template <typename T>
void Write(std::string fname, std::vector<T> vec){
	// std::cout.precision(10);
	std::ofstream myfile (fname);

	for (int i = 0; i < vec.size(); ++i) {	
		myfile << vec[i] << "\n";
	}
	myfile.close();	
}


template <typename T>
void AppendToFile(std::string fname, T value){

	std::ofstream outfile;
	outfile.open(fname, std::ios_base::app);
	outfile << value << "\n";
}

template <typename T>
void AppendToFile(std::string fname, std::vector<T> vec){

	std::ofstream outfile;
	outfile.open(fname, std::ios_base::app);
	for(auto&& val : vec) {
		outfile << val << " ";		
	}
	outfile << "\n";
}


// void write_xyz_win(std::string fname, Array2D<uint32_t>& arr, Grid& grid){
// 	// std::cout.precision(10);
// 	std::ofstream myfile (fname);

// 	float loc[3];
// 	// auto locwin = Array2D<float>(win_loc.size_, 3);

// 	for (int i = 0; i < arr.nrow_; ++i) {	
// 		for (int j = 0; j < arr.ncol_; ++j) {
// 			grid.get_point(arr(i, j), loc);

// 			for (int k = 0; k < 3; ++k)
// 			{
// 				myfile << loc[k] << " ";
// 			}
// 		}
// 		myfile << "\n";
// 	}
// 	myfile.close();	

// }


template <typename T>
void print_time_sigs(T* data, int nsig, int npts){
	for (int i = 0; i < nsig * npts; ++i) {
		if (i % npts == 0){printf("\n");}
			printf("%.5f ", float(data[i]));
	}
	printf("\n");
}

template <typename T>
void print_freq_sigs(T* fdata, int nsig, int npts){
	for (int i = 0; i < nsig * npts; ++i) {
		if (i % npts == 0){printf("\n");}
			printf("[%.5f %.5fi] ", fdata[i][0], fdata[i][1]);
	}
	printf("\n");
}

template <typename T>
void PrintArraySize(Array2D<T>& arr){
	// T val = arr[0];
	float sizemb = (float) arr.size_ * sizeof(arr[0]) / (1024 * 1024);
	printf("(%u x %u) = %lu, %.1f mb (type: %s) \n", arr.nrow_, arr.ncol_,
			 arr.size_, sizemb,  typeid(arr[0]).name());
	// printf("(%u x %u), %.1f mb (%s) \n", arr.nrow_, arr.ncol_,
			 // sizemb,  typeid(arr[0]).name());
	// printf("type size %.1f bytes \n", (float) sizeof(arr[0]));
}

template <typename T>
void PrintArraySize(Array2D<T>& arr, std::string tag){

	float sizemb = (float) arr.size_ * sizeof(arr[0]) / (1024 * 1024);
	// printf("---------%s-------------\n", tag);
	std::cout << "\n" << tag << "--------------------------\n";
	printf("(%u x %u) = %lu, %.1f mb (type: %s) \n", arr.nrow_, arr.ncol_,
			 arr.size_, sizemb,  typeid(arr[0]).name());
	// printf("(%u x %u), %.1f mb (%s) \n", arr.nrow_, arr.ncol_,
			 // sizemb,  typeid(arr[0]).name());
	// printf("type size %.1f bytes \n", (float) sizeof(arr[0]));
}

std::string ZeroPadInt(unsigned val, unsigned npad=5){
	std::stringstream ss;
	ss << std::setw(npad) << std::setfill('0') << val;
	return ss.str();
}

void PrintMaxAndLoc(std::vector<float> v){
	printf("%.6f (max) @ [%.0f, %.0f, %.0f] \n", v[0], v[1], v[2], v[3]);
}

void PrintMaxAndLoc(uint32_t vmax, float* loc, float scale=10000.){
	printf("%.2f (max) @ [%.0f, %.0f, %.0f] \n", vmax / scale, loc[0], loc[1], loc[2]);
}

template <typename T>
void PrintVec(std::vector<T> v){
	for(auto& x : v) std::cout << x << ", ";		
	printf("\n");
}




// copy-assignment contructor (needed to create std::vector of arrays)
// template<typename T>
// Array2D(const Array2D& other) : nrow_(other.nrow_), ncol_(other.ncol_), size_(other.size_),
// 	  data_(size_ ? new T[size_]() : nullptr), owns_(true)
// {
// 	std::copy(other.data_, other.data_ + size_, data_);
// 	std::cout << "Warning: copying Array2D of size " << size_ << '\n';
// }


// template<typename T>
// Array2D<T> ConcatenateRows(Array2D<T>& a1, Array2D<T>& a2) {
	
// 	size_t nrow = a1.nrow_ + a2.nrow_;
// 	size_t ncol = a1.ncol_;

// 	auto ac = Array2D<T>(nrow, ncol);

// 	float *drow = nullptr;
// 	for(unsigned i = 0; i < keys.size_; ++i) {
// 		drow = data.row(kc[i]);
// 		std::copy(drow, drow + data.ncol_, dnew.row(i));		
// 	}

// 	return dnew;

// }

	// size_t argmax(){
	// 	return std::distance(data_, std::max_element(data_, data_ + size_));
	// }

	// T max(){return *std::max_element(data_, data_ + size_);}

}

#endif

