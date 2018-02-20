#ifndef UTILS_H
#define UTILS_H
#include <iostream>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <typeinfo>
#include "xseis/structures.h"

namespace utils {

Array2D<uint16_t> unique_pairs(Vector<uint16_t>& keys)
{
	uint64_t npair = 0;

	// crude way to calc nkeys (wil use dist filters later)
	for (int i = 0; i < keys.size_; ++i)
	{
		for (int j = i + 1; j < keys.size_; ++j)
		{
			npair += 1;			
		}
	}

	auto ckeys = Array2D<uint16_t>(npair, 2);
	uint64_t row_ix = 0;

	for (int i = 0; i < keys.size_; ++i)
	{
		for (int j = i + 1; j < keys.size_; ++j)
		{
			ckeys(row_ix, 0) = keys[i];
			ckeys(row_ix, 1) = keys[j];
			row_ix += 1;
		}
	}
	return ckeys;
}


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


void copy_slice(Array2D<float>& data_in, Vector<uint32_t>& keeprows, uint32_t offset, Array2D<float>& data_out)
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

	for (int i = 0; i < arr.nrow_; ++i)
	{	
		printf("[");		
		if (i > trunc) {printf("..\n"); break;}

		for (int j = 0; j < arr.ncol_; ++j)
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

	for (uint32_t i = 0; i < pwr.size_; ++i) {	
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


void write_xyz_win(std::string fname, Array2D<uint32_t>& arr, Grid& grid){
	// std::cout.precision(10);
	std::ofstream myfile (fname);

	float loc[3];
	// auto locwin = Array2D<float>(win_loc.size_, 3);

	for (int i = 0; i < arr.nrow_; ++i) {	
		for (int j = 0; j < arr.ncol_; ++j) {
			grid.get_point(arr(i, j), loc);

			for (int k = 0; k < 3; ++k)
			{
				myfile << loc[k] << " ";
			}
		}
		myfile << "\n";
	}
	myfile.close();	

}


template <typename T>
void print_time_sigs(T* data, int nsig, int npts){
	for (int i = 0; i < nsig * npts; ++i) {
		if (i % npts == 0){printf("\n");}
			printf("%.3f ", float(data[i]));
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

std::string ZeroPadInt(unsigned val, unsigned npad){
	std::stringstream ss;
	ss << std::setw(npad) << std::setfill('0') << val;
	return ss.str();
}

}

#endif

