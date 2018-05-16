/*
Some thin wrappers around data, mainly so I dont have to pass
nsig, npts to every function. Should probably use some existing
ndim array library.
*/


#ifndef STRUCTURES_H
#define STRUCTURES_H
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <algorithm>

/// namespace structures {

template <typename T>
class Vector {	
public:
	size_t size_ = 0;
	T *data_ = nullptr;
	bool owns_ = true;


	Vector() {}
	// init from existing c-array with optional ownership
	Vector(T *data, size_t size, bool owns=true)
	:size_(size), data_(data), owns_(owns)
	{}

	// init and allocate dynamic memory
	Vector(size_t size)
	: size_(size), data_(size_ ? new T[size_]() : nullptr), owns_(true)
	{}

	// init from std::vector, copies data
	Vector(std::vector<T>& vec): size_(vec.size()), data_(new T[size_]()), owns_(true){
		std::copy(vec.begin(), vec.end(), data_);
	} 


	// destructor
	~Vector(){if (owns_ == true) {delete [] data_;}}
	
	// copy-assignment contructor
	Vector(const Vector& other) : size_(other.size_),
		  data_(size_ ? new T[size_]() : nullptr), owns_(true)
	{
		std::cout << "Warning: copying Vector of size " << size_ << '\n';
		std::copy(other.data_, other.data_ + size_, data_);
	}

	// helper for move-constructor and assignment operator
	friend void swap(Vector& first, Vector& second) // nothrow
	{
		// https://stackoverflow.com/questions/3279543/what-is-the-copy-and-swap-idiom
		using std::swap;
		swap(first.size_, second.size_);
		swap(first.data_, second.data_);
	}

	// overload assignment operator
	Vector& operator=(Vector other)
	{
		swap(*this, other); 

		return *this;
	}

	// move constructor
	Vector(Vector&& other)
		: Vector() 
	{
		swap(*this, other);
	}

	void arange(T start, T stop, T step){
		size_t size = (stop - start) / step;

		for (size_t i = 0; i < size; ++i) {
			data_[i] = start + i * step;			
		}
	}

	void linspace(float start, float stop){
		// size_ =  / step;
		float step = (stop - start) / static_cast<float>(size_);

		for (size_t i = 0; i < size_; ++i) {
			data_[i] = start + step * static_cast<float>(i);
		}
	}

	T& operator[] (size_t ix) {return data_[ix];}
	T& operator() (size_t ix){return data_[ix];}
	T* ptr(size_t ix) {
		return data_ + ix;
	}
	T* begin() {return data_;}
	T* end() {return data_ + size_;}

	size_t argmax(){
		return std::distance(data_, std::max_element(data_, data_ + size_));
	}

	T max(){return *std::max_element(data_, data_ + size_);}
	void fill(T value){std::fill(data_, data_ + size_, value);}
	
	void multiply(T value){
		for (size_t i = 0; i < size_; ++i) {
			data_[i] *= value;
		}
	}

	T sum(){
		T total = 0;
		for (size_t i = 0; i < size_; ++i) {
			total += data_[i];
		}
		return total;
	}

	T energy(){
		T total = 0;
		for (size_t i = 0; i < size_; ++i) {
			total += data_[i] * data_[i];
		}
		return total;
	}
	
};

template <typename T>
class Array2D {
public:
	// std::unique_ptr<T> data_ = nullptr;
	size_t nrow_ = 0;
	size_t ncol_ = 0;
	size_t size_ = 0;
	// size_t shape_[2];
	T *data_ = nullptr;
	bool owns_ = true;


	// default constructor
	Array2D() {}

	// init from existing c-array with optional ownership
	Array2D(T *data, size_t nrow, size_t ncol, bool owns=true)	 
	:data_(data), nrow_(nrow), ncol_(ncol), size_(nrow * ncol), owns_(owns)
	{}

	// init and allocate dynamic memory
	Array2D(size_t nrow, size_t ncol)
	: nrow_(nrow), ncol_(ncol), owns_(true){
		size_ = (size_t) nrow_ * ncol_;
		data_ = size_ ? new T[size_]() : nullptr;
	}

	// init from std::vector, copies data
	Array2D(std::vector<T>& vec, size_t ncol): nrow_(vec.size() / ncol), ncol_(ncol), size_(vec.size()), owns_(true){
		data_ = new T[size_]();
		std::copy(vec.begin(), vec.end(), data_);
	} 


	// destructor
	~Array2D(){if (owns_ == true) {delete [] data_;}}
	
	// copy-assignment contructor (needed to create std::vector of arrays)
	Array2D(const Array2D& other) : nrow_(other.nrow_), ncol_(other.ncol_), size_(other.size_),
		  data_(size_ ? new T[size_]() : nullptr), owns_(true)
	{
		std::copy(other.data_, other.data_ + size_, data_);
		std::cout << "Warning: copying Array2D of size " << size_ << '\n';
	}

	// helper for move-constructor and assignment operator
	friend void swap(Array2D& first, Array2D& second) // nothrow
	{
		// https://stackoverflow.com/questions/3279543/what-is-the-copy-and-swap-idiom
		using std::swap;
		// std::cout << "MOVING:********************" << second.size_ << '\n';
		swap(first.nrow_, second.nrow_);
		swap(first.ncol_, second.ncol_);
		swap(first.size_, second.size_);
		swap(first.data_, second.data_);
	}

	// overload assignment operator
	Array2D& operator=(Array2D other)
	{
		swap(*this, other); 
		return *this;
	}

	// move constructor
	Array2D(Array2D&& other)
		: Array2D() 
	{
		swap(*this, other);
	}



	// Array2D(const Array2D &that) {
	// 	nrow_ = that.nrow_;
	// 	ncol_ = that.ncol_;
	// 	size_ = that.size_;
	// 	shape_[0] = nrow_;
	// 	shape_[1] = ncol_;
	// 	// data_ = that.data_;
	// 	// owns_ = false;
	// 	data_ = new T[size_];
	// 	std::copy(that.data_, that.data_ + size_, data_);
	// 	// data_ = that.data_;
	// 	owns_ = true;		
	// }

	// Get value at flattened index ix
	T& operator[] (size_t ix){return data_[ix];}


	// Get value at simulated (row, col)
	T& operator() (size_t ix_row, size_t ix_col){
		return data_[ix_row * ncol_ + ix_col];
	}


	// Return pointer to i'th row
	T* row(size_t irow) {
		return data_ + (irow * ncol_);
	}

	// Return pointer to zeroth element
	T* begin() {return data_;}
	T* end() {return data_ + size_;}

	Vector<T> sum_rows() {

		Vector<T> vec = Vector<T>(ncol_);
		std::copy(data_, data_ + ncol_, vec.data_);

		T *out_ptr = nullptr;

		// sum each row after first			
		for (size_t i = 1; i < nrow_; ++i)
		{
			out_ptr = data_ + i * ncol_;

			for (size_t j = 0; j < ncol_; ++j) {
				vec[j] += out_ptr[j];
			}				
		}

		return vec;	
	}

	Array2D<T> transpose() {

		auto arr = Array2D<T>(ncol_, nrow_);


		T *out_ptr = nullptr;

		for (size_t i = 0; i < nrow_; ++i)
		{
			out_ptr = data_ + i * ncol_;

			for (size_t j = 0; j < ncol_; ++j) {
				arr(j, i) = out_ptr[j];
			}				
		}

		return arr;	
	}

	void arange(T start, T stop, T step){
		size_ = (stop - start) / step;

		for (size_t i = 0; i < size_; ++i) {
			data_[i] = start + i * step;			
		}
	}

	void multiply(T value){
		for (size_t i = 0; i < size_; ++i) {
			data_[i] *= value;
		}
	}

	Vector<T> copy_col(size_t icol) {

		Vector<T> vcopy = Vector<T>(nrow_);
		
		for (size_t i = 0; i < nrow_; ++i) {
			vcopy[i] = data_[i * ncol_ + icol];
		}
		
		return vcopy;	
	}

	Vector<T> copy_row(size_t irow) {

		Vector<T> vcopy = Vector<T>(ncol_);
		T *start = data_ + irow * ncol_;

		std::copy(start, start + ncol_, vcopy.data_);
		return vcopy;		
	}

	Vector<T> row_view(size_t irow) {

		T *start = data_ + irow * ncol_;
		Vector<T> vec = Vector<T>(start, ncol_, false);
		return vec;		
	}

	// Array2D<T> copy()
	// {
	// 	auto acopy = Array2D<T>(nrow_, ncol_);
	// 	std::copy(data_, data_ + size_, acopy.data_);
	// 	return acopy;
	// }
	
	void fill(T value){std::fill(data_, data_ + size_, value);}	
};



class Grid {
public:
	// e.g (xmin, xmax, ymin, ymax, zmin, zmax, spacing)
	std::vector<float> lims;
	// (dx, dy, dz)
	float spacing;


	float xmin, ymin, zmin;
	float x, y, z;
	// float zmax = lims[5];
	size_t nx, ny, nz;
	size_t ix, iy, iz;
	size_t npts;
	size_t size;
	// size_t size = 0;

	Grid() {}
	Grid(std::vector<float> lims):
	lims(lims), spacing(lims[6]), xmin(lims[0]), ymin(lims[2]), zmin(lims[4]){
		nx = (lims[1] - lims[0]) / spacing;
		ny = (lims[3] - lims[2]) / spacing;
		nz = (lims[5] - lims[4]) / spacing;
		npts = (size_t) nx * ny * nz;
		size = npts * 3;
		printf("Grid (%lu x %lu x %lu) = %lu\n", nx, ny, nz, npts);
	}
	~Grid(){}


	Array2D<float> build_points(){

		auto points = Array2D<float>(npts, 3);
		float* row_ix = points.data_;
		
		for (size_t i = 0; i < nz; ++i) {
			for (size_t j = 0; j < ny; ++j) {
				for (size_t k = 0; k < nx; ++k) {					
					row_ix[0] = xmin + k * spacing;
					row_ix[1] = ymin + j * spacing;
					row_ix[2] = zmin + i * spacing;
					row_ix += 3;
				}			
			}
		}
		return points;
	}

	size_t get_index(float *point){
		
		x = point[0];
		y = point[1];
		z = point[2];
		ix = (x - xmin) / spacing;
		iy = (y - ymin) / spacing;
		iz = (z - zmin) / spacing;
		return (iz * nx * ny) + (iy * nx) + ix;
	}

	void get_point(size_t index, float *buf){
				
		ix = index % nx;
		iy = ((index - ix) / nx) % ny;
		iz = index / (nx * ny);

		x = ix * spacing + xmin;
		y = iy * spacing + ymin;
		z = iz * spacing + zmin;
		buf[0] = x;
		buf[1] = y;
		buf[2] = z;
	}
	
};




struct Clock {
	typedef std::chrono::steady_clock clock;	
	typedef std::chrono::microseconds micro;
	// typedef std::chrono::milliseconds ms;
	// typedef std::chrono::system_clock clock;	
	typedef std::pair<std::string, micro> stamp;
	std::vector<stamp> stamps;
	std::chrono::time_point<clock> t0, tnow;

	Clock(){start();}
	
	void start(){
		t0 = clock::now();
	}

	void stop(std::string name){
		tnow = clock::now();
		stamps.push_back(stamp(name, std::chrono::duration_cast<micro>(tnow - t0)));
	}
	
	void log(std::string name){
		tnow = clock::now();
		stamps.push_back(stamp(name, std::chrono::duration_cast<micro>(tnow - t0)));
		t0 = tnow;
	}

	void print(){
		double elapsed;
		std::cout.precision(10);
		std::cout << "_____________________________________________\n";
		std::cout << std::left << std::setw(20) << "Name";
		std::cout << std::left << std::setw(20) << "Time (ms)" << "\n";
		std::cout << "_____________________________________________\n";

		for (auto&& stamp: stamps) {
			elapsed = (double) stamp.second.count() / 1000.;
			// std::cout << stamp.first << ": " << elapsed << " ms \n";
			// std::cout << elapsed << " ms   (" << stamp.first << ")\n";
			std::cout << std::left << std::setw(20) << stamp.first;
			std::cout << std::left << std::setw(20) << elapsed << "\n";
		}
	}	
};


// }
#endif

