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
		T *data_ = nullptr;
		uint32_t size_ = 0;

		Vector() {}
		Vector(T *data, uint32_t size): data_(data), size_(size) {}
		Vector(uint32_t size): size_(size) {data_ = new T[size_]();}

		~Vector(){delete data_;}
		void set_data_(T* arg) {data_ = arg;}

		void arange(T start, T stop, T step){
			size_ = (stop - start) / step;

			for (uint32_t i = 0; i < size_; ++i) {
				data_[i] = start + i * step;			
			}
		}

		T& operator[] (uint32_t ix) {return data_[ix];}
		T& operator() (uint32_t ix){return data_[ix];}
		T* ptr(uint32_t ix) {
			return data_ + ix;
		}
		T* begin() {return data_;}
		T* end() {return data_ + size_;}

		uint32_t argmax(){
			return std::distance(data_, std::max_element(data_, data_ + size_));
		}

		T max(){return *std::max_element(data_, data_ + size_);}
		void fill(T value){std::fill(data_, data_ + size_, value);}
	
};

template <typename T>
class Array2D {
	public:
		T *data_ = nullptr;
		uint32_t nrow_, ncol_;
		uint64_t size_;
		uint32_t shape_[2];

		// Array2D() {}
		// init from existing allocated memory - deprecated
		Array2D(T *data, uint32_t nrow, uint32_t ncol)
		: data_(data), nrow_(nrow), ncol_(ncol), size_(nrow * ncol) 
		{
			shape_[0] = nrow_;
			shape_[1] = ncol_;
		}

		// init array and allocate memory
		Array2D(uint32_t nrow, uint32_t ncol)
		: nrow_(nrow), ncol_(ncol), size_(nrow * ncol){
			shape_[0] = nrow_;
			shape_[1] = ncol_;
			data_ = new T[size_];
		}

		// init array from shape and allocate memory
		Array2D(uint32_t shape[2])
		: nrow_(shape[0]), ncol_(shape[1]), size_(nrow_ * ncol_){
			shape_[0] = nrow_;
			shape_[1] = ncol_;
			data_ = new T[size_];
		}

		~Array2D(){delete data_;}
		// force move constructor
		Array2D(Array2D&&) = default;

		// Get value at flattened index ix
		T& operator[] (uint32_t ix){return data_[ix];}

		// Get value at simulated (row, col)
		T& operator() (uint32_t ix_row, uint32_t ix_col){
			return data_[ix_row * ncol_ + ix_col];
		}
		// Return pointer to i'th row
		T* row(uint32_t irow) {
			return data_ + irow * ncol_;
		}

		// Return pointer to zeroth element
		T* begin() {return data_;}
		T* end() {return data_ + size_;}

		Vector<T> sum_rows() {

			Vector<T> out = Vector<T>(ncol_);
			out.fill(0);
			float *out_ptr = nullptr;

			for (uint32_t i = 0; i < nrow_; ++i)
			{
				out_ptr = data_ + i * ncol_;		
				for (uint32_t j = 0; j < ncol_; ++j)
				{
					out[j] += out_ptr[j];
				}
			}
			return out;	

		}

		void fill(T value){std::fill(data_, data_ + size_, value);}	
};


struct Grid {
	// e.g (xmin, xmax, ymin, ymax, zmin, zmax)
	std::array<float, 6> lims; 
	// (dx, dy, dz)
	float spacing;


	float xmin, ymin, zmin;
	float x, y, z;
	// float zmax = lims[5];
	uint32_t nx, ny, nz;
	uint32_t ix, iy, iz;
	uint64_t npts;
	uint64_t size;
	// uint32_t size = 0;

	Grid() {}
	Grid(std::array<float, 6> lims, float spacing):
	lims(lims), spacing(spacing), xmin(lims[0]), ymin(lims[2]), zmin(lims[4]){
		nx = (lims[1] - lims[0]) / spacing;
		ny = (lims[3] - lims[2]) / spacing;
		nz = (lims[5] - lims[4]) / spacing;
		npts = nx * ny * nz;
		size = npts * 3;
		printf("Grid (%u x %u x %u) = %lu\n", nx, ny, nz, npts);
	}
	~Grid(){}

	Array2D<float> build_points(){

		auto points = Array2D<float>(npts, 3);
		uint32_t ix = 0;
		for (uint32_t i = 0; i < nz; ++i) {
			for (uint32_t j = 0; j < ny; ++j) {
				for (uint32_t k = 0; k < nx; ++k) {
					points.row(ix)[0] = xmin + k * spacing;
					points.row(ix)[1] = ymin + j * spacing;
					points.row(ix)[2] = zmin + i * spacing;
					ix++;
				}			
			}
		}
		return points;
	}

	uint64_t get_index(float *point){
		
		x = point[0];
		y = point[1];
		z = point[2];
		ix = (x - xmin) / spacing;
		iy = (y - ymin) / spacing;
		iz = (z - zmin) / spacing;
		return (iz * nx * ny) + (iy * nx) + ix;
	}

	void get_point(uint64_t index, float *buf){
				
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

	std::array<float, 3> get_point(uint64_t index){
				
		ix = index % nx;
		iy = ((index - ix) / nx) % ny;
		iz = index / (nx * ny);

		x = ix * spacing + xmin;
		y = iy * spacing + ymin;
		z = iz * spacing + zmin;
		std::array<float, 3> buf = {x, y, z};
		return buf;
		// buf[0] = x;
		// buf[1] = y;
		// buf[2] = z;
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

	Clock(){}
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

