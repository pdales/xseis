#ifndef H5WRAP_H
#define H5WRAP_H
#include <iostream>
#include <string>

#include "H5Cpp.h"
#include "xseis/structures.h"


// H5::DataType dt_char(H5::PredType::NATIVE_CHAR); 
// H5::DataType dt_int32(H5::PredType::NATIVE_INT32); 
// H5::DataType dt_float(H5::PredType::NATIVE_FLOAT);
// H5::DataType dt3(H5::PredType::NATIVE_DOUBLE);     

namespace h5wrap {

struct Dataset {
	H5::DataSet dset_;
	H5::DataSpace filespace_;
	hsize_t nrow_, ncol_, size_;
	size_t rank_;
	H5::DataType dtype_;
	hsize_t shape_[2] = {1, 1};    // dataset dimensions

	// const H5std_string dset_name; 
	
	// Dataset() {}
	Dataset(H5::DataSet dset): dset_(dset){
		filespace_ = dset_.getSpace();
		rank_ = filespace_.getSimpleExtentNdims();
		// hsize_t shape_[rank];    // dataset dimensions
		rank_ = filespace_.getSimpleExtentDims(shape_);
		nrow_ = shape_[0];
		ncol_ = shape_[1];
		size_ = shape_[0] * shape_[1];		
		dtype_ = dset_.getDataType();		
	}

	// template <typename T>	
	// void load_full_buffer(T *buffer) {
	// 	hsize_t dim_flat[1] = {size_};
	// 	H5::DataSpace mspace(1, dim_flat);		
	// 	dset.read(buffer, dtype, mspace, filespace);
	// 	}

	template <typename T>	
	void load_full(Array2D<T> &arr) {
		// hsize_t dim_flat[1] = {size_};
		H5::DataSpace mspace(rank_, shape_);
		dset_.read(arr.data_, dtype_, mspace, filespace_);

		}


	Array2D<float> load_float_array() {
		auto arr = Array2D<float>({(size_t) nrow_, (size_t) ncol_});
		H5::DataSpace mspace(rank_, shape_);
		dset_.read(arr.data_, dtype_, mspace, filespace_);
		return arr;		
		}

	Vector<float> load_float_vector() {
		auto vec = Vector<float>(nrow_);
		H5::DataSpace mspace(1, shape_);
		dset_.read(vec.data_, dtype_, mspace, filespace_);
		return vec;		
		}

	template <typename T>	
	void LoadChunk(Array2D<T> &arr, hsize_t offset[2]) {

		// Define slab size
		hsize_t count[2] = {arr.nrow_, arr.ncol_};
		filespace_.selectHyperslab(H5S_SELECT_SET, count, offset);

		H5::DataSpace mspace(2, count);
		dset_.read(arr.data_, dtype_, mspace, filespace_);
		}

	template <typename T>	
	void LoadRows(Array2D<T>& arr, Vector<uint16_t>& keys, hsize_t col_offset) {

		if(arr.nrow_ != keys.size_) {
			printf("WARNING nrows does not match buffer: %lu != %lu\n", arr.nrow_, keys.size_);
		}
		// Define slab size
		hsize_t count[2] = {1, arr.ncol_};
		hsize_t offset[2] = {0, col_offset};
		H5::DataSpace mspace(2, count);

		for(size_t i = 0; i < keys.size_; ++i) {
			offset[0] = keys[i];
			filespace_.selectHyperslab(H5S_SELECT_SET, count, offset);
			dset_.read(arr.row(i), dtype_, mspace, filespace_);			
		}

		
		}	
};

struct File {
	// size_t nrow_, ncol_;
	// const H5std_string file_path; 
	H5::H5File hfile;
	
	// File() {}
	// File(const H5std_string file_path): file_path(file_path){
	// 	hfile = H5::H5File(file_path, H5F_ACC_RDONLY);
	// }
	File(const H5std_string file_path){
		hfile = H5::H5File(file_path, H5F_ACC_RDONLY);
	}

	Dataset operator[] (const H5std_string dset_name){		
		return Dataset(hfile.openDataSet(dset_name));
	}

	template <typename T>
	void read_attr(const H5std_string attr_name, T val){
	// float sr = 0.0;
	H5::Attribute attr = hfile.openAttribute(attr_name);
	attr.read(attr.getDataType(), val);

}

};

}

#endif



