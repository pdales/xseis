/*
Classes to read H5 files.

Cant figure out a way to convert H5 datatypes into equivalent c-types,
hence all the load_type_array like functions. 
*/

#ifndef H5WRAP_H
#define H5WRAP_H
#include <iostream>
#include <string>

#include "H5Cpp.h"
#include "xseis/structures.h"
#include <assert.h>


// H5::DataType dt_char(H5::PredType::NATIVE_CHAR); 
// H5::DataType dt_int32(H5::PredType::NATIVE_INT32); 
// H5::DataType dt_float(H5::PredType::NATIVE_FLOAT);
// H5::DataType dt3(H5::PredType::NATIVE_DOUBLE);     

namespace h5wrap {

// wrapper for h5 dataset
struct Dataset {
	H5::DataSet dset_;
	H5::DataSpace filespace_;
	hsize_t nrow_, ncol_, size_;
	size_t rank_;
	H5::DataType dtype_;
	hsize_t shape_[2] = {1, 1};    // dataset dimensions
	
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

	template <typename T>	
	void LoadChunk(Array2D<T> &arr, hsize_t offset[2]) {
		// Loads hyperslab with arr dimensions at specified offset		
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


	// template <typename T>	
	// void load_full_buffer(T *buffer) {
	// 	H5::DataSpace mspace(rank_, shape_);
	// 	dset_.read(buffer, dtype_, mspace, filespace_);
	// }		

	template <typename T>
	Array2D<T> LoadArray() {
		auto arr = Array2D<T>({(size_t) nrow_, (size_t) ncol_});
		H5::DataSpace mspace(rank_, shape_);
		dset_.read(arr.data_, dtype_, mspace, filespace_);
		return arr;		
	}

	template <typename T>	
	void LoadArray(Array2D<T> &arr) {
		assert(arr.size_ == size_);
		H5::DataSpace mspace(rank_, shape_);
		dset_.read(arr.data_, dtype_, mspace, filespace_);
	}
	
	template <typename T>
	Vector<T> LoadVector() {
		auto vec = Vector<T>((size_t) nrow_);
		H5::DataSpace mspace(1, shape_);
		dset_.read(vec.data_, dtype_, mspace, filespace_);
		return vec;
	}

};


// wrapper for h5 file
struct File {
	H5::H5File hfile;

	File(const H5std_string file_path){
		hfile = H5::H5File(file_path, H5F_ACC_RDONLY);		
	}

	Dataset operator[] (const H5std_string dset_name){		
		return Dataset(hfile.openDataSet(dset_name));
	}

	// template <typename T>
	// void read_attr(const H5std_string attr_name, T val){		 
	// 	H5::Attribute attr = hfile.openAttribute(attr_name);
	// 	attr.read(attr.getDataType(), val);
	// }

	template <typename T>
	T load_attr(const H5std_string attr_name){
		T val;
		H5::Attribute attr = hfile.openAttribute(attr_name);
		attr.read(attr.getDataType(), &val);
		return val;
	}

};

}

#endif



