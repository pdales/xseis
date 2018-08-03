/*
* @Author: Philippe Dales
* @Date:   2018-07-26 14:26:23
* @Last Modified by:   Philippe Dales
* @Last Modified time: 2018-07-26 14:26:23
*/
/*
Ray-tracer helper functions for estuary package
*/
#ifndef RAYTRACE_H
#define RAYTRACE_H

#include <assert.h>
// #include "xseis/process.h"
#include "xseis/structures.h"
#include "xseis/beamform.h"
// #include "xseis/npy.h"
#include "narray.hpp"
#include "nvect.hpp"
#include "solver.hpp"

namespace raytrace {



// Compute traveltimes from stalocs to gridlocs given 1d viscosity (slowness) model
// Uses correct depths but places gridloc cartesian(xy) from staloc
// Make sure visc/tag grids are big enough to account for longest xy cartesian dist
// (tag is modified in FMM, careful)

Array2D<uint16_t> TTableFromVisc1D(Array2D<float>& stalocs, Array2D<float>& gridlocs,
									Array2D<double>& viscosity, Array2D<int>& tag,
									const float visc_spacing, float sr)
{
	// consts used to make grids
	const size_t ndim = 2;
	const size_t npad = 2;	
	// const float zshift = 500;
	const float xseed = 100;

	const auto shape = agsis::vect<size_t, ndim>(viscosity.ncol_, viscosity.nrow_);
	const size_t size = agsis::prod(shape);

	// array descriptor of viscosity
	auto viscosity_ad = ArrayDescriptor<double, ndim>(shape, viscosity.data_);
	
	// array descriptor of tag
	auto tag_ad = ArrayDescriptor<MarchingTag, ndim>(shape, new MarchingTag[size]);

	// array descriptor of traveltime grid, fill with INFS
	auto tgrid = Array2D<double>(shape[0], shape[1]);
	auto tgrid_ad = OrthogonalGrid<double, ndim>(shape, tgrid.data_, visc_spacing);

	size_t nsta = stalocs.nrow_;
	auto ttable = Array2D<uint16_t>(stalocs.nrow_, gridlocs.nrow_);
	auto *ttablerow = ttable.row(0);

	float *sl = nullptr;
	float *gl = nullptr;
	float dxy;
	float xdest, zdest;
	float tt;
	float zseed;

	auto seed = agsis::vect<double, ndim>();
	
	// #pragma omp parallel for private(ttablerow, sl, gl, dxy, xdest, zdest, tt, zseed)
	for(size_t i = 0; i < nsta; ++i) {
		printf("%lu\n", i);
		ttablerow = ttable.row(i);
		sl = stalocs.row(i);

		zseed = sl[2];
		seed[0] = static_cast<int>(zseed / visc_spacing + npad + 0.5);
		seed[1] = static_cast<int>(xseed / visc_spacing + npad + 0.5);

		for(size_t j = 0; j < tag.size_; ++j) {
			tag_ad[j] = static_cast<MarchingTag>(tag[j]);
		}

		for(size_t j = 0; j < tgrid_ad.size(); ++j) {
			tgrid_ad[j] = INFINITY;
		}

		FMM_SecondOrder(seed, tag_ad, viscosity_ad, tgrid_ad);		

		for(size_t j = 0; j < gridlocs.nrow_; ++j) {
			gl = gridlocs.row(j);
			dxy = process::DistCartesian2D(gl, sl);
			xdest = static_cast<size_t>((xseed + dxy) / visc_spacing + npad + 0.5);
			zdest = static_cast<size_t>(gl[2] / visc_spacing + npad + 0.5);

			assert(xdest < tgrid.ncol_);
			assert(zdest < tgrid.nrow_);
			tt = tgrid(zdest, xdest);
			ttablerow[j] = static_cast<uint16_t>(tt * sr + 0.5);
		}
	}

	return ttable;
}


// Ray trace for 1D velocity model (tag is modified in FMM, careful)
// Make sure visc/tag grids are big enough to account for longest xy cartesian dist
Array2D<uint16_t> BuildTravelTime1D(Array2D<float>& stalocs, Array2D<float>& gridlocs,
									Array2D<double>& viscosity, Array2D<int>& tag,
									float sr)
{
	// consts I use to make grids
	const size_t ndim = 2;
	const size_t npad = 2;
	const size_t visc_spacing = 5;
	const size_t zpad = 100;
	const float xseed = 100;


	// const auto shape = agsis::vect<size_t, ndim>(ds.nrow_, ds.ncol_);
	const auto shape = agsis::vect<size_t, ndim>(viscosity.ncol_, viscosity.nrow_);
	const size_t size = agsis::prod(shape);

	// double *data = new double[size];
	// ds.load_full_buffer(data);	
	auto viscosity_ad = ArrayDescriptor<double, ndim>(shape, viscosity.data_);
	// int *dtag = new int[size];
	// hf["tag"].load_full_buffer(dtag);

	// auto tag_ad = ArrayDescriptor<MarchingTag, ndim>(shape, (MarchingTag *)tag.data_);
	auto tag_ad = ArrayDescriptor<MarchingTag, ndim>(shape, new MarchingTag[size]);

	auto tgrid = Array2D<double>(shape[0], shape[1]);
	auto tgrid_ad = OrthogonalGrid<double, ndim>(shape, tgrid.data_, visc_spacing);

	for(size_t j = 0; j < tgrid_ad.size(); ++j) {
		tgrid_ad[j] = INFINITY;
	}

	for(size_t j = 0; j < tag.size_; ++j) {
		tag_ad[j] = static_cast<MarchingTag>(tag[j]);
	}

	auto seed = agsis::vect<double, ndim>();
	float zseed = zpad;
	seed[0] = static_cast<int>(zseed / visc_spacing + npad + 0.5);
	seed[1] = static_cast<int>(xseed / visc_spacing + npad + 0.5);
	FMM_SecondOrder(seed, tag_ad, viscosity_ad, tgrid_ad);		


	size_t nsta = stalocs.nrow_;
	// auto ttable = Array2D<float>(nsta, gridlocs.nrow_);
	auto ttable = Array2D<uint16_t>(stalocs.nrow_, gridlocs.nrow_);
	auto *ttablerow = ttable.row(0);

	float *sl = nullptr;
	float *gl = nullptr;
	float dxy;
	float xdest, zdest;
	float tt;
	// auto ttable = Vector<float>(gridlocs.nrow_);
	// size_t nsta = 100;

	
	#pragma omp parallel for private(ttablerow, sl, gl, dxy, xdest, zdest, tt)
	for(size_t i = 0; i < nsta; ++i) {
		// printf("%lu\n", i);
		ttablerow = ttable.row(i);
		sl = stalocs.row(i);

		for(size_t j = 0; j < gridlocs.nrow_; ++j) {
			gl = gridlocs.row(j);
			dxy = process::DistCartesian2D(gl, sl) + std::abs(sl[2]);
			xdest = (xseed + dxy) / visc_spacing + npad;
			zdest = (gl[2] + zpad) / visc_spacing + npad;
			assert(xdest < tgrid.ncol_);
			assert(zdest < tgrid.nrow_);
			tt = tgrid(static_cast<size_t>(zdest + 0.5), static_cast<size_t>(xdest + 0.5));
			ttablerow[j] = static_cast<uint16_t>(tt * sr + 0.5);
		}
	}

	return ttable;
}


// rows will be locs1 and cols locs2
Array2D<uint16_t> BuildRow1D(Array2D<float>& locs1, Array2D<float>& locs2,
									Array2D<double>& viscosity, Array2D<int>& tag,
									float sr)
{
	// consts I use to make grids
	const size_t ndim = 2;
	const size_t npad = 2;
	const size_t visc_spacing = 5;
	const size_t zpad = 100;
	const float xseed = 100;


	// const auto shape = agsis::vect<size_t, ndim>(ds.nrow_, ds.ncol_);
	const auto shape = agsis::vect<size_t, ndim>(viscosity.ncol_, viscosity.nrow_);
	const size_t size = agsis::prod(shape);

	// double *data = new double[size];
	// ds.load_full_buffer(data);	
	auto viscosity_ad = ArrayDescriptor<double, ndim>(shape, viscosity.data_);
	// int *dtag = new int[size];
	// hf["tag"].load_full_buffer(dtag);

	// auto tag_ad = ArrayDescriptor<MarchingTag, ndim>(shape, (MarchingTag *)tag.data_);
	auto tag_ad = ArrayDescriptor<MarchingTag, ndim>(shape, new MarchingTag[size]);

	auto tgrid = Array2D<double>(shape[0], shape[1]);
	auto tgrid_ad = OrthogonalGrid<double, ndim>(shape, tgrid.data_, visc_spacing);

	for(size_t j = 0; j < tgrid_ad.size(); ++j) {
		tgrid_ad[j] = INFINITY;
	}

	for(size_t j = 0; j < tag.size_; ++j) {
		tag_ad[j] = static_cast<MarchingTag>(tag[j]);
	}

	auto seed = agsis::vect<double, ndim>();
	float zseed = zpad;
	seed[0] = static_cast<int>(zseed / visc_spacing + npad + 0.5);
	seed[1] = static_cast<int>(xseed / visc_spacing + npad + 0.5);
	FMM_SecondOrder(seed, tag_ad, viscosity_ad, tgrid_ad);		


	auto ttable = Array2D<uint16_t>(locs1.nrow_, locs2.nrow_);
	
	#pragma omp parallel for
	for(size_t i = 0; i < locs1.nrow_; ++i) {
		// printf("%lu\n", i);
		uint16_t *ttablerow = ttable.row(i);
		float *l1p = locs1.row(i);

		for(size_t j = 0; j < locs2.nrow_; ++j) {
			float *l2p = locs2.row(j);
			float dxy = process::DistCartesian2D(l2p, l1p);
			float dz = std::abs(l1p[2] - l2p[2]);
			float xdest = (xseed + dxy) / visc_spacing + npad;
			float zdest = (zpad + dz) / visc_spacing + npad;
			assert(xdest < tgrid.ncol_);
			assert(zdest < tgrid.nrow_);
			float tt = tgrid(static_cast<size_t>(zdest + 0.5), static_cast<size_t>(xdest + 0.5));
			ttablerow[j] = static_cast<uint16_t>(tt * sr + 0.5);
		}
	}

	return ttable;
}


}

#endif
