#ifndef BEAMFORM_H
#define BEAMFORM_H

#include "xseis/process.h"
#include "xseis/structures.h"


namespace beamform {

inline float DistCartesian(float* a, float* b)
{	
	float dx = a[0] - b[0];
	float dy = a[1] - b[1];
	float dz = a[2] - b[2];
	return std::sqrt(dx * dx + dy * dy + dz * dz);
}

unsigned mod_floor(int a, int n) {
	return ((a % n) + n) % n;
}


Array2D<float> InterLoc(Array2D<float>& data_cc, Array2D<uint16_t>& ckeys, Array2D<uint16_t>& ttable, uint16_t nthreads)
{

	const uint32_t cclen = data_cc.ncol_;
	const uint32_t ncc = data_cc.nrow_;
	const uint64_t ngrid = ttable.ncol_;

	uint16_t ixc;
	uint16_t *tts_sta1, *tts_sta2;
	float *cc_ptr = nullptr;

	auto output = Array2D<float>(nthreads, ngrid);
	uint32_t niter = 0;
	#pragma omp parallel private(ixc, tts_sta1, tts_sta2, cc_ptr) num_threads(nthreads)
	{
		float *out_ptr = output.row(omp_get_thread_num());
		std::fill(out_ptr, out_ptr + ngrid, 0);
		// play around with loop scheduling here (later iterations should be slightly slower due to faster changin ckeys)
		#pragma omp for
		for (uint32_t i = 0; i < ncc; ++i)
		{
			// if (i % 10000 == 0) {
			// 	printf("Prog: %.2f \r", ((float) i / ncc * 100));
			// 	std::cout.flush();
			// }

			tts_sta1 = ttable.row(ckeys(i, 0));	
			tts_sta2 = ttable.row(ckeys(i, 1));
			cc_ptr = data_cc.row(i);

			#pragma omp simd \
			aligned(tts_sta1, tts_sta2, out_ptr, cc_ptr: MEM_ALIGNMENT)
			for (uint64_t j = 0; j < ngrid; ++j)
			{

				if (tts_sta2[j] >= tts_sta1[j])
				{
					out_ptr[j] += cc_ptr[tts_sta2[j] - tts_sta1[j]];					
				}
				else
				{
					out_ptr[j] += cc_ptr[cclen - tts_sta1[j] + tts_sta2[j]];
				}
			}
		}
	}	
	return output;
}



Array2D<uint16_t> BuildTravelTimeTable(Array2D<float>& stalocs, Array2D<float>& gridlocs, float vel, float sr)
{
	auto ttable = Array2D<uint16_t>(stalocs.nrow_, gridlocs.nrow_);

	float vsr = sr / vel;
	float dist;
	float *sloc = nullptr;
	uint16_t *tt_row = nullptr;

	#pragma omp parallel for private(sloc, tt_row, dist)
	for (uint32_t i = 0; i < ttable.nrow_; ++i)
	{
		sloc = stalocs.row(i);
		tt_row = ttable.row(i);

		for (uint32_t j = 0; j < ttable.ncol_; ++j) 
		{
			dist = DistCartesian(sloc, gridlocs.row(j));			
			tt_row[j] = static_cast<uint16_t>(dist * vsr + 0.5);
		}
	}
	return ttable;
}


Array2D<uint16_t> BuildTravelTimeTable(Array2D<float>& stalocs, Array2D<float>& gridlocs, Vector<float>& vel_effective, float sr)
{
	uint32_t ngrid = gridlocs.nrow_;
	uint32_t nsta = stalocs.nrow_;

	auto ttable = Array2D<uint16_t>(nsta, ngrid);

	// compute velocity sampling rate
	auto vsr = Vector<float>(vel_effective.size_);
	for (uint32_t i = 0; i < vsr.size_; ++i)
	{
		vsr[i] = sr / vel_effective[i];
	}

	auto vsr_grid = Vector<float>(ngrid);

	for (uint32_t i = 0; i < ngrid; ++i){
		vsr_grid[i] = vsr[static_cast<uint16_t>(gridlocs[i * 3 + 2])];
	}

	float dist;
	float *sloc = nullptr;
	uint16_t *tt_row = nullptr;

	#pragma omp parallel for private(sloc, tt_row, dist)
	for (uint32_t i = 0; i < nsta; ++i)
	{
		sloc = stalocs.row(i);
		tt_row = ttable.row(i);

		for (uint32_t j = 0; j < ngrid; ++j) 
		{	
			dist = DistCartesian(sloc, gridlocs.row(j));
			tt_row[j] = static_cast<uint16_t>(dist * vsr_grid[j] + 0.5);
		}
	}

	return ttable;
}


// Vector<uint16_t> GetTTSourceToStations(float* src, Array2D<float>& stalocs, Vector<float>& vel_effective, float sr)
// {

// 	uint32_t nsta = stalocs.nrow_;
// 	auto tts = Vector<uint16_t>(nsta);

// 	// compute velocity sampling rate
// 	auto vsr = Vector<float>(vel_effective.size_);
// 	for (uint32_t i = 0; i < vsr.size_; ++i)
// 	{
// 		vsr[i] = sr / vel_effective[i];
// 	}

// 	auto vsr_grid = Vector<float>(ngrid);

// 	for (uint32_t i = 0; i < ngrid; ++i){
// 		vsr_grid[i] = vsr[static_cast<uint16_t>(gridlocs[i * 3 + 2])];
// 	}

// 	float dist;

// 	for (uint32_t j = 0; j < ngrid; ++j) 
// 	{	
// 		dist = DistCartesian(staloc, gridlocs.row(j));
// 		tts[j] = static_cast<uint16_t>(dist * vsr_grid[j] + 0.5);
// 	}

// 	return tts;
// }


Vector<uint16_t> GetTTOneToMany(float* loc_src, Array2D<float>& locs, float vel, float sr)
{
	uint32_t nlocs = locs.nrow_;
	auto tts = Vector<uint16_t>(nlocs);
	float vsr = sr / vel;
	float dist;

	for (uint32_t j = 0; j < nlocs; ++j) 
	{	
		dist = DistCartesian(loc_src, locs.row(j));
		tts[j] = static_cast<uint16_t>(dist * vsr + 0.5);
	}

	return tts;
}


void tt_homo_ix(Array2D<float> &sta_locs, float *src_loc, float vsr, Vector<uint32_t> &tts)
{	
	float dist;
	for (uint32_t j = 0; j < tts.size_; ++j) {
		dist = beamform::DistCartesian(src_loc, sta_locs.row(j));
		tts[j] = static_cast<uint32_t>(dist * vsr + 0.5);
	}
}


void tt_homo(Vector<float> &tts, Array2D<float> &sta_locs, float *src_loc, float velocity)
{
	float dist;

	for (int32_t j = 0; j < sta_locs.nrow_; ++j) {
		// dist = DistCartesian(src_loc, &sta_locs[j]);
		dist = DistCartesian(src_loc, sta_locs.row(j));
		tts[j] = dist / velocity;
	}
}

void tt_diff(float *tts, float *tts_cc, int *ckeys, int ncc)
{
	int key1, key2;

	for (int i = 0; i < ncc; ++i) {
		key1 = ckeys[i * 2];
		key2 = ckeys[i * 2 + 1];
		// printf("%d_%d\n", key1, key2);
		tts_cc[i] = tts[key2] - tts[key1];
	}
}


float slant_stack(Array2D<float> &data, Vector<float> &tts, Vector<int> &ix_keep, float src_time, float sr)
{
	float sum = 0;
	int nvalid = 0;
	int32_t col_ix;
	
	for (int32_t i = 0; i < ix_keep.size_; ++i) {
		col_ix = (src_time + tts[i]) * sr;

		if (0 <= col_ix && col_ix < data.ncol_) {
			sum += data(i, col_ix);
			// printf("i: %d col_ix: %d, val: %.2f\n", i, col_ix, sig_ptr[col_ix]);
			nvalid++;
		}
	}
	if (nvalid == 0){
		return 0;
	}
	else {
		return sum / nvalid;
	}
}


float slant_stack_no_check(Array2D<float> &data, Vector<float> &tts, Vector<int> &ix_keep, float src_time, float sr){
	// No bounds checking, care for segfaults
	float sum = 0;
	// float t_exp;	
	// int32_t col_ix;
	int col_ix;

	for (int32_t i = 0; i < ix_keep.size_; ++i) {
		// col_ix = (int32_t) (sr * (src_time + tts[i]));
		// col_ix = (int32_t) (sr * (src_time + tts[i]));
		// col_ix = static_cast<int32_t>((src_time + tts[i]) * sr);
		col_ix = static_cast<int>((src_time + tts[i]) * sr);
		// col_ix = (int) ((src_time + tts[i]) * sr);
		sum += data(i, col_ix);
		// sum += data(i, col_ix);
	}
	return sum / ix_keep.size_;
}


void beampower_homo(Array2D<float> &points, Vector<float> &out, Array2D<float> &data, Vector<float> &tts, Array2D<float> &sta_locs, Vector<int> &ix_keep, float velocity, float src_time, float sr) {

	// Vector<float> &tts2 = Vector<float>(&&tts[0], tts.size_);

	for (int32_t i = 0; i < points.nrow_; ++i) {
		tt_homo(tts, sta_locs, points.row(i), velocity);
		// tts2.set_data(&tts[0]);
		// out[i] = slant_stack(data, tts, ix_keep, src_time, sr);
		out[i] = slant_stack_no_check(data, tts, ix_keep, src_time, sr);
	}
}


void search_grid(Array2D<float>& data, Array2D<float>& locs,
			Grid& grid, uint32_t gix_start, uint32_t gix_end,
			uint32_t nt_search, float vsr,
			float* win_val, uint32_t* win_loc)
{
	uint32_t nchan = data.nrow_;
	auto tt_ixs = Vector<uint32_t>(nchan);	
	auto output = Vector<float>(nt_search);
	float src_loc[3];
	float *dptr = nullptr;
	float dist;

	printf("Searching grid points: %d to %d\n", gix_start, gix_end);

	for (uint32_t ipt = gix_start; ipt < gix_end; ++ipt)
	{	
		// if (ipt % 1000 == 0) {printf("Point: %d / %d\n", ipt, (int) gix_end);}
		// grid.get_point(ipt, src_loc);

		if (ipt % 1000 == 0) {
			printf("Progress: %.2f \n", ((float)(ipt - gix_start) / (gix_end - gix_start) * 100));
		}

		grid.get_point(ipt, src_loc);
		beamform::tt_homo_ix(locs, src_loc, vsr, tt_ixs);

		output.fill(0);

		// For each channel add time comb values to output
		for (uint32_t i = 0; i < nchan; ++i) 
		{
			dptr = data.row(i) + tt_ixs[i];
		
			for (uint32_t j = 0; j < nt_search; j++) 
			{
				output[j] += dptr[j];
			}
		}

		for (uint32_t j = 0; j < nt_search; ++j) 
		{
			if (std::abs(output[j]) > std::abs(win_val[j])) {
				win_val[j] = output[j];
				win_loc[j] = ipt;
			}
		}
	
	}

}

void search_grid_parallel(std::vector<uint32_t>& parts, Array2D<float>& data, Array2D<float>& locs, Grid& grid, uint32_t nt_search, float vsr,	Array2D<float>& win_val, Array2D<uint32_t>& win_loc)
{
	std::vector<std::thread> pool;

	for (uint32_t i = 0; i < parts.size() - 1; i++){

		pool.push_back(std::thread([=, &data, &locs, &grid, &win_val, &win_loc] {search_grid(data, locs, grid, parts[i], parts[i + 1], nt_search, vsr, win_val.row(i), win_loc.row(i));}));
	}

	for(auto& thread : pool) thread.join();
}


uint64_t factorial(uint64_t n)
{
	uint64_t ret = 1;
	for(uint64_t i = 1; i <= n; ++i)
		ret *= i;
	return ret;
}

uint64_t NChoose2(uint64_t n)
{
	return (n * (n-1)) / 2;
}



Array2D<uint16_t> unique_pairs(Vector<uint16_t>& keys)
{
	uint64_t npair = 0;

	// crude way to calc nkeys (wil use dist filters later)
	for (unsigned i = 0; i < keys.size_; ++i)
	{
		for (unsigned j = i + 1; j < keys.size_; ++j)
		{
			npair += 1;			
		}
	}

	auto ckeys = Array2D<uint16_t>(npair, 2);
	uint64_t row_ix = 0;

	for (unsigned i = 0; i < keys.size_; ++i)
	{
		for (unsigned j = i + 1; j < keys.size_; ++j)
		{
			ckeys(row_ix, 0) = keys[i];
			ckeys(row_ix, 1) = keys[j];
			row_ix += 1;
		}
	}
	return ckeys;
}

Array2D<uint16_t> BuildPairsDistFilt(Vector<uint16_t>& keys, Array2D<float>& stalocs, float min_dist)
{
	// uint64_t npair = 0;
	uint64_t npair_max = NChoose2(keys.size_);
	printf("max pairs %lu\n", npair_max);
	auto ckeys = Array2D<uint16_t>(npair_max, 2);
	uint64_t row_ix = 0;
	float dist;
	float* loc1 = nullptr;
	float* loc2 = nullptr;
	
	for (unsigned i = 0; i < keys.size_; ++i)
	{
		loc1 = stalocs.row(keys[i]);

		for (unsigned j = i + 1; j < keys.size_; ++j)
		{
			loc2 = stalocs.row(keys[j]);
			dist = DistCartesian(loc1, loc2);
			// printf("[%u,%u] %.2f \n",i, j, dist);
			if (dist > min_dist)
			{
				// printf("%u\n", row_ix);
				ckeys(row_ix, 0) = keys[i];
				ckeys(row_ix, 1) = keys[j];
				row_ix += 1;
			}			
		}
	}

	auto ckeys2 = Array2D<uint16_t>(row_ix, 2);
	for(unsigned i = 0; i < ckeys2.size_; ++i) {
		ckeys2[i] = ckeys[i];
	}

	return ckeys2;
}



}

#endif
