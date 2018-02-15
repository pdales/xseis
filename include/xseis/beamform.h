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

Array2D<float> InterLoc(Array2D<float>& data_cc, Array2D<uint16_t>& ckeys, Array2D<uint16_t>& ttable, uint16_t nthreads)
{

	uint32_t cclen = data_cc.ncol_;
	uint16_t ixc;
	uint16_t *tts_sta1, *tts_sta2;
	float *cc_ptr = nullptr;

	auto output = Array2D<float>(nthreads, ttable.ncol_);
	output.fill(0);

	#pragma omp parallel private(ixc, tts_sta1, tts_sta2, cc_ptr) num_threads(nthreads)
	{
		float *out_ptr = output.row(omp_get_thread_num());
		printf("thread %u\n", omp_get_thread_num());
		std::cout.flush();

		// play around with loop scheduling here (later iterations should be slightly slower due to faster changin ckeys)
		#pragma omp for
		for (uint32_t i = 0; i < ckeys.nrow_; ++i)
		{
			if (i % 10000 == 0) {			
				printf("Prog: %.2f \r", ((float) i / ckeys.nrow_ * 100));
				std::cout.flush();
			}
			tts_sta1 = ttable.row(ckeys(i, 0));	
			tts_sta2 = ttable.row(ckeys(i, 1));
			cc_ptr = data_cc.row(i);

			#pragma omp simd \
			aligned(tts_sta1, tts_sta2, out_ptr, cc_ptr: MEM_ALIGNMENT)
			for (uint32_t j = 0; j < ttable.ncol_; ++j)
				{
					ixc = (tts_sta2[j] - tts_sta1[j]) % cclen;
					out_ptr[j] += cc_ptr[ixc];
				}	
		}

	}	
	return output;
}



Array2D<uint16_t> BuildTravelTimeTable(Array2D<float>& stalocs, Array2D<float>& gridlocs, float sr, float vel)
{
	auto ttable = Array2D<uint16_t>(stalocs.nrow_, gridlocs.nrow_);

	float vsr = sr / vel;
	float dist;

	for (uint32_t i = 0; i < ttable.nrow_; ++i)
	{
		for (uint32_t j = 0; j < ttable.ncol_; ++j) 
		{
			dist = DistCartesian(stalocs.row(i), gridlocs.row(j));			
			ttable(i, j) = static_cast<uint16_t>(dist * vsr + 0.5);
		}
	}

	return ttable;
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

}

#endif
