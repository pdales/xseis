/*
Beamforming functions.
*/
#ifndef BEAMFORM_H
#define BEAMFORM_H

#include "xseis/process.h"
#include "xseis/structures.h"
#include <random>
#include <omp.h>


namespace beamform {


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

uint mod_floor(int a, int n) {
	return ((a % n) + n) % n;
}


Array2D<float> InterLoc(Array2D<float>& data_cc, Array2D<uint16_t>& ckeys, Array2D<uint16_t>& ttable, uint16_t nthreads)
{
	// Each thread given own output buffer to prevent cache invalidations

	const size_t cclen = data_cc.ncol_;
	const size_t ncc = data_cc.nrow_;
	const size_t ngrid = ttable.ncol_;

	uint16_t *tts_sta1, *tts_sta2;
	float *cc_ptr = nullptr;

	auto output = Array2D<float>(nthreads, ngrid);
	size_t niter = 0;
	#pragma omp parallel private(tts_sta1, tts_sta2, cc_ptr) num_threads(nthreads)
	{
		float *out_ptr = output.row(omp_get_thread_num());
		std::fill(out_ptr, out_ptr + ngrid, 0);
		// play around with omp loop scheduling here
		#pragma omp for
		for (size_t i = 0; i < ncc; ++i)
		{
			// if (i % 10000 == 0) {
			// 	printf("Prog: %.2f \r", ((float) i / ncc * 100));
			// 	std::cout.flush();
			// }

			tts_sta1 = ttable.row(ckeys(i, 0));	
			tts_sta2 = ttable.row(ckeys(i, 1));
			cc_ptr = data_cc.row(i);

			// Migrate single ccf on to grid based on tt difference
			#pragma omp simd \
			aligned(tts_sta1, tts_sta2, out_ptr, cc_ptr: MEM_ALIGNMENT)
			for (size_t j = 0; j < ngrid; ++j)
			{
				// Get appropriate ix of unrolled ccfs (same as mod_floor)
				// by wrapping negative traveltime differences
				// if-else much faster than more elegant mod function
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




Vector<float> InterLocBlocks(Array2D<float>& data_cc, Array2D<uint16_t>& ckeys, Array2D<uint16_t>& ttable, size_t blocksize=1024 * 5, float scale_pwr=100)
{
	// Divide grid into chunks to prevent cache invalidations during writing (see Ben Baker migrate)
	// This uses less memory but was a bit slower atleast in my typical grid/ccfs sizes
	// UPdate: When grid sizes >> nccfs and using more than 15 cores faster than InterLoc above

	const size_t cclen = data_cc.ncol_;
	const size_t ncc = data_cc.nrow_;
	const size_t ngrid = ttable.ncol_;
	size_t blocklen;

	uint16_t *tts_sta1, *tts_sta2;
	float *cc_ptr = nullptr;
	float *out_ptr = nullptr;

	auto output = Vector<float>(ngrid);
	output.fill(0);

	// printf("blocksize %lu, ngrid %lu \n", blocksize, ngrid);

	#pragma omp parallel for private(tts_sta1, tts_sta2, cc_ptr, out_ptr, blocklen)
	for(size_t iblock = 0; iblock < ngrid; iblock += blocksize) {

		blocklen = std::min(ngrid - iblock, blocksize);

		out_ptr = output.data_ + iblock;
		// out_ptr = output.data_ + iblock * blocklen;
		// std::fill(out_ptr, out_ptr + blocklen, 0);
		
		for (size_t i = 0; i < ncc; ++i)
		{				
			tts_sta1 = ttable.row(ckeys(i, 0)) + iblock;	
			tts_sta2 = ttable.row(ckeys(i, 1)) + iblock;
			cc_ptr = data_cc.row(i);

			// Migrate single ccf on to grid based on tt difference
			#pragma omp simd \
			aligned(tts_sta1, tts_sta2, out_ptr, cc_ptr: MEM_ALIGNMENT)
			for (size_t j = 0; j < blocklen; ++j)
			{
				// Get appropriate ix of unrolled ccfs (same as mod_floor)
				// by wrapping negative traveltime differences
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

	float norm = scale_pwr / static_cast<float>(ncc);	
	for(size_t i = 0; i < output.size_; ++i) {
		output[i] *= norm;
	}

	// printf("completed\n");	
	return output;
}



Vector<float> InterLocPatch(Array2D<float>& data_cc, Array2D<uint16_t>& ckeys, std::vector<uint>& ix_patch, Array2D<uint16_t>& ttable, size_t blocksize=1024 * 5, float scale_pwr=100)
{
	// Divide grid into chunks to prevent cache invalidations during writing (see Ben Baker migrate)
	// This uses less memory but was a bit slower atleast in my typical grid/ccfs sizes
	// UPdate: When grid sizes >> nccfs and using more than 15 cores faster than InterLoc above

	const size_t cclen = data_cc.ncol_;
	// const size_t ncc = data_cc.nrow_;
	const size_t ngrid = ttable.ncol_;
	size_t blocklen;

	uint16_t *tts_sta1, *tts_sta2;
	float *cc_ptr = nullptr;
	float *out_ptr = nullptr;

	auto output = Vector<float>(ngrid);
	output.fill(0);

	// printf("blocksize %lu, ngrid %lu \n", blocksize, ngrid);

	#pragma omp parallel for private(tts_sta1, tts_sta2, cc_ptr, out_ptr, blocklen) 
	for(size_t iblock = 0; iblock < ngrid; iblock += blocksize) {

		blocklen = std::min(ngrid - iblock, blocksize);

		out_ptr = output.data_ + iblock;
		// out_ptr = output.data_ + iblock * blocklen;
		// std::fill(out_ptr, out_ptr + blocklen, 0);
		// for (size_t i = 0; i < ncc; ++i)
		for(auto&& i : ix_patch) {

			tts_sta1 = ttable.row(ckeys(i, 0)) + iblock;	
			tts_sta2 = ttable.row(ckeys(i, 1)) + iblock;
			cc_ptr = data_cc.row(i);

			// Migrate single ccf on to grid based on tt difference
			#pragma omp simd \
			aligned(tts_sta1, tts_sta2, out_ptr, cc_ptr: MEM_ALIGNMENT)
			for (size_t j = 0; j < blocklen; ++j)
			{
				// Get appropriate ix of unrolled ccfs (same as mod_floor)
				// by wrapping negative traveltime differences
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

	float norm = scale_pwr / static_cast<float>(ix_patch.size());	
	for(size_t i = 0; i < output.size_; ++i) {
		output[i] *= norm;
	}
	// printf("completed\n");	
	return output;
}

// Delay and summing raw waveforms for all gridlocs for all possible starttimes
void NaiveSearch(Array2D<float>& data, Array2D<uint16_t>& ttable, size_t tmin, size_t tmax, Vector<float>& wpower, Vector<size_t>& wlocs)
{

	// std::cout << data.nrow_ << '\n';
	// std::cout << ttable.ncol_ << '\n';
	size_t nt = tmax - tmin;
	size_t nsig = data.nrow_;
	size_t ngrid = ttable.nrow_;

	// auto tt_ixs = Vector<size_t>(nsig);	
	auto tmp_stack = Vector<float>(nt);

	float* best_vals = wpower.data_;
	size_t* best_locs = wlocs.data_;
	// auto win_val = Vector<float>(nt);
	// auto win_loc = Vector<size_t>(nt);

	float *dptr = nullptr;
	uint16_t *tt_ixs = nullptr;
	// printf("Searching grid points: %lu to %lu\n", gix_start, gix_end);

	std::cout << "nt: " << nt << '\n';
	std::cout << "ngrid: " << ngrid << '\n';
	std::cout << "nsig: " << nsig << '\n';

	for (size_t ipt = 0; ipt < ngrid; ++ipt)
	{	
		tmp_stack.fill(0);
		tt_ixs = ttable.row(ipt);

		if (ipt % 1000 == 0) {
			printf("Progress: %.2f \n", ((float)(ipt) / (ngrid) * 100));
		}

		// For each channel add time comb values to output
		for (size_t i = 0; i < nsig; ++i) 
		{
			dptr = data.row(i) + tt_ixs[i] + tmin;
		
			for (size_t j = 0; j < nt; ++j) 
			{
				tmp_stack[j] += dptr[j];
			}
		}

		// #pragma omp simd aligned(tts_sta1, tts_sta2, out_ptr, cc_ptr: MEM_ALIGNMENT)
		for (size_t j = 0; j < nt; ++j) 
		{
			if (std::abs(tmp_stack[j]) > std::abs(best_vals[j])) {
				best_vals[j] = tmp_stack[j];
				best_locs[j] = ipt;
			}
		}	
	}
	// return std::make_tuple(&power, &locs);
}

// Uses constant velocity medium, introduce random errors
Array2D<uint16_t> BuildTTablePerturbVel(Array2D<float>& stalocs, Array2D<float>& gridlocs, float vel, float sr, float perturb)
{
	auto ttable = Array2D<uint16_t>(stalocs.nrow_, gridlocs.nrow_);

	// float vsr = sr / vel;
	float dist;
	float *sloc = nullptr;
	uint16_t *tt_row = nullptr;

	std::mt19937::result_type seed = time(0);
	auto rand = std::bind(std::uniform_real_distribution<float>(vel - perturb, vel + perturb), std::mt19937(seed));

	#pragma omp parallel for private(sloc, tt_row, dist)
	for (size_t i = 0; i < ttable.nrow_; ++i)
	{
		sloc = stalocs.row(i);
		tt_row = ttable.row(i);

		for (size_t j = 0; j < ttable.ncol_; ++j) 
		{
			dist = DistCartesian(sloc, gridlocs.row(j));			
			tt_row[j] = static_cast<uint16_t>(dist * sr / rand() + 0.5);
		}
	}
	return ttable;
}


// Uses constant velocity medium
Array2D<uint16_t> BuildTravelTimeTable(Array2D<float>& stalocs, Array2D<float>& gridlocs, float vel, float sr)
{
	auto ttable = Array2D<uint16_t>(stalocs.nrow_, gridlocs.nrow_);

	float vsr = sr / vel;
	float dist;
	float *sloc = nullptr;
	uint16_t *tt_row = nullptr;

	#pragma omp parallel for private(sloc, tt_row, dist)
	for (size_t i = 0; i < ttable.nrow_; ++i)
	{
		sloc = stalocs.row(i);
		tt_row = ttable.row(i);

		for (size_t j = 0; j < ttable.ncol_; ++j) 
		{
			dist = DistCartesian(sloc, gridlocs.row(j));			
			tt_row[j] = static_cast<uint16_t>(dist * vsr + 0.5);
		}
	}
	return ttable;
}

// Uses 1D effective velocity model (1 value per meter)
Array2D<uint16_t> BuildTravelTimeTable(Array2D<float>& stalocs, Array2D<float>& gridlocs, Vector<float>& vel_effective, float sr)
{
	size_t ngrid = gridlocs.nrow_;
	size_t nsta = stalocs.nrow_;

	auto ttable = Array2D<uint16_t>(nsta, ngrid);

	// compute velocity sampling rate
	auto vsr = Vector<float>(vel_effective.size_);
	for (size_t i = 0; i < vsr.size_; ++i)
	{
		vsr[i] = sr / vel_effective[i];
	}

	auto vsr_grid = Vector<float>(ngrid);

	float zval;
	for (size_t i = 0; i < ngrid; ++i){
		zval = gridlocs[i * 3 + 2];
		if(zval < 0) {
			zval = 0;
		}
		vsr_grid[i] = vsr[static_cast<uint16_t>(zval)];
	}

	float dist;
	float *sloc = nullptr;
	uint16_t *tt_row = nullptr;

	#pragma omp parallel for private(sloc, tt_row, dist)
	for (size_t i = 0; i < nsta; ++i)
	{
		sloc = stalocs.row(i);
		tt_row = ttable.row(i);

		for (size_t j = 0; j < ngrid; ++j) 
		{	
			dist = DistCartesian(sloc, gridlocs.row(j));
			tt_row[j] = static_cast<uint16_t>(dist * vsr_grid[j] + 0.5);
		}
	}

	return ttable;
}


Vector<uint16_t> GetTTOneToMany(float* loc_src, Array2D<float>& locs, float vel, float sr)
{
	size_t nlocs = locs.nrow_;
	auto tts = Vector<uint16_t>(nlocs);
	float vsr = sr / vel;
	float dist;

	for (size_t j = 0; j < nlocs; ++j) 
	{	
		dist = DistCartesian(loc_src, locs.row(j));
		tts[j] = static_cast<uint16_t>(dist * vsr + 0.5);
	}

	return tts;
}


size_t NChoose2(size_t n)
{
	return (n * (n-1)) / 2;
}

Array2D<uint16_t> unique_pairs(uint nsig)
{
	
	auto ckeys = Array2D<uint16_t>(NChoose2(nsig), 2);
	size_t row_ix = 0;

	for (uint i = 0; i < nsig; ++i)
	{
		for (uint j = i + 1; j < nsig; ++j)
		{
			ckeys(row_ix, 0) = i;
			ckeys(row_ix, 1) = j;
			row_ix += 1;
		}
	}
	std::cout << "row_ix: " << row_ix << '\n';
	std::cout << "nkeys: " << ckeys.nrow_ << '\n';
	return ckeys;
}


Array2D<uint16_t> unique_pairs(Vector<uint16_t>& keys)
{
	size_t npair = 0;

	// crude way to calc nkeys (wil use dist filters later)
	for (uint i = 0; i < keys.size_; ++i)
	{
		for (uint j = i + 1; j < keys.size_; ++j)
		{
			npair += 1;			
		}
	}

	auto ckeys = Array2D<uint16_t>(npair, 2);
	size_t row_ix = 0;

	for (uint i = 0; i < keys.size_; ++i)
	{
		for (uint j = i + 1; j < keys.size_; ++j)
		{
			ckeys(row_ix, 0) = keys[i];
			ckeys(row_ix, 1) = keys[j];
			row_ix += 1;
		}
	}
	return ckeys;
}

std::vector<uint16_t> UniquePairsFlat(Vector<uint16_t>& keys)
{

	std::vector<uint16_t> ckeys;
	// auto ckeys = Array2D<uint16_t>(npair, 2);
	size_t row_ix = 0;

	for (uint i = 0; i < keys.size_; ++i)
	{
		for (uint j = i + 1; j < keys.size_; ++j)
		{
			ckeys.push_back(keys[i]);
			ckeys.push_back(keys[j]);
		}
	}
	return ckeys;
}

Array2D<uint16_t> AllPairsDistFilt(Vector<uint16_t>& keys, Array2D<float>& stalocs, float min_dist, float max_dist)
{
	// size_t npair = 0;
	size_t npair_max = NChoose2(keys.size_);
	printf("max pairs %lu\n", npair_max);
	auto ckeys = Array2D<uint16_t>(npair_max, 2);
	size_t row_ix = 0;
	float dist;
	float* loc1 = nullptr;
	float* loc2 = nullptr;
	
	for (uint i = 0; i < keys.size_; ++i)
	{
		loc1 = stalocs.row(keys[i]);

		for (uint j = i + 1; j < keys.size_; ++j)
		{
			loc2 = stalocs.row(keys[j]);
			dist = DistCartesian(loc1, loc2);
			// printf("[%u,%u] %.2f \n",i, j, dist);
			if (dist > min_dist && dist < max_dist)
			{
				// printf("%u\n", row_ix);
				ckeys(row_ix, 0) = keys[i];
				ckeys(row_ix, 1) = keys[j];
				row_ix += 1;
			}			
		}
	}

	auto ckeys2 = Array2D<uint16_t>(row_ix, 2);
	for(uint i = 0; i < ckeys2.size_; ++i) {
		ckeys2[i] = ckeys[i];
	}

	return ckeys2;
}

Array2D<uint16_t> AllPairsDistAngleFilt(Vector<uint16_t>& keys, Array2D<float>& stalocs, float min_dist, float max_dist)
{
	// size_t npair = 0;
	size_t npair_max = NChoose2(keys.size_);
	printf("max pairs %lu\n", npair_max);
	auto ckeys = Array2D<uint16_t>(npair_max, 2);
	size_t row_ix = 0;
	float dist;
	float* loc1 = nullptr;
	float* loc2 = nullptr;
	float angle;
	
	for (uint i = 0; i < keys.size_; ++i)
	{
		loc1 = stalocs.row(keys[i]);

		for (uint j = i + 1; j < keys.size_; ++j)
		{
			loc2 = stalocs.row(keys[j]);
			dist = DistCartesian(loc1, loc2);
			// printf("[%u,%u] %.2f \n",i, j, dist);

			if (dist > min_dist && dist < max_dist)
			{	
				angle = AngleBetweenPoints(loc1, loc2);

				if(angle < -0.14 || angle > -0.10) {
					ckeys(row_ix, 0) = keys[i];
					ckeys(row_ix, 1) = keys[j];
					row_ix += 1;
				}
			}			
		}
	}

	auto ckeys2 = Array2D<uint16_t>(row_ix, 2);
	for(uint i = 0; i < ckeys2.size_; ++i) {
		ckeys2[i] = ckeys[i];
	}

	return ckeys2;
}



Array2D<uint16_t> BuildNPairsDistFilt(Vector<uint16_t>& keys, Array2D<float>& stalocs, float min_dist, float max_dist, uint ncc)
{
	auto ckeys = Array2D<uint16_t>(ncc, 2);
	
	std::mt19937::result_type seed = time(0);
	auto rand_int = std::bind(std::uniform_int_distribution<uint>(0, keys.size_), std::mt19937(seed));

	uint16_t k1, k2;
	float dist;
	float* loc1 = nullptr;
	float* loc2 = nullptr;

	uint i = 0;
	while(i < ncc) {
		k1 = rand_int();
		k2 = rand_int();
		if(k1 != k2) {
			loc1 = stalocs.row(keys[k1]);
			loc2 = stalocs.row(keys[k2]);
			dist = DistCartesian(loc1, loc2);
			// printf("[%u,%u] %.2f \n",i, j, dist);
			if (dist > min_dist && dist < max_dist)
			{	
				ckeys.row(i)[0] = keys[k1];
				ckeys.row(i)[1] = keys[k2];
				i++;
			}
		}
	}
	
	return ckeys;
}


Array2D<uint16_t> BuildNPairsDistAngleFilt(Vector<uint16_t>& keys, Array2D<float>& stalocs, float min_dist, float max_dist, uint ncc)
{
	auto ckeys = Array2D<uint16_t>(ncc, 2);
	
	std::mt19937::result_type seed = time(0);
	auto rand_int = std::bind(std::uniform_int_distribution<uint>(0, keys.size_), std::mt19937(seed));

	uint16_t k1, k2; 
	float dist, angle;
	float* loc1 = nullptr;
	float* loc2 = nullptr;

	uint i = 0;
	while(i < ncc) {
		k1 = rand_int();
		k2 = rand_int();
		if(k1 != k2) {
			loc1 = stalocs.row(keys[k1]);
			loc2 = stalocs.row(keys[k2]);
			dist = DistCartesian(loc1, loc2);
			// printf("[%u,%u] %.2f \n",i, j, dist);
			if (dist > min_dist && dist < max_dist)
			{	
				angle = AngleBetweenPoints(loc1, loc2);

				if(angle < -0.14 || angle > -0.10) {
					ckeys.row(i)[0] = keys[k1];
					ckeys.row(i)[1] = keys[k2];
					i++;
				}
			}
		}
	}

	
	return ckeys;
}

uint TotalNPairsDistAngleFilt(Vector<uint16_t>& keys, Array2D<float>& stalocs, float min_dist, float max_dist)
{
	// auto ckeys = Array2D<uint16_t>(ncc, 2);
	uint ncc = 0;
	// std::mt19937::result_type seed = time(0);
	// auto rand_int = std::bind(std::uniform_int_distribution<uint>(0, keys.size_), std::mt19937(seed));

	uint16_t k1, k2; 
	float dist, angle;
	float* loc1 = nullptr;
	float* loc2 = nullptr;

	uint ntot = 0;

	for(size_t i = 0; i < keys.size_; ++i) {
		for(size_t j = i + 1; j < keys.size_; ++j) {
			ntot++;	
			loc1 = stalocs.row(keys[i]);
			loc2 = stalocs.row(keys[j]);
			dist = DistCartesian(loc1, loc2);
			// printf("[%u,%u] %.2f \n",i, j, dist);
			if (dist > min_dist && dist < max_dist)
			{	
				angle = AngleBetweenPoints(loc1, loc2);

				if(angle < -0.14 || angle > -0.10) {					
					ncc++;
				}
			}
		}
	}
	std::cout << "nt: " << ntot << '\n';
	return ncc;
}



Vector<uint16_t> GetStationKeysNear(Vector<float>& loc, Array2D<float>& stalocs, float max_dist) {

	std::vector<uint16_t> stakeep;
	float dist;

	for(size_t i = 0; i < stalocs.nrow_; ++i) {

		dist = DistCartesian2D(loc.data_, stalocs.row(i));		
		if(dist < max_dist) {
			stakeep.push_back(i);
		}
	}
	auto out = Vector<uint16_t>(stakeep);

	return out;
}

Vector<float> DistDiffFromCkeys(Array2D<uint16_t>& ckeys, Array2D<float>& stalocs, float sr) {

	auto dist_diff = Vector<float>(ckeys.nrow_);

	uint16_t *ckp = nullptr;

	for(size_t i = 0; i < ckeys.nrow_; ++i) {
		ckp = ckeys.row(i);

		dist_diff[i] = DistCartesian(stalocs.row(ckp[0]), stalocs.row(ckp[1]));
		
	}
	return dist_diff;
}

std::vector<float> MaxAndLoc(Vector<float>& power, Array2D<float>& gridlocs) {

	size_t amax = std::distance(power.data_,
			 std::max_element(power.begin(), power.end()));

	// float max = output.max();
	// size_t amax = output.argmax();
	float *wloc = gridlocs.row(amax);

	std::vector<float> stats = {power[amax], wloc[0], wloc[1], wloc[2]};
	return stats;
}



// std::vector<uint16_t> GetStationKeysNear(Vector<float>& loc, Array2D<float>& stalocs, float max_dist) {

// 	std::vector<uint16_t> stakeep;
// 	float dist;

// 	for(size_t i = 0; i < stalocs.nrow_; ++i) {

// 		dist = DistCartesian2D(loc.data_, stalocs.row(i));		
// 		if(dist < max_dist) {
// 			stakeep.push_back(i);
// 		}
// 	}
// 	// auto out = Vector<uint16_t>(stakeep);
// 	return stakeep;
// }


// Array2D<float> EnergyCC(Array2D<float>& data_cc) {
// 	/* code */
// }

// auto energy_cc = Array2D<float>(data_cc.nrow_, 4);

// 		float e1;
// 		float e2;
// 		float enoise1;
// 		float enoise2;
// 		float *sig = nullptr;
// 		uint ixp;
// 		uint hlen = npts / 2;

// 		for(size_t i = 0; i < data_cc.nrow_; ++i) {
// 			sig = data_cc.row(i);
// 			ixp = ixphys[i];
// 			printf("%d\n", i);
// 			enoise1 = process::rms_energy(sig + ixp, hlen - ixp);
// 			enoise2 = process::rms_energy(sig + hlen, hlen - ixp);
// 			e1 = process::rms_energy(sig, ixphys[i]);
// 			e2 = process::rms_energy(sig + npts - ixphys[i], npts);
// 			energy_cc(i, 0) = e1;
// 			energy_cc(i, 1) = e2;
// 			energy_cc(i, 2) = enoise1;
// 			energy_cc(i, 3) = enoise2;
// 		}
		


// void tt_homo_ix(Array2D<float> &sta_locs, float *src_loc, float vsr, Vector<size_t> &tts)
// {	
// 	float dist;
// 	for (size_t j = 0; j < tts.size_; ++j) {
// 		dist = beamform::DistCartesian(src_loc, sta_locs.row(j));
// 		tts[j] = static_cast<size_t>(dist * vsr + 0.5);
// 	}
// }


// void tt_homo(Vector<float> &tts, Array2D<float> &sta_locs, float *src_loc, float velocity)
// {
// 	float dist;

// 	for (int32_t j = 0; j < sta_locs.nrow_; ++j) {
// 		// dist = DistCartesian(src_loc, &sta_locs[j]);
// 		dist = DistCartesian(src_loc, sta_locs.row(j));
// 		tts[j] = dist / velocity;
// 	}
// }

// void tt_diff(float *tts, float *tts_cc, int *ckeys, int ncc)
// {
// 	int key1, key2;

// 	for (int i = 0; i < ncc; ++i) {
// 		key1 = ckeys[i * 2];
// 		key2 = ckeys[i * 2 + 1];
// 		// printf("%d_%d\n", key1, key2);
// 		tts_cc[i] = tts[key2] - tts[key1];
// 	}
// }


// float slant_stack(Array2D<float> &data, Vector<float> &tts, Vector<int> &ix_keep, float src_time, float sr)
// {
// 	float sum = 0;
// 	int nvalid = 0;
// 	int32_t col_ix;
	
// 	for (int32_t i = 0; i < ix_keep.size_; ++i) {
// 		col_ix = (src_time + tts[i]) * sr;

// 		if (0 <= col_ix && col_ix < data.ncol_) {
// 			sum += data(i, col_ix);
// 			// printf("i: %d col_ix: %d, val: %.2f\n", i, col_ix, sig_ptr[col_ix]);
// 			nvalid++;
// 		}
// 	}
// 	if (nvalid == 0){
// 		return 0;
// 	}
// 	else {
// 		return sum / nvalid;
// 	}
// }


// float slant_stack_no_check(Array2D<float> &data, Vector<float> &tts, Vector<int> &ix_keep, float src_time, float sr){
// 	// No bounds checking, care for segfaults
// 	float sum = 0;
// 	// float t_exp;	
// 	// int32_t col_ix;
// 	int col_ix;

// 	for (int32_t i = 0; i < ix_keep.size_; ++i) {
// 		// col_ix = (int32_t) (sr * (src_time + tts[i]));
// 		// col_ix = (int32_t) (sr * (src_time + tts[i]));
// 		// col_ix = static_cast<int32_t>((src_time + tts[i]) * sr);
// 		col_ix = static_cast<int>((src_time + tts[i]) * sr);
// 		// col_ix = (int) ((src_time + tts[i]) * sr);
// 		sum += data(i, col_ix);
// 		// sum += data(i, col_ix);
// 	}
// 	return sum / ix_keep.size_;
// }


// void beampower_homo(Array2D<float> &points, Vector<float> &out, Array2D<float> &data, Vector<float> &tts, Array2D<float> &sta_locs, Vector<int> &ix_keep, float velocity, float src_time, float sr) {

// 	// Vector<float> &tts2 = Vector<float>(&&tts[0], tts.size_);

// 	for (int32_t i = 0; i < points.nrow_; ++i) {
// 		tt_homo(tts, sta_locs, points.row(i), velocity);
// 		// tts2.set_data(&tts[0]);
// 		// out[i] = slant_stack(data, tts, ix_keep, src_time, sr);
// 		out[i] = slant_stack_no_check(data, tts, ix_keep, src_time, sr);
// 	}
// }



// void search_grid(Array2D<float>& data, Array2D<float>& locs,
// 			Grid& grid, size_t gix_start, size_t gix_end,
// 			size_t nt_search, float vsr,
// 			float* win_val, size_t* win_loc)
// {
// 	size_t nchan = data.nrow_;
// 	auto tt_ixs = Vector<size_t>(nchan);	
// 	auto output = Vector<float>(nt_search);
// 	float src_loc[3];
// 	float *dptr = nullptr;
// 	float dist;

// 	printf("Searching grid points: %lu to %lu\n", gix_start, gix_end);

// 	for (size_t ipt = gix_start; ipt < gix_end; ++ipt)
// 	{	
// 		// if (ipt % 1000 == 0) {printf("Point: %d / %d\n", ipt, (int) gix_end);}
// 		// grid.get_point(ipt, src_loc);

// 		if (ipt % 1000 == 0) {
// 			printf("Progress: %.2f \n", ((float)(ipt - gix_start) / (gix_end - gix_start) * 100));
// 		}

// 		grid.get_point(ipt, src_loc);
// 		beamform::tt_homo_ix(locs, src_loc, vsr, tt_ixs);

// 		output.fill(0);

// 		// For each channel add time comb values to output
// 		for (size_t i = 0; i < nchan; ++i) 
// 		{
// 			dptr = data.row(i) + tt_ixs[i];
		
// 			for (size_t j = 0; j < nt_search; j++) 
// 			{
// 				output[j] += dptr[j];
// 			}
// 		}

// 		for (size_t j = 0; j < nt_search; ++j) 
// 		{
// 			if (std::abs(output[j]) > std::abs(win_val[j])) {
// 				win_val[j] = output[j];
// 				win_loc[j] = ipt;
// 			}
// 		}
	
// 	}

// }

// void search_grid_parallel(std::vector<size_t>& parts, Array2D<float>& data, Array2D<float>& locs, Grid& grid, size_t nt_search, float vsr,	Array2D<float>& win_val, Array2D<size_t>& win_loc)
// {
// 	std::vector<std::thread> pool;

// 	for (size_t i = 0; i < parts.size() - 1; i++){

// 		pool.push_back(std::thread([=, &data, &locs, &grid, &win_val, &win_loc] {search_grid(data, locs, grid, parts[i], parts[i + 1], nt_search, vsr, win_val.row(i), win_loc.row(i));}));
// 	}

// 	for(auto& thread : pool) thread.join();
// }



}

#endif
