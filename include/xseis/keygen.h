/*
Beamforming functions.
*/
#ifndef KEYGEN_H
#define KEYGEN_H

#include "xseis/structures.h"
#include "xseis/process.h"
#include "xseis/utils.h"
#include <random>

typedef std::vector<std::vector<uint16_t>> ChanGroups; 

namespace keygen {


size_t NChoose2(size_t n)
{
	return (n * (n-1)) / 2;
}


ChanGroups RandGroupsN(std::vector<uint16_t> keys, uint32_t ngroup) {
	
	// auto keys = utils::arange<uint16_t>(0, nkeys);
	auto kc = keys;	
	std::srand(std::time(0));
	std::random_shuffle(kc.begin(), kc.end());

	uint32_t chunk = keys.size() / ngroup;
	uint32_t remain = keys.size() % ngroup;
	ChanGroups kgroup;

	uint32_t c0 = chunk + 1;

	for(size_t i = 0; i < remain; ++i) {
		// auto last = i + chunk + 1;
		auto first = i * c0;
		auto last = (i + 1) * c0;
		kgroup.emplace_back(kc.begin() + first, kc.begin() + last);
	}

	c0 = chunk;
	for(size_t i = remain; i < ngroup; ++i) {
		// auto last = i + chunk + 1;
		auto first = i * c0;
		auto last = (i + 1) * c0;
		kgroup.emplace_back(kc.begin() + first, kc.begin() + last);
	}

	// for(size_t i = ; i < kc.size(); i += chunk) {
	// 	auto last = std::min(kc.size(), i + chunk);
	// 	kgroup.emplace_back(kc.begin() + i, kc.begin() + last);
	// }

	return kgroup;
}



ChanGroups GroupChannels(std::vector<uint16_t>& keys, Vector<uint16_t>& chan_map)
{

	ChanGroups groups;
	size_t nkeys = keys.size();

	for(auto&& k : keys) {

		std::vector<uint16_t> group;
		
		for(size_t j = 0; j < chan_map.size_; ++j) {
			if (k == chan_map[j]) {
				group.push_back(j);				
			}			
		}
		groups.push_back(group);
	}
	
	return groups;
}



Array2D<uint16_t> DistFilt(std::vector<uint16_t>& keys, Array2D<float>& stalocs, float min_dist, float max_dist)
{
	// size_t npair = 0;

	size_t nkeys = keys.size();
	// size_t npair_max = NChoose2(nkeys);
	std::vector<uint16_t> ckeys_flat;
	// size_t row_ix = 0;
	float dist;
	float* loc1 = nullptr;
	float* loc2 = nullptr;
	
	for (uint i = 0; i < nkeys; ++i)
	{
		loc1 = stalocs.row(keys[i]);

		for (uint j = i + 1; j < nkeys; ++j)
		{
			loc2 = stalocs.row(keys[j]);
			dist = process::DistCartesian(loc1, loc2);

			if (dist > min_dist && dist < max_dist)
			{
				ckeys_flat.push_back(keys[i]);
				ckeys_flat.push_back(keys[j]);				
			}			
		}
	}
	auto ckeys = Array2D<uint16_t>(ckeys_flat, 2);

	return ckeys;
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


std::vector<uint16_t> AllPairsContamFilt(Vector<uint16_t>& keys, Array2D<float>& stalocs,
					 std::vector<std::vector<float>> sources, float min_dist, float max_dist)
{
	
	std::vector<uint16_t> ckeys;
	
	for (uint i = 0; i < keys.size_; ++i)
	{
		float* loc1 = stalocs.row(keys[i]);

		for (uint j = i + 1; j < keys.size_; ++j)
		{
			float* loc2 = stalocs.row(keys[j]);
			float dist = process::DistCartesian(loc1, loc2);

			if (dist < min_dist && dist > max_dist) continue;

			bool pass = true;
			for(auto&& src : sources) {
				float dd = process::DistDiff(loc1, loc2, src.data());
				float dcut = src[3];
				if(dd < dcut) pass = false;
			}

			if (pass == false) continue;

			ckeys.push_back(keys[i]);
			ckeys.push_back(keys[j]);
		}
	}

	return ckeys;
}

std::vector<uint16_t> AllPairsFilt(Vector<uint16_t>& keys, Array2D<float>& stalocs, float min_dist, float max_dist, bool ang_filt=true)
{

	float min_ang = -0.14;
	float max_ang = -0.10;	

	float* loc1 = nullptr;
	float* loc2 = nullptr;
	float angle;
	float dist;

	std::vector<uint16_t> ckeys;
	
	for (size_t i = 0; i < keys.size_; ++i)
	{
		loc1 = stalocs.row(keys[i]);

		for (size_t j = i + 1; j < keys.size_; ++j)
		{
			loc2 = stalocs.row(keys[j]);
			dist = process::DistCartesian(loc1, loc2);
			// printf("[%u,%u] %.2f \n",i, j, dist);

			if (dist > min_dist && dist < max_dist)
			{	

				if (ang_filt == true)
				{
					angle = process::AngleBetweenPoints(loc1, loc2);
					if(min_ang < angle && angle < max_ang) {
						continue;
					}
				}
				ckeys.push_back(keys[i]);
				ckeys.push_back(keys[j]);					
			}			
		}
	}

	return ckeys;
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
			dist = process::DistCartesian(loc1, loc2);
			// printf("[%u,%u] %.2f \n",i, j, dist);

			if (dist > min_dist && dist < max_dist)
			{	
				angle = process::AngleBetweenPoints(loc1, loc2);

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


// RNG broken check later
Array2D<uint16_t> BuildNPairsDistFilt(std::vector<uint16_t>& keys, Array2D<float>& stalocs, float min_dist, float max_dist, uint32_t ncc)
{
	auto ckeys = Array2D<uint16_t>(ncc, 2);
	uint32_t nkeys = keys.size();

	std::cout << "nkeys: " << nkeys << '\n';
	// std::mt19937::result_type seed = time(0);
	// std::mt19937::result_type seed = 1;
	std::mt19937::result_type seed = 0;
	auto rand_int = std::bind(std::uniform_int_distribution<uint32_t>(0, nkeys - 1), std::mt19937(seed));

	float dist;
	float* loc1 = nullptr;
	float* loc2 = nullptr;
	// uint32_t k1, k2;

	size_t i = 0;
	while(i < ncc) {
		uint32_t k1 = rand_int();
		uint32_t k2 = rand_int();		
		assert(k1 < keys.size());
		assert(k2 < keys.size());

		if(k1 != k2) {
			loc1 = stalocs.row(keys[k1]);
			loc2 = stalocs.row(keys[k2]);
			dist = process::DistCartesian(loc1, loc2);
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
	auto rand_int = std::bind(std::uniform_int_distribution<uint>(0, keys.size_ - 1), std::mt19937(seed));

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
			dist = process::DistCartesian(loc1, loc2);
			// printf("[%u,%u] %.2f \n",i, j, dist);
			if (dist > min_dist && dist < max_dist)
			{	
				angle = process::AngleBetweenPoints(loc1, loc2);

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
			dist = process::DistCartesian(loc1, loc2);
			// printf("[%u,%u] %.2f \n",i, j, dist);
			if (dist > min_dist && dist < max_dist)
			{	
				angle = process::AngleBetweenPoints(loc1, loc2);

				if(angle < -0.14 || angle > -0.10) {					
					ncc++;
				}
			}
		}
	}
	std::cout << "nt: " << ntot << '\n';
	return ncc;
}



Vector<uint16_t> GetStationKeysNear(float loc[3], Array2D<float>& stalocs, float max_dist) {

	std::vector<uint16_t> stakeep;

	for(size_t i = 0; i < stalocs.nrow_; ++i) {
		
		if ( process::DistCartesian2D(loc, stalocs.row(i)) < max_dist) {
			stakeep.push_back(i);
		}
	}
	auto out = Vector<uint16_t>(stakeep);

	return out;
}


std::vector<uint16_t> WithinDistOf(float loc[3], Array2D<float>& stalocs, float max_dist) {

	std::vector<uint16_t> stakeep;

	for(size_t i = 0; i < stalocs.nrow_; ++i) {		
		if (process::DistCartesian2D(loc, stalocs.row(i)) < max_dist) {
			stakeep.push_back(i);
		}
	}

	return stakeep;
}


Vector<float> DistDiffFromCkeys(Array2D<uint16_t>& ckeys, Array2D<float>& stalocs, float sr) {

	auto dist_diff = Vector<float>(ckeys.nrow_);

	uint16_t *ckp = nullptr;

	for(size_t i = 0; i < ckeys.nrow_; ++i) {
		ckp = ckeys.row(i);

		dist_diff[i] = process::DistCartesian(stalocs.row(ckp[0]), stalocs.row(ckp[1]));
		
	}
	return dist_diff;
}

// // Build groups of ckeys for stas within radius of mid_stas
// std::vector<std::vector<uint16_t>> CkeyPatchesFromStations(std::vector<uint>& mid_stas, Array2D<float>& stalocs, float radius, float cdist_min, float cdist_max, bool ang_filt=true) 
// {

// 	// std::vector<uint16_t> ckeys_vec;
// 	std::vector<std::vector<uint16_t>> patches;

// 	for(auto&& ix : mid_stas) {
// 		auto loc_patch = stalocs.row_view(ix);
// 		auto pkeys = GetStationKeysNear(loc_patch, stalocs, radius);
// 		patches.push_back(AllPairsFilt(pkeys, stalocs, cdist_min, cdist_max, ang_filt));
// 	}
// 	return patches;
// }

// Builds ckey index from groups of ckeys
std::vector<std::vector<uint>> IndexesFromCkeyPatches(std::vector<std::vector<uint16_t>>& patches) 
{
	std::vector<std::vector<uint>> ipatches;

	size_t csum = 0;
	for(auto&& patch : patches) {
		size_t ncc = patch.size() / 2;
		std::vector<uint> ipatch;
		ipatch.reserve(ncc);

		for(size_t i = 0; i < ncc; ++i) {
			ipatch.push_back(i + csum);
		}
		ipatches.push_back(ipatch);
		csum += ncc;
	}
	return ipatches;
}

// Builds ckey index from groups of ckeys
Array2D<uint16_t> CkeysFromPatches(std::vector<std::vector<uint16_t>>& patches) 
{
	// std::vector<std::vector<size_t>> ipatches;
	size_t ncc = 0;
	for(auto&& patch : patches) {ncc += patch.size() / 2;}

	auto ckeys = Array2D<uint16_t>(ncc, 2);

	size_t csum = 0;
	auto ptr = ckeys.data_;

	for(auto&& vec : patches) {
		std::copy(vec.begin(), vec.end(), ptr);
		ptr += vec.size();		
	}
	return ckeys;
}




// ckeys_vec.insert(ckeys_vec.end(), ck_patch.begin(), ck_patch.end());
// uint patch_len = ck_patch.size() / 2;

// std::vector<uint> ipatch(patch_len);
// for(size_t i = 0; i < patch_len; ++i) {
// 	ipatch.push_back(i + csum);
// }
// ipatches.push_back(ipatch);
// csum += patch_len;
// std::cout << "nkeys: " << patch_len << '\n';



// std::vector<float> MaxAndLoc(Vector<float>& power, Array2D<float>& gridlocs) {

// 	size_t amax = std::distance(power.data_,
// 			 std::max_element(power.begin(), power.end()));

// 	// float max = output.max();
// 	// size_t amax = output.argmax();
// 	float *wloc = gridlocs.row(amax);

// 	std::vector<float> stats = {power[amax], wloc[0], wloc[1], wloc[2]};
// 	return stats;
// }



// Array2D<float> InterLocOld(Array2D<float>& data_cc, Array2D<uint16_t>& ckeys, Array2D<uint16_t>& ttable, uint16_t nthreads)
// {
// 	// Each thread given own output buffer to prevent cache invalidations

// 	const size_t cclen = data_cc.ncol_;
// 	const size_t ncc = data_cc.nrow_;
// 	const size_t ngrid = ttable.ncol_;

// 	uint16_t *tts_sta1, *tts_sta2;
// 	float *cc_ptr = nullptr;

// 	auto output = Array2D<float>(nthreads, ngrid);
// 	size_t niter = 0;
// 	#pragma omp parallel private(tts_sta1, tts_sta2, cc_ptr) num_threads(nthreads)
// 	{
// 		float *out_ptr = output.row(omp_get_thread_num());
// 		std::fill(out_ptr, out_ptr + ngrid, 0);
// 		// play around with omp loop scheduling here
// 		#pragma omp for
// 		for (size_t i = 0; i < ncc; ++i)
// 		{
// 			// if (i % 10000 == 0) {
// 			// 	printf("Prog: %.2f \r", ((float) i / ncc * 100));
// 			// 	std::cout.flush();
// 			// }

// 			tts_sta1 = ttable.row(ckeys(i, 0));	
// 			tts_sta2 = ttable.row(ckeys(i, 1));
// 			cc_ptr = data_cc.row(i);

// 			// Migrate single ccf on to grid based on tt difference
// 			#pragma omp simd \
// 			aligned(tts_sta1, tts_sta2, out_ptr, cc_ptr: MEM_ALIGNMENT)
// 			for (size_t j = 0; j < ngrid; ++j)
// 			{
// 				// Get appropriate ix of unrolled ccfs (same as mod_floor)
// 				// by wrapping negative traveltime differences
// 				// if-else much faster than more elegant mod function
// 				if (tts_sta2[j] >= tts_sta1[j])
// 				{
// 					out_ptr[j] += cc_ptr[tts_sta2[j] - tts_sta1[j]];					
// 				}
// 				else
// 				{
// 					out_ptr[j] += cc_ptr[cclen - tts_sta1[j] + tts_sta2[j]];
// 				}
// 			}
// 		}
// 	}	

// 	return output;
// }


// std::vector<uint16_t> GetStationKeysNear(Vector<float>& loc, Array2D<float>& stalocs, float max_dist) {

// 	std::vector<uint16_t> stakeep;
// 	float dist;

// 	for(size_t i = 0; i < stalocs.nrow_; ++i) {

// 		dist = process::DistCartesian2D(loc.data_, stalocs.row(i));		
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
// 		dist = beamform::process::DistCartesian(src_loc, sta_locs.row(j));
// 		tts[j] = static_cast<size_t>(dist * vsr + 0.5);
// 	}
// }


// void tt_homo(Vector<float> &tts, Array2D<float> &sta_locs, float *src_loc, float velocity)
// {
// 	float dist;

// 	for (int32_t j = 0; j < sta_locs.nrow_; ++j) {
// 		// dist = process::DistCartesian(src_loc, &sta_locs[j]);
// 		dist = process::DistCartesian(src_loc, sta_locs.row(j));
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
