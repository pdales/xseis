#ifndef MSEED_H
#define MSEED_H

#include <iostream>
#include <string>
#include <map>
#include <libmseed.h>
#include "xseis/structures.h"

#include <fstream>
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;


namespace mseed {



std::vector<std::string> FilesMatching(std::string path, std::string ext)
{	
    std::vector<std::string> v;

    for(auto& p: fs::recursive_directory_iterator(path))  {
        if(p.path().extension() == ext) v.push_back(p.path());
    }
    return v;
}



template<typename KeyType, typename ValueType> 
std::pair<KeyType,ValueType> MapMaxVal(const std::map<KeyType,ValueType>& x ) {
  using pairtype=std::pair<KeyType,ValueType>; 
  return *std::max_element(x.begin(), x.end(), [] (const pairtype & p1, const pairtype & p2) {
        return p1.second < p2.second;
  }); 
}

std::string Concat(char* sta, char* chan, uint16_t npad=3) {
	std::stringstream ss;
	ss << std::setw(npad) << std::setfill('0') << sta;
	ss << "_" << chan;
	return ss.str();
}

// char* file2buf(char* FLE){
std::vector<char> FileToBuf(char* FLE){
	std::ifstream fin(FLE);
	// get pointer to associated buffer object
	std::filebuf* pbuf = fin.rdbuf();
	// get file size using buffer's members
	size_t size = pbuf->pubseekoff(0,fin.end,fin.in);
	pbuf->pubseekpos (0,fin.in);
	// allocate memory to contain file data
	// char* buffer=new char[size];
	std::vector<char> buffer (size);
	// get file data
	pbuf->sgetn(&buffer[0], size);
	fin.close();
	return buffer;
}


Array2D<float> ToData(std::vector<char>& fbuf, std::map<std::string, size_t>& chanmap, size_t& epoch, uint64_t reclen=4096) {

	char* buffer = &fbuf[0];
	static flag verbose = 1;
	int retcode;
	
	MSRecord *msr = 0;
	int64_t totalrecs = 0;
	uint32_t nrecs = fbuf.size() / reclen;

	std::vector<size_t> starttimes;

	std::cout << "nrecs: " << nrecs << '\n';
	// build map of unique channels
	for(size_t i = 0; i < nrecs; ++i) {
		msr_parse(buffer + i * reclen, reclen, &msr, reclen, false, verbose);
		size_t nsamp = msr->samplecnt;
		std::string key = Concat(msr->station, msr->channel);
		if(chanmap.count(key) == 0) chanmap[key] = nsamp;		
		else chanmap[key] += nsamp;
		starttimes.push_back(msr->starttime);
		// std::cout << "key: " << key << '\n';
	}
	// clock.log("meta");
	// return 0;

	epoch = *std::min_element(starttimes.begin(), starttimes.end());
	std::cout << "epoch: " << epoch << '\n';
	// get longest channel length
	// size_t nmax = utils::PadToBytes<float>(utils::MapMaxVal(chanmap).second);
	size_t nmax = MapMaxVal(chanmap).second;
	// replace map values (prev nsamp) with buf row index
	uint32_t irow = 0;
	for(auto& x : chanmap) x.second = irow++;

	// for(const auto& n : chanmap ) {
	// 	std::cout << "Key:[" << n.first << "] Value:[" << n.second << "]\n";
	// }

	auto buf = Array2D<float>(chanmap.size(), nmax);
	buf.fill(0);

	std::cout << "nmax: " << nmax << '\n';
	// clock.log("alloc buffer");

	for(size_t i = 0; i < nrecs; ++i) {
		msr_parse(buffer + i * reclen, reclen, &msr, reclen, true, verbose);
		size_t ts = msr->starttime;
		size_t icol = (ts - epoch) * msr->samprate / 1000000. + 0.5;
		size_t nsamp = msr->samplecnt;
		std::string key = Concat(msr->station, msr->channel);
		float *sptr = (float *) msr->datasamples;		
		auto ptr_buf = buf.row(chanmap[key]) + icol;
		std::copy(sptr, sptr + nsamp, ptr_buf);
	}

	return buf;	
}

Array2D<float> ToDataFixed(std::vector<char>& fbuf, std::map<std::string, size_t>& chanmap, size_t& epoch, uint32_t maxlen, uint64_t reclen=4096) {

	char* buffer = &fbuf[0];
	static flag verbose = 1;
	int retcode;
	
	MSRecord *msr = 0;
	int64_t totalrecs = 0;
	uint32_t nrecs = fbuf.size() / reclen;

	std::vector<size_t> starttimes;

	std::cout << "nrecs: " << nrecs << '\n';
	// build map of unique channels
	for(size_t i = 0; i < nrecs; ++i) {
		msr_parse(buffer + i * reclen, reclen, &msr, reclen, false, verbose);
		size_t nsamp = msr->samplecnt;
		std::string key = Concat(msr->station, msr->channel);
		if(chanmap.count(key) == 0) chanmap[key] = nsamp;		
		else chanmap[key] += nsamp;
		starttimes.push_back(msr->starttime);
		// std::cout << "key: " << key << '\n';
	}
	// clock.log("meta");
	// return 0;

	epoch = *std::min_element(starttimes.begin(), starttimes.end());
	std::cout << "epoch: " << epoch << '\n';
	// get longest channel length
	// size_t nmax = utils::PadToBytes<float>(utils::MapMaxVal(chanmap).second);
	// size_t nmax = MapMaxVal(chanmap).second;

	// replace map values (prev nsamp) with buf row index
	uint32_t irow = 0;
	for(auto& x : chanmap) x.second = irow++;

	auto buf = Array2D<float>(chanmap.size(), maxlen);
	buf.fill(0);

	// std::cout << "nmax: " << nmax << '\n';

	for(size_t i = 0; i < nrecs; ++i) {
		msr_parse(buffer + i * reclen, reclen, &msr, reclen, true, verbose);
		size_t ts = msr->starttime;
		size_t icol = (ts - epoch) * msr->samprate / 1000000. + 0.5;
		size_t nsamp = msr->samplecnt;
		if (icol + nsamp < maxlen)
		{				
			std::string key = Concat(msr->station, msr->channel);
			float *sptr = (float *) msr->datasamples;		
			auto ptr_buf = buf.row(chanmap[key]) + icol;
			std::copy(sptr, sptr + nsamp, ptr_buf);			
		}
	}

	return buf;	
}

Array2D<float> ToDataFixed(char* buffer, size_t nbytes, std::map<std::string, size_t>& chanmap, size_t& epoch, uint32_t maxlen, uint64_t reclen=4096) {

	static flag verbose = 1;
	int retcode;
	
	MSRecord *msr = 0;
	int64_t totalrecs = 0;
	uint32_t nrecs = nbytes / reclen;

	std::vector<size_t> starttimes;

	std::cout << "nrecs: " << nrecs << '\n';
	// build map of unique channels
	for(size_t i = 0; i < nrecs; ++i) {
		msr_parse(buffer + i * reclen, reclen, &msr, reclen, false, verbose);
		size_t nsamp = msr->samplecnt;
		std::string key = Concat(msr->station, msr->channel);
		if(chanmap.count(key) == 0) chanmap[key] = nsamp;		
		else chanmap[key] += nsamp;
		starttimes.push_back(msr->starttime);
		// std::cout << "key: " << key << '\n';
	}
	// clock.log("meta");
	// return 0;

	epoch = *std::min_element(starttimes.begin(), starttimes.end());
	std::cout << "epoch: " << epoch << '\n';
	// get longest channel length
	// size_t nmax = utils::PadToBytes<float>(utils::MapMaxVal(chanmap).second);
	// size_t nmax = MapMaxVal(chanmap).second;

	// replace map values (prev nsamp) with buf row index
	uint32_t irow = 0;
	for(auto& x : chanmap) x.second = irow++;

	auto buf = Array2D<float>(chanmap.size(), maxlen);
	buf.fill(0);

	// std::cout << "nmax: " << nmax << '\n';

	for(size_t i = 0; i < nrecs; ++i) {
		msr_parse(buffer + i * reclen, reclen, &msr, reclen, true, verbose);
		size_t ts = msr->starttime;
		size_t icol = (ts - epoch) * msr->samprate / 1000000. + 0.5;
		size_t nsamp = msr->samplecnt;
		if (icol + nsamp < maxlen)
		{				
			std::string key = Concat(msr->station, msr->channel);
			float *sptr = (float *) msr->datasamples;		
			auto ptr_buf = buf.row(chanmap[key]) + icol;
			std::copy(sptr, sptr + nsamp, ptr_buf);			
		}
	}

	return buf;	
}
}

#endif

