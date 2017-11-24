/*
This software is a core part of AlphaPathMoves and is provied "AS IS" without any warranty.
Copyright holder Hossam Isack <isack.hossam@gmail.com>.
*/
#ifndef __MARGINEDGESCUDA_CUH__
#define __MARGINEDGESCUDA_CUH__
#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <cstdint>
#include "Utilities.h"


using namespace Utilities;
class MarginEdgesCuda
{

private:
	uint32_t  max_path;
	int64_t nlabels, ndspl, total_shifts_size, dims[3];
	uint64_t n_banks, n_usedbits, *margin_edges_64, *dev_marign_edges_64;
	int32_t *x2ypath, *shifts, *label2shifts, *child2parent, *shift2bitid, max_r;
	int32_t *dev_x2ypath, *dev_shifts, *dev_label2shifts, *dev_child2parent, *dev_shift2bitid;
	uint32_t *dev_labeling;
	int64_t *dev_dims;
	bool lessthan35mode;

public:
	MarginEdgesCuda(int64_t, int64_t  *, int32_t *, uint32_t, Array3D<int32_t> *
		, Array2D<int32_t> **, int32_t ***, int32_t, uint64_t *, uint64_t, uint64_t);
	void runKernel(uint32_t, uint32_t, uint32_t *);
	~MarginEdgesCuda();
};



#endif

