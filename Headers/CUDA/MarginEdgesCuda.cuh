#ifndef __MARGINEDGESCUDA_CUH__
#define __MARGINEDGESCUDA_CUH__
#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include "Utilities.h"


using namespace Utilities;
class MarginEdgesCuda
{

private:
	unsigned long int  max_path;
	ssize_t nlabels, ndspl, total_shifts_size, dims[3];
	size_t n_banks, n_usedbits, *margin_edges_64, *dev_marign_edges_64;
	long int *x2ypath, *shifts, *label2shifts, *child2parent, *shift2bitid, max_r;
	long int *dev_x2ypath, *dev_shifts, *dev_label2shifts, *dev_child2parent, *dev_shift2bitid;
	unsigned long int *dev_labeling;
	ssize_t *dev_dims;
	bool lessthan35mode;

public:
	MarginEdgesCuda(ssize_t, ssize_t  *, long int *, unsigned long int, Array3D<long int> *
		, Array2D<long int> **, long int ***, long int, size_t *, size_t, size_t);
	void runKernel(unsigned long int, unsigned long int, unsigned long int *);
	~MarginEdgesCuda();
};



#endif

