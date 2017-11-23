#include "MarginEdgesCuda.cuh"

template <class BankType>
__global__ void setMinMarginArcsKernel(BankType *margin_edges_b2t
	, __int64 in_ndspl, unsigned long int in_nlabels, unsigned long int explbl, long int layer_id, unsigned long int in_max_path, long int in_max_r, unsigned __int64 in_n_banks
	, const __int64 * __restrict__ dims
	, const unsigned long int * __restrict__ labeling
	, const long int * __restrict__ paths
	, const long int * __restrict__ label2shifts
	, const long int * __restrict__ shifts
	, const long int * __restrict__ child2parent
	, const long int * __restrict__ shift2bitid)
{
	__int64 q, p = threadIdx.x + blockIdx.x*blockDim.x, ndspl = in_ndspl, constX = in_max_path, constXY = in_nlabels*in_max_path, stride = gridDim.x*blockDim.x;
	__int64 dim0 = dims[0], dim1 = dims[1], dim2 = dims[2], dim01 = dim0*dim1;
	__int64 px, py, pz, qx, qy, qz;
	const long int * P, *Q;
	long int Pi, in_Pi = layer_id, max_path = in_max_path, dX, dY, dZ, g_shiftid;
	long int max_r = in_max_r, b0 = 2 * max_r + 1, b1 = b0*b0;
	__int64 idx, bnk_id, n_banks = in_n_banks;
	__int64 n_bitsperbank = sizeof(BankType) * 8;
	BankType one = 1;
	while (p < ndspl)
	{
		//convert p into px,py,pz
		pz = p / dim01;
		py = (p - pz*dim01) / dim0;
		px = p - py*dim0 - pz*dim01;
		Pi = in_Pi;
		P = paths + labeling[p] * constX + explbl*constXY;
		//bottom-2-top  
		if (P[Pi] != -1 && Pi + 1 < max_path && P[Pi + 1] != -1)
		{
			for (unsigned long int shift_idx = label2shifts[P[Pi]]; shift_idx < label2shifts[P[Pi] + 1]; shift_idx+=3)
			{
				dX = shifts[shift_idx + 0]; qx = px + dX; if (qx < 0 || qx >= dim0) continue;
				dY = shifts[shift_idx + 1]; qy = py + dY; if (qy < 0 || qy >= dim1) continue;
				dZ = shifts[shift_idx + 2]; qz = pz + dZ; if (qz < 0 || qz >= dim2) continue;
				q = qx + qy*dim0 + qz*dim01;
				Q = paths + labeling[q] * constX + explbl*constXY;
				if (Q[Pi + 1] != -1 && Pi + 2 < max_path && Q[Pi + 2] != -1 && Q[Pi + 1] == child2parent[P[Pi]])
				{
					g_shiftid = dX + max_r + (dY + max_r)*b0 + (dZ + max_r)*b1;
					idx = shift2bitid[g_shiftid];
					bnk_id = idx / n_bitsperbank;
					atomicOr(margin_edges_b2t + p*n_banks + bnk_id, (one << (idx - bnk_id * n_bitsperbank)));
				}
			}
		}


		//top-2-bottom
		Pi += 2;
		if (Pi<max_path && P[Pi] != -1 && Pi - 1>0 && Pi - 1 < max_path && P[Pi - 1] != -1) 
		{
			for (unsigned long int shift_id = label2shifts[P[Pi]]; shift_id < label2shifts[P[Pi] + 1]; shift_id += 3)
			{
				dX = shifts[shift_id + 0]; qx = px + dX; if (qx < 0 || qx >= dim0) continue;
				dY = shifts[shift_id + 1]; qy = py + dY; if (qy < 0 || qy >= dim1) continue;
				dZ = shifts[shift_id + 2]; qz = pz + dZ; if (qz < 0 || qz >= dim2) continue;
				q = qx + qy*dim0 + qz*dim01;
				Q = paths + labeling[q] * constX + explbl*constXY;
				if (Q[Pi - 1] != -1 && 
					Q[Pi - 1] == child2parent[P[Pi]]) 
				{
					g_shiftid = -dX + max_r + (-dY + max_r)*b0 + (-dZ + max_r)*b1;
					idx = shift2bitid[g_shiftid];
					bnk_id = idx / n_bitsperbank;
					atomicOr(margin_edges_b2t + q*n_banks + bnk_id, (one << (idx - bnk_id * n_bitsperbank)));
				}
			}
		}
		p += stride;
	}
}
MarginEdgesCuda::MarginEdgesCuda(ssize_t ndspl, ssize_t* dims, long int * child2parent, unsigned long int nlabels
	, Array3D<long int> * x2ypath, Array2D<long int> ** inclusion_shifts, long int ***shift2bitid, long int max_r, size_t * margin_edges
	, size_t n_banks, size_t n_usedbits)
{
	//initilize memebers 
	this->ndspl = ndspl;
	this->dims[0] = dims[0];
	this->dims[1] = dims[1];
	this->dims[2] = dims[2];
	this->nlabels = nlabels;
	this->max_r = max_r;
	this->x2ypath = x2ypath->data;
	this->max_path = (long int) x2ypath->X;
	this->margin_edges_64 = margin_edges;
	this->child2parent = child2parent;
	
	this->label2shifts = new long int[nlabels + 1];
	this->label2shifts[0] = 0;
	for (unsigned long int i = 1; i <= nlabels; ++i) 
		this->label2shifts[i] = this->label2shifts[i - 1] + (long int)inclusion_shifts[i - 1]->Y*3;
	this->total_shifts_size = this->label2shifts[nlabels];
	this->shifts = new long int[this->total_shifts_size];
	for (unsigned long int i = 0, g_shiftidx = 0; i < nlabels; ++i)
	{
		if (inclusion_shifts[i]->totalsize == 0)
			continue;
		memcpy(this->shifts + g_shiftidx, inclusion_shifts[i]->data, sizeof(long int)*inclusion_shifts[i]->totalsize);
		g_shiftidx += (long int) inclusion_shifts[i]->totalsize;
	}

	this->n_banks = n_banks;
	this->n_usedbits = n_usedbits;

	this->shift2bitid = new long int[pow3(2*this->max_r+1)];
	for (long int z = -max_r, g_shiftid = 0; z <= max_r; ++z)
	for (long int y = -max_r; y <= max_r; ++y)
	for (long int x = -max_r; x <= max_r; ++x, ++g_shiftid)
		this->shift2bitid[g_shiftid] = shift2bitid[x][y][z];		

	//create device memeory handels  
	cudaError_t cudaStatus;
	//get device
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) 
	{
		bgn_log << LogType::ERROR << "MinMarginsCUDA: cudaSetDevice failed!: " << cudaGetErrorString(cudaStatus) << end_log;
		exit(EXIT_FAILURE); 
	}
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	
	lessthan35mode = false;
	if (prop.major < 3 && prop.minor < 2)
		lessthan35mode = true;
	//std::cout << "For lessthan35mode is forced to be true\n";
	//lessthan35mode = true;


	// allocate device memory
	cudaStatus = cudaMalloc((void**)&dev_x2ypath, sizeof(long int)*this->nlabels*this->nlabels*this->max_path);
	if (cudaStatus != cudaSuccess) 
	{
		bgn_log << LogType::ERROR << "MinMarginsCUDA: cudaMalloc failed!: " << cudaGetErrorString(cudaStatus) << end_log;
		exit(EXIT_FAILURE); 
	}

	cudaStatus = cudaMalloc((void**)&dev_shifts, sizeof(long int)*this->total_shifts_size);
	if (cudaStatus != cudaSuccess) 
	{
		bgn_log << LogType::ERROR << "MinMarginsCUDA: cudaMalloc failed!: " << cudaGetErrorString(cudaStatus) << end_log;
		exit(EXIT_FAILURE); 
	}

	cudaStatus = cudaMalloc((void**)&dev_label2shifts, sizeof(long int)*(this->nlabels+1));
	if (cudaStatus != cudaSuccess) 
	{ 
		bgn_log << LogType::ERROR << "MinMarginsCUDA: cudaMalloc failed!: " << cudaGetErrorString(cudaStatus) << end_log;
		exit(EXIT_FAILURE); 
	}
	
	cudaStatus = cudaMalloc((void**)&dev_dims, sizeof(ssize_t) * 3);
	if (cudaStatus != cudaSuccess) 
	{
		bgn_log << LogType::ERROR << "MinMarginsCUDA: cudaMalloc failed!: " << cudaGetErrorString(cudaStatus) << end_log;
		exit(EXIT_FAILURE);
	}

	cudaStatus = cudaMalloc((void**)&dev_labeling, sizeof(unsigned long int)* this->ndspl);
	if (cudaStatus != cudaSuccess) 
	{
		bgn_log << LogType::ERROR << "MinMarginsCUDA: cudaMalloc failed!: " << cudaGetErrorString(cudaStatus) << end_log;
		exit(EXIT_FAILURE);
	}

	cudaStatus = cudaMalloc((void**)&dev_child2parent, sizeof(long int)* this->nlabels);
	if (cudaStatus != cudaSuccess)
	{
		bgn_log << LogType::ERROR << "MinMarginsCUDA: cudaMalloc failed!: " << cudaGetErrorString(cudaStatus) << end_log;
		exit(EXIT_FAILURE);
	}

	cudaStatus = cudaMalloc((void**)&dev_shift2bitid, sizeof(long int)*pow3(2 * this->max_r + 1));
	if (cudaStatus != cudaSuccess)
	{
		bgn_log << LogType::ERROR << "MinMarginsCUDA: cudaMalloc failed!: " << cudaGetErrorString(cudaStatus) << end_log;
		exit(EXIT_FAILURE);
	}


	cudaStatus = cudaMalloc((void**)&dev_marign_edges_64, sizeof(size_t)*this->ndspl*this->n_banks);
	if (cudaStatus != cudaSuccess)
	{
		bgn_log << LogType::ERROR << "MinMarginsCUDA: cudaMalloc failed!: " << cudaGetErrorString(cudaStatus) << end_log;
		exit(EXIT_FAILURE);
	}

	// Copy host vectors to device
	cudaStatus = cudaMemcpyAsync(this->dev_x2ypath, this->x2ypath, sizeof(long int)*this->nlabels*this->nlabels*this->max_path, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) 
	{
		bgn_log << LogType::ERROR << "MinMarginsCUDA: cudaMemcpyAsync failed!: " << cudaGetErrorString(cudaStatus) << end_log;
		exit(EXIT_FAILURE);
	}

	cudaStatus = cudaMemcpyAsync(this->dev_shifts, this->shifts, sizeof(long int)*this->total_shifts_size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		bgn_log << LogType::ERROR << "MinMarginsCUDA: cudaMemcpyAsync failed!: " << cudaGetErrorString(cudaStatus) << end_log;
		exit(EXIT_FAILURE);
	}

	cudaStatus = cudaMemcpyAsync(this->dev_label2shifts, this->label2shifts, sizeof(long int)*(this->nlabels + 1), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		bgn_log << LogType::ERROR << "MinMarginsCUDA: cudaMemcpyAsync failed!: " << cudaGetErrorString(cudaStatus) << end_log;
		exit(EXIT_FAILURE);
	}

	cudaStatus = cudaMemcpyAsync(this->dev_dims, this->dims, sizeof(ssize_t) * 3, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		bgn_log << LogType::ERROR << "MinMarginsCUDA: cudaMemcpyAsync failed!: " << cudaGetErrorString(cudaStatus) << end_log;
		exit(EXIT_FAILURE);
	}


	cudaStatus = cudaMemcpyAsync(this->dev_child2parent, this->child2parent, sizeof(long int)* this->nlabels, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		bgn_log << LogType::ERROR << "MinMarginsCUDA: cudaMemcpyAsync failed!: " << cudaGetErrorString(cudaStatus) << end_log;
		exit(EXIT_FAILURE);
	}

	cudaStatus = cudaMemcpyAsync(this->dev_shift2bitid, this->shift2bitid, sizeof(long int)* pow3(2 * this->max_r + 1), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		bgn_log << LogType::ERROR << "MinMarginsCUDA: cudaMemcpyAsync failed!: " << cudaGetErrorString(cudaStatus) << end_log;
		exit(EXIT_FAILURE);
	}

}

void MarginEdgesCuda::runKernel(unsigned long int layerid, unsigned long int explbl, unsigned long int * labeling)
{
	char input;
	cudaError_t cudaStatus;
	//reset device output memeory
	cudaStatus = cudaMemset(dev_marign_edges_64, 0, sizeof(size_t)*this->ndspl*this->n_banks);
	if (cudaStatus != cudaSuccess) 
	{ 
		bgn_log << LogType::ERROR << "MinMarginsCUDA: cudaMemset failed!: " << cudaGetErrorString(cudaStatus) << end_log;
		exit(EXIT_FAILURE); 
	}
	
	cudaStatus = cudaMemcpyAsync(this->dev_labeling, labeling, sizeof(unsigned long int)* this->ndspl, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) 
	{ 
		bgn_log << LogType::ERROR << "MinMarginsCUDA: cudaMemcpyAsync failed!: " << cudaGetErrorString(cudaStatus) << end_log; 
		exit(EXIT_FAILURE); 
	}
	
	//// Execute the kernel
	//if (this->lessthan35mode)
		setMinMarginArcsKernel << < 256, 64 >> > (((unsigned __int32*)dev_marign_edges_64), ndspl, nlabels, explbl, layerid, max_path, max_r, n_banks * 2
			, dev_dims, dev_labeling, dev_x2ypath, dev_label2shifts
			, dev_shifts, dev_child2parent, dev_shift2bitid);
	//else
	//	setMinMarginArcsKernel << < 256, 64 >> > (dev_marign_edges_64, ndspl, nlabels, explbl, layerid, max_path, max_r, n_banks
	//	, dev_dims, dev_labeling, dev_x2ypath, dev_label2shifts
	//	, dev_shifts, dev_child2parent, dev_shift2bitid);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) 
	{
		bgn_log << LogType::ERROR << "MinMarginsCUDA: setMinMarginArcsKernel lunch failed!: " << cudaGetErrorString(cudaStatus) << end_log;
		std::cin >> input;
		exit(EXIT_FAILURE);
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		bgn_log << LogType::ERROR << "MinMarginsCUDA: cudaDeviceSynchronize failed!: " << cudaGetErrorString(cudaStatus) << end_log;
		exit(EXIT_FAILURE);
	}

	//// Copy result back to host
	cudaStatus = cudaMemcpy(this->margin_edges_64, this->dev_marign_edges_64, sizeof(size_t)*this->ndspl*this->n_banks, cudaMemcpyDeviceToHost);

	if (cudaStatus != cudaSuccess) 
	{ 
		bgn_log << LogType::ERROR<< "MinMarginsCUDA: cudaMemcpy failed!: "<< cudaGetErrorString(cudaStatus) << end_log;
		exit(EXIT_FAILURE); 
	}	
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) 
	{
		bgn_log << LogType::ERROR << "MinMarginsCUDA: cudaDeviceSynchronize failed!: " << cudaGetErrorString(cudaStatus) << end_log;
		exit(EXIT_FAILURE);
	}
}
MarginEdgesCuda::~MarginEdgesCuda()
{
	//std::cout << "Now we are freeing the host and device memory\n";
	CLNDEL1D(shifts);
	CLNDEL1D(label2shifts); 
	CLNDEL1D(shift2bitid); 
	if (dev_dims)                { cudaFree(dev_dims);                dev_dims                = nullptr; }
	if (dev_x2ypath)             { cudaFree(dev_x2ypath);             dev_x2ypath             = nullptr; }
	if (dev_shifts)              { cudaFree(dev_shifts);              dev_shifts              = nullptr; }
	if (dev_label2shifts)        { cudaFree(dev_label2shifts);        dev_label2shifts        = nullptr; }
	if (dev_child2parent)        { cudaFree(dev_child2parent);        dev_child2parent        = nullptr; }
	if (dev_marign_edges_64)     { cudaFree(dev_marign_edges_64);     dev_marign_edges_64     = nullptr; }
	

	//cudaError_t cudaStatus = cudaDeviceReset();
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaDeviceReset failed!");
	//	exit(EXIT_FAILURE);
	//}

}
