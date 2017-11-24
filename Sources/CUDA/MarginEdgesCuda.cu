#include "MarginEdgesCuda.cuh"

template <class BankType>
__global__ void setMinMarginArcsKernel(BankType *margin_edges_b2t
	, int64_t in_ndspl, uint32_t in_nlabels, uint32_t explbl, int32_t layer_id, uint32_t in_max_path, int32_t in_max_r, uint64_t in_n_banks
	, const int64_t * __restrict__ dims
	, const uint32_t * __restrict__ labeling
	, const int32_t * __restrict__ paths
	, const int32_t * __restrict__ label2shifts
	, const int32_t * __restrict__ shifts
	, const int32_t * __restrict__ child2parent
	, const int32_t * __restrict__ shift2bitid)
{
	int64_t q, p = threadIdx.x + blockIdx.x*blockDim.x, ndspl = in_ndspl, constX = in_max_path, constXY = in_nlabels*in_max_path, stride = gridDim.x*blockDim.x;
	int64_t dim0 = dims[0], dim1 = dims[1], dim2 = dims[2], dim01 = dim0*dim1;
	int64_t px, py, pz, qx, qy, qz;
	const int32_t * P, *Q;
	int32_t Pi, in_Pi = layer_id, max_path = in_max_path, dX, dY, dZ, g_shiftid;
	int32_t max_r = in_max_r, b0 = 2 * max_r + 1, b1 = b0*b0;
	int64_t idx, bnk_id, n_banks = in_n_banks;
	int64_t n_bitsperbank = sizeof(BankType) * 8;
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
			for (uint32_t shift_idx = label2shifts[P[Pi]]; shift_idx < label2shifts[P[Pi] + 1]; shift_idx+=3)
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
			for (uint32_t shift_id = label2shifts[P[Pi]]; shift_id < label2shifts[P[Pi] + 1]; shift_id += 3)
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
MarginEdgesCuda::MarginEdgesCuda(int64_t ndspl, int64_t* dims, int32_t * child2parent, uint32_t nlabels
	, Array3D<int32_t> * x2ypath, Array2D<int32_t> ** inclusion_shifts, int32_t ***shift2bitid, int32_t max_r, uint64_t * margin_edges
	, uint64_t n_banks, uint64_t n_usedbits)
{
	//initilize memebers 
	this->ndspl = ndspl;
	this->dims[0] = dims[0];
	this->dims[1] = dims[1];
	this->dims[2] = dims[2];
	this->nlabels = nlabels;
	this->max_r = max_r;
	this->x2ypath = x2ypath->data;
	this->max_path = (int32_t) x2ypath->X;
	this->margin_edges_64 = margin_edges;
	this->child2parent = child2parent;
	
	this->label2shifts = new int32_t[nlabels + 1];
	this->label2shifts[0] = 0;
	for (uint32_t i = 1; i <= nlabels; ++i) 
		this->label2shifts[i] = this->label2shifts[i - 1] + (int32_t)inclusion_shifts[i - 1]->Y*3;
	this->total_shifts_size = this->label2shifts[nlabels];
	this->shifts = new int32_t[this->total_shifts_size];
	for (uint32_t i = 0, g_shiftidx = 0; i < nlabels; ++i)
	{
		if (inclusion_shifts[i]->totalsize == 0)
			continue;
		memcpy(this->shifts + g_shiftidx, inclusion_shifts[i]->data, sizeof(int32_t)*inclusion_shifts[i]->totalsize);
		g_shiftidx += (int32_t) inclusion_shifts[i]->totalsize;
	}

	this->n_banks = n_banks;
	this->n_usedbits = n_usedbits;

	this->shift2bitid = new int32_t[pow3(2*this->max_r+1)];
	for (int32_t z = -max_r, g_shiftid = 0; z <= max_r; ++z)
	for (int32_t y = -max_r; y <= max_r; ++y)
	for (int32_t x = -max_r; x <= max_r; ++x, ++g_shiftid)
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
	cudaStatus = cudaMalloc((void**)&dev_x2ypath, sizeof(int32_t)*this->nlabels*this->nlabels*this->max_path);
	if (cudaStatus != cudaSuccess) 
	{
		bgn_log << LogType::ERROR << "MinMarginsCUDA: cudaMalloc failed!: " << cudaGetErrorString(cudaStatus) << end_log;
		exit(EXIT_FAILURE); 
	}

	cudaStatus = cudaMalloc((void**)&dev_shifts, sizeof(int32_t)*this->total_shifts_size);
	if (cudaStatus != cudaSuccess) 
	{
		bgn_log << LogType::ERROR << "MinMarginsCUDA: cudaMalloc failed!: " << cudaGetErrorString(cudaStatus) << end_log;
		exit(EXIT_FAILURE); 
	}

	cudaStatus = cudaMalloc((void**)&dev_label2shifts, sizeof(int32_t)*(this->nlabels+1));
	if (cudaStatus != cudaSuccess) 
	{ 
		bgn_log << LogType::ERROR << "MinMarginsCUDA: cudaMalloc failed!: " << cudaGetErrorString(cudaStatus) << end_log;
		exit(EXIT_FAILURE); 
	}
	
	cudaStatus = cudaMalloc((void**)&dev_dims, sizeof(int64_t) * 3);
	if (cudaStatus != cudaSuccess) 
	{
		bgn_log << LogType::ERROR << "MinMarginsCUDA: cudaMalloc failed!: " << cudaGetErrorString(cudaStatus) << end_log;
		exit(EXIT_FAILURE);
	}

	cudaStatus = cudaMalloc((void**)&dev_labeling, sizeof(uint32_t)* this->ndspl);
	if (cudaStatus != cudaSuccess) 
	{
		bgn_log << LogType::ERROR << "MinMarginsCUDA: cudaMalloc failed!: " << cudaGetErrorString(cudaStatus) << end_log;
		exit(EXIT_FAILURE);
	}

	cudaStatus = cudaMalloc((void**)&dev_child2parent, sizeof(int32_t)* this->nlabels);
	if (cudaStatus != cudaSuccess)
	{
		bgn_log << LogType::ERROR << "MinMarginsCUDA: cudaMalloc failed!: " << cudaGetErrorString(cudaStatus) << end_log;
		exit(EXIT_FAILURE);
	}

	cudaStatus = cudaMalloc((void**)&dev_shift2bitid, sizeof(int32_t)*pow3(2 * this->max_r + 1));
	if (cudaStatus != cudaSuccess)
	{
		bgn_log << LogType::ERROR << "MinMarginsCUDA: cudaMalloc failed!: " << cudaGetErrorString(cudaStatus) << end_log;
		exit(EXIT_FAILURE);
	}


	cudaStatus = cudaMalloc((void**)&dev_marign_edges_64, sizeof(uint64_t)*this->ndspl*this->n_banks);
	if (cudaStatus != cudaSuccess)
	{
		bgn_log << LogType::ERROR << "MinMarginsCUDA: cudaMalloc failed!: " << cudaGetErrorString(cudaStatus) << end_log;
		exit(EXIT_FAILURE);
	}

	// Copy host vectors to device
	cudaStatus = cudaMemcpyAsync(this->dev_x2ypath, this->x2ypath, sizeof(int32_t)*this->nlabels*this->nlabels*this->max_path, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) 
	{
		bgn_log << LogType::ERROR << "MinMarginsCUDA: cudaMemcpyAsync failed!: " << cudaGetErrorString(cudaStatus) << end_log;
		exit(EXIT_FAILURE);
	}

	cudaStatus = cudaMemcpyAsync(this->dev_shifts, this->shifts, sizeof(int32_t)*this->total_shifts_size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		bgn_log << LogType::ERROR << "MinMarginsCUDA: cudaMemcpyAsync failed!: " << cudaGetErrorString(cudaStatus) << end_log;
		exit(EXIT_FAILURE);
	}

	cudaStatus = cudaMemcpyAsync(this->dev_label2shifts, this->label2shifts, sizeof(int32_t)*(this->nlabels + 1), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		bgn_log << LogType::ERROR << "MinMarginsCUDA: cudaMemcpyAsync failed!: " << cudaGetErrorString(cudaStatus) << end_log;
		exit(EXIT_FAILURE);
	}

	cudaStatus = cudaMemcpyAsync(this->dev_dims, this->dims, sizeof(int64_t) * 3, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		bgn_log << LogType::ERROR << "MinMarginsCUDA: cudaMemcpyAsync failed!: " << cudaGetErrorString(cudaStatus) << end_log;
		exit(EXIT_FAILURE);
	}


	cudaStatus = cudaMemcpyAsync(this->dev_child2parent, this->child2parent, sizeof(int32_t)* this->nlabels, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		bgn_log << LogType::ERROR << "MinMarginsCUDA: cudaMemcpyAsync failed!: " << cudaGetErrorString(cudaStatus) << end_log;
		exit(EXIT_FAILURE);
	}

	cudaStatus = cudaMemcpyAsync(this->dev_shift2bitid, this->shift2bitid, sizeof(int32_t)* pow3(2 * this->max_r + 1), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		bgn_log << LogType::ERROR << "MinMarginsCUDA: cudaMemcpyAsync failed!: " << cudaGetErrorString(cudaStatus) << end_log;
		exit(EXIT_FAILURE);
	}

}

void MarginEdgesCuda::runKernel(uint32_t layerid, uint32_t explbl, uint32_t * labeling)
{
	char input;
	cudaError_t cudaStatus;
	//reset device output memeory
	cudaStatus = cudaMemset(dev_marign_edges_64, 0, sizeof(uint64_t)*this->ndspl*this->n_banks);
	if (cudaStatus != cudaSuccess) 
	{ 
		bgn_log << LogType::ERROR << "MinMarginsCUDA: cudaMemset failed!: " << cudaGetErrorString(cudaStatus) << end_log;
		exit(EXIT_FAILURE); 
	}
	
	cudaStatus = cudaMemcpyAsync(this->dev_labeling, labeling, sizeof(uint32_t)* this->ndspl, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) 
	{ 
		bgn_log << LogType::ERROR << "MinMarginsCUDA: cudaMemcpyAsync failed!: " << cudaGetErrorString(cudaStatus) << end_log; 
		exit(EXIT_FAILURE); 
	}
	
	//// Execute the kernel
	//if (this->lessthan35mode)
		setMinMarginArcsKernel << < 256, 64 >> > (((uint32_t*)dev_marign_edges_64), ndspl, nlabels, explbl, layerid, max_path, max_r, n_banks * 2
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
	cudaStatus = cudaMemcpy(this->margin_edges_64, this->dev_marign_edges_64, sizeof(uint64_t)*this->ndspl*this->n_banks, cudaMemcpyDeviceToHost);

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
