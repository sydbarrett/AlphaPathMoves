#include "HedgehogConstraints.cuh"



template<class VFType, class ShiftType>
__global__ void HedgehogKernel_shrdmem(uint32_t * constraints, const VFType *__restrict__ vectorField
	, uint64_t inX, uint64_t inY, uint64_t inZ
	, ShiftType *in_shifts, double *in_nrm_shifts, uint32_t inRadius, uint64_t in_n_shifts, uint64_t inconstriant_Nhs, double theta)
{
	extern __shared__ int32_t shared_mem[];
	__shared__ double * nrm_shifts;
	__shared__ ShiftType * shifts;
	double gamma = 90 - theta;
	uint64_t n_voxels = inX*inY*inZ, n_shifts = in_n_shifts, X = inX, Y = inY, Z = inZ, XY = X*Y;
	uint64_t r = inRadius, twoR1 = 2 * r + 1, twoR1sq = twoR1*twoR1, q, constriant_Nhs = inconstriant_Nhs;
	int64_t qx, qy, qz, px, py, pz;
	int64_t coutZ = inZ == 1 ? 0 : 1, nIdx, GIndx;
	double delta = 0;
	
	if (threadIdx.x == 0)
	{
		shifts = (ShiftType*) & shared_mem[0];
		memcpy(shifts, in_shifts, sizeof(int32_t)*n_shifts * 3);
		nrm_shifts = (double *)& shared_mem[n_shifts * 3];
		memcpy(nrm_shifts, in_nrm_shifts, sizeof(double)*n_shifts * 3);
	}
	__syncthreads();
	double vx, vy, vz;
	
	for (uint64_t p = threadIdx.x + blockIdx.x*blockDim.x; p < n_voxels; p += blockDim.x*gridDim.x)
	{
		pz = p / XY;
		py = (p - pz*XY) / X;
		px = p - py*X - pz*XY;

		vx = vectorField[p * 3 + 0];
		vy = vectorField[p * 3 + 1];
		vz = vectorField[p * 3 + 2];
		for (uint32_t s = 0; s < n_shifts; ++s)
		{
			delta = acos(nrm_shifts[s * 3 + 0] * vx + nrm_shifts[s * 3 + 1] * vy + nrm_shifts[s * 3 + 2] * vz) / CUDART_PI_F*180;
			nIdx = shifts[s * 3 + 0] +r + (shifts[s * 3 + 1] + r)*twoR1 + (shifts[s * 3 + 2] + r)*twoR1sq*coutZ;
			if (delta <= gamma)
			{
				GIndx = p*constriant_Nhs + nIdx;
				atomicOr( (uint32_t*)(constraints + (GIndx >> 2)), 1 << ((GIndx % 4) * 8));  //polar cone
				qx = px - shifts[s * 3 + 0]; if (qx < 0 || qx >= X) continue; //might be able to ignore that condition to avoid diverging threads
				qy = py - shifts[s * 3 + 1]; if (qy < 0 || qy >= Y) continue; //might be able to ignore that condition to avoid diverging threads
				qz = pz - shifts[s * 3 + 2]; if (qz < 0 || qz >= Z) continue; //might be able to ignore that condition to avoid diverging threads
				q = qx + qy*X + qz*XY;
				GIndx =  q*constriant_Nhs + nIdx;
				atomicOr((uint32_t *) (constraints + (GIndx >> 2)), 1 << ((GIndx % 4) * 8));  //dual-cone of the-polar cone
			}
		}
	}
}
template<class VFType, class ShiftType>
__global__ void HedgehogKernel_gblm(uint32_t * constraints, const VFType *__restrict__ vectorField
	, uint64_t inX, uint64_t inY, uint64_t inZ
	, const ShiftType * __restrict__ shifts, const double *__restrict__ nrm_shifts, uint32_t inRadius, uint64_t in_n_shifts, uint64_t inconstriant_Nhs, double theta)
{
	double gamma = 90 - theta;
	uint64_t n_voxels = inX*inY*inZ, n_shifts = in_n_shifts, X = inX, Y = inY, Z = inZ, XY = X*Y;
	uint64_t r = inRadius, twoR1 = 2 * r + 1, twoR1sq = twoR1*twoR1, q, constriant_Nhs = inconstriant_Nhs;
	int64_t qx, qy, qz, px, py, pz;
	int64_t coutZ = inZ == 1 ? 0 : 1, GIndx;
	double delta = 0 ;
	int32_t  nIdx;
	double vx, vy, vz;

	for (uint64_t p = threadIdx.x + blockIdx.x*blockDim.x; p < n_voxels; p += blockDim.x*gridDim.x)
	{
		pz = p / XY;
		py = (p - pz*XY) / X;
		px = p - py*X - pz*XY;
		vx = vectorField[p * 3 + 0];
		vy = vectorField[p * 3 + 1];
		vz = vectorField[p * 3 + 2];
		for (uint32_t s = 0; s < n_shifts; ++s)
		{
			delta = acos(nrm_shifts[s * 3 + 0] * vx + nrm_shifts[s * 3 + 1] * vy + nrm_shifts[s * 3 + 2] * vz) / CUDART_PI_F * 180;
			nIdx = shifts[s * 3 + 0] + r + (shifts[s * 3 + 1] + r)*twoR1 + (shifts[s * 3 + 2] + r)*twoR1sq*coutZ;
			if (delta <= gamma)
			{
				GIndx =  p*constriant_Nhs + nIdx;
				atomicOr((uint32_t *)(constraints + (GIndx >> 2)), 1 << ((GIndx % 4) * 8));  //polar cone
				qx = px - shifts[s * 3 + 0]; if (qx < 0 || qx >= X) continue; //might be able to ignore that condition to avoid diverging threads
				qy = py - shifts[s * 3 + 1]; if (qy < 0 || qy >= Y) continue; //might be able to ignore that condition to avoid diverging threads
				qz = pz - shifts[s * 3 + 2]; if (qz < 0 || qz >= Z) continue; //might be able to ignore that condition to avoid diverging threads
				q = qx + qy*X + qz*XY;
				GIndx = q*constriant_Nhs + nIdx;
				atomicOr((uint32_t *)(constraints + (GIndx >> 2)), 1 << ((GIndx % 4) * 8));  //dual-cone of the-polar cone
			}
		}
	}
}
template<class VFType, class ShiftType>
Array2D<char> * getHedgehogConstraints(NDField<VFType> * vectorField, Array2D<ShiftType> * hhogshifts, uint32_t hhog_radius, double theta)
{
	double mag;
	VFType * dev_vectorField = nullptr;
	uint32_t * dev_constraints = nullptr;
	ShiftType * dev_hhogshifts = nullptr;
	double * dev_nrm_hhogshifts = nullptr;
	uint64_t nVoxels = vectorField->X*vectorField->Y*vectorField->Z, nNhs = hhogshifts->Y;
	uint64_t constriant_Nhs = pow2(2 * hhog_radius + 1);
	if (vectorField->Z > 1)
		constriant_Nhs *= 2 * hhog_radius + 1;
	Array2D<char> * constraintArray = new Array2D<char>(); constraintArray->allocate(constriant_Nhs, nVoxels);

	Array2D<double> nrm_hhogshifts; nrm_hhogshifts.allocate(hhogshifts->X, hhogshifts->Y);
	for (auto s = 0; s < hhogshifts->Y; ++s)
	{
		mag = sqrt(pow2(hhogshifts->data[3 * s + 0]) + pow2(hhogshifts->data[3 * s + 1]) + pow2(hhogshifts->data[3 * s + 2]));
		nrm_hhogshifts.data[3 * s + 0] = hhogshifts->data[3 * s + 0] / mag;
		nrm_hhogshifts.data[3 * s + 1] = hhogshifts->data[3 * s + 1] / mag;
		nrm_hhogshifts.data[3 * s + 2] = hhogshifts->data[3 * s + 2] / mag;
	}
	
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "Hedgehog: cudaSetDevice failed!"); exit(EXIT_FAILURE); }

	cudaStatus = cudaMalloc((void**)&dev_vectorField, sizeof(VFType)*vectorField->totalsize);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "Hedgehog: cudaMalloc failed!"); exit(EXIT_FAILURE); }
	cudaStatus = cudaMalloc((void**)&dev_hhogshifts, sizeof(ShiftType)*hhogshifts->totalsize);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "Hedgehog: cudaMalloc failed!"); exit(EXIT_FAILURE); }
	cudaStatus = cudaMalloc((void**)&dev_nrm_hhogshifts, sizeof(double)*nrm_hhogshifts.totalsize);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "Hedgehog: cudaMalloc failed!"); exit(EXIT_FAILURE); }
	
	
	uint64_t dev_constraint_sizeinInts = nVoxels*constriant_Nhs/4;
	if ((nVoxels*constriant_Nhs) % 4 != 0)
		dev_constraint_sizeinInts++;
	cudaStatus = cudaMalloc((void**)&dev_constraints, sizeof(uint32_t)*dev_constraint_sizeinInts);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "Hedgehog: cudaMalloc failed!"); exit(EXIT_FAILURE); }
	cudaStatus = cudaMemset(dev_constraints, 0, sizeof(uint32_t)*dev_constraint_sizeinInts);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "Hedgehog: cudaMemset failed!"); exit(EXIT_FAILURE); }


	cudaStatus = cudaMemcpyAsync(dev_vectorField, vectorField->field, sizeof(VFType)*vectorField->totalsize, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "Hedgehog: cudaMemcpy failed!"); exit(EXIT_FAILURE); }
	cudaStatus = cudaMemcpyAsync(dev_hhogshifts, hhogshifts->data, sizeof(ShiftType)*hhogshifts->totalsize, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "Hedgehog: cudaMemcpy failed!"); exit(EXIT_FAILURE); }
	cudaStatus = cudaMemcpyAsync(dev_nrm_hhogshifts, nrm_hhogshifts.data, sizeof(double)*nrm_hhogshifts.totalsize, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "Hedgehog: cudaMemcpy failed!"); exit(EXIT_FAILURE); }

	int32_t nBlocks = 32;
	int32_t nThreads = 128;
	uint64_t sharedmem_perblock = (sizeof(double)*hhogshifts->totalsize + sizeof(int32_t)*hhogshifts->totalsize);
	uint64_t total_sharedmem = nBlocks*sharedmem_perblock;
	if (total_sharedmem <= 46*1024 )
		HedgehogKernel_shrdmem << <nBlocks, nThreads, (uint32_t)(sharedmem_perblock) >> >(dev_constraints, dev_vectorField, vectorField->X, vectorField->Y, vectorField->Z, dev_hhogshifts, dev_nrm_hhogshifts, hhog_radius, nNhs, constriant_Nhs, theta);
	else 
		HedgehogKernel_gblm << <nBlocks, nThreads >> >(dev_constraints, dev_vectorField, vectorField->X, vectorField->Y, vectorField->Z, dev_hhogshifts, dev_nrm_hhogshifts, hhog_radius, nNhs, constriant_Nhs, theta);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Hedgehog: Kernel failed error code %d after launching Kernel!\n", cudaStatus);
		std::cout << cudaGetErrorString(cudaStatus) << std::endl;
		exit(EXIT_FAILURE);
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Hedgehog: cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
		std::cout << cudaGetErrorString(cudaStatus) << std::endl;
		exit(EXIT_FAILURE);
	}
	cudaStatus = cudaMemcpy(constraintArray->data, dev_constraints, sizeof(char)*nVoxels * constriant_Nhs, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Hedgehog: output cudaMemcpy failed!\n");
		std::cout << cudaGetErrorString(cudaStatus)<<std::endl;
		exit(EXIT_FAILURE);
	}
	if (dev_vectorField)     { cudaFree(dev_vectorField);      dev_vectorField = nullptr; }
	if (dev_hhogshifts)      { cudaFree(dev_hhogshifts);       dev_hhogshifts = nullptr; }
	if (dev_nrm_hhogshifts)  { cudaFree(dev_nrm_hhogshifts);   dev_nrm_hhogshifts = nullptr; }
	if (dev_constraints)     { cudaFree(dev_constraints);      dev_constraints = nullptr; }

	return constraintArray;
}

template Array2D<char> * getHedgehogConstraints(NDField<float> *, Array2D<int32_t> *, uint32_t, double);