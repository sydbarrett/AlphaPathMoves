#include "SobelFilter.cuh"

#define RelP(dx,dy,dz) (p+dx+dy*X+dz*XY)
#define SUB2IDX(x,y,z) ((x+1)+(y+1)*3+(z+1)*9)
#define USETEXMEM false
#define UNDEFMAGTOL 0.0001
texture<float, 2>  texInVol;
__constant__ int hx[27];
__constant__ int hy[27];
__constant__ int hz[27];
int host_hx[27] = { -1, 0, 1, -2, 0, 2, -1, 0, 1, -2, 0, 2, -4, 0, 4, -2, 0, 2, -1, 0, 1, -2, 0, 2, -1, 0, 1 };
int host_hy[27] = { -1, -2, -1, 0, 0, 0, 1, 2, 1, -2, -4, -2, 0, 0, 0, 2, 4, 2, -1, -2, -1, 0, 0, 0, 1, 2, 1 };
int host_hz[27] = { -1, -2, -1, -2, -4, -2, -1, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 2, 4, 2, 1, 2, 1 };

template <class InType, class OutType>
__global__ void sobelFilter3D(const InType * __restrict__ inVol
	, const uint64_t inX, const uint64_t inY, const uint64_t inZ, bool inNormalize, bool inReverse
	, OutType * vfield)

{
	int64_t X = inX, Y = inY, XY = X*Y, Z = inZ, XYZ = X*Y*Z;
	int64_t x, y, z;
	bool normalize = inNormalize, reverse = inReverse;
	double mag, gx, gy, gz;
	for (int64_t p = threadIdx.x + blockIdx.x* blockDim.x; p < XYZ; p += gridDim.x*blockDim.x)
	{

		z = p / XY;
		y = (p - z*XY) / X;
		x = p - y*X - z*XY;  
		gx = 0;
		gy = 0;
		gz = 0;
		if (x != 0 && x != (X - 1) && y != 0 && y != (Y - 1) &&  z != 0 && z != (Z - 1))
		{
			gx =  hx[SUB2IDX(-1,-1,-1)] * inVol[RelP(-1,-1,-1)] + hx[SUB2IDX(0,-1,-1)] * inVol[RelP(0,-1,-1)] + hx[SUB2IDX(1,-1,-1)] * inVol[RelP(1,-1,-1)]
				+ hx[SUB2IDX(-1, 1,-1)] * inVol[RelP(-1, 1,-1)] + hx[SUB2IDX(0, 1,-1)] * inVol[RelP(0, 1,-1)] + hx[SUB2IDX(1, 1,-1)] * inVol[RelP(1, 1,-1)]
				+ hx[SUB2IDX(-1,-1, 0)] * inVol[RelP(-1,-1, 0)] + hx[SUB2IDX(0,-1, 0)] * inVol[RelP(0,-1, 0)] + hx[SUB2IDX(1,-1, 0)] * inVol[RelP(1,-1, 0)]
				+ hx[SUB2IDX(-1, 1, 0)] * inVol[RelP(-1, 1, 0)] + hx[SUB2IDX(0, 1, 0)] * inVol[RelP(0, 1, 0)] + hx[SUB2IDX(1, 1, 0)] * inVol[RelP(1, 1, 0)]
				+ hx[SUB2IDX(-1,-1, 1)] * inVol[RelP(-1,-1, 1)] + hx[SUB2IDX(0,-1, 1)] * inVol[RelP(0,-1, 1)] + hx[SUB2IDX(1,-1, 1)] * inVol[RelP(1,-1, 1)]
				+ hx[SUB2IDX(-1, 1, 1)] * inVol[RelP(-1, 1, 1)] + hx[SUB2IDX(0, 1, 1)] * inVol[RelP(0, 1, 1)] + hx[SUB2IDX(1, 1, 1)] * inVol[RelP(1, 1, 1)];

			gy =  hy[SUB2IDX(-1,-1,-1)] * inVol[RelP(-1,-1,-1)] + hy[SUB2IDX(1,-1,-1)] * inVol[RelP(1,-1,-1)]
				+ hy[SUB2IDX(-1, 0,-1)] * inVol[RelP(-1, 0,-1)] + hy[SUB2IDX(1, 0,-1)] * inVol[RelP(1, 0,-1)]
				+ hy[SUB2IDX(-1, 1,-1)] * inVol[RelP(-1, 1,-1)] + hy[SUB2IDX(1, 1,-1)] * inVol[RelP(1, 1,-1)]
				+ hy[SUB2IDX(-1,-1, 0)] * inVol[RelP(-1,-1, 0)] + hy[SUB2IDX(1,-1, 0)] * inVol[RelP(1,-1, 0)]
				+ hy[SUB2IDX(-1, 0, 0)] * inVol[RelP(-1, 0, 0)] + hy[SUB2IDX(1, 0, 0)] * inVol[RelP(1, 0, 0)]
				+ hy[SUB2IDX(-1, 1, 0)] * inVol[RelP(-1, 1, 0)] + hy[SUB2IDX(1, 1, 0)] * inVol[RelP(1, 1, 0)]
				+ hy[SUB2IDX(-1,-1, 1)] * inVol[RelP(-1,-1, 1)] + hy[SUB2IDX(1,-1, 1)] * inVol[RelP(1,-1, 1)]
				+ hy[SUB2IDX(-1, 0, 1)] * inVol[RelP(-1, 0, 1)] + hy[SUB2IDX(1, 0, 1)] * inVol[RelP(1, 0, 1)]
				+ hy[SUB2IDX(-1, 1, 1)] * inVol[RelP(-1, 1, 1)] + hy[SUB2IDX(1, 1, 1)] * inVol[RelP(1, 1, 1)];

			gz =  hz[SUB2IDX(-1,-1,-1)] * inVol[RelP(-1,-1,-1)] + hz[SUB2IDX(0,-1,-1)] * inVol[RelP(0,-1,-1)] + hz[SUB2IDX(1,-1,-1)] * inVol[RelP(1,-1,-1)]
				+ hz[SUB2IDX(-1, 0,-1)] * inVol[RelP(-1, 0,-1)] + hz[SUB2IDX(0, 0,-1)] * inVol[RelP(0, 0,-1)] + hz[SUB2IDX(1, 0,-1)] * inVol[RelP(1, 0,-1)]
				+ hz[SUB2IDX(-1, 1,-1)] * inVol[RelP(-1, 1,-1)] + hz[SUB2IDX(0, 1,-1)] * inVol[RelP(0, 1,-1)] + hz[SUB2IDX(1, 1,-1)] * inVol[RelP(1, 1,-1)]
				+ hz[SUB2IDX(-1,-1, 1)] * inVol[RelP(-1,-1, 1)] + hz[SUB2IDX(0,-1, 1)] * inVol[RelP(0,-1, 1)] + hz[SUB2IDX(1,-1, 1)] * inVol[RelP(1,-1, 1)]
				+ hz[SUB2IDX(-1, 0, 1)] * inVol[RelP(-1, 0, 1)] + hz[SUB2IDX(0, 0, 1)] * inVol[RelP(0, 0, 1)] + hz[SUB2IDX(1, 0, 1)] * inVol[RelP(1, 0, 1)]
				+ hz[SUB2IDX(-1, 1, 1)] * inVol[RelP(-1, 1, 1)] + hz[SUB2IDX(0, 1, 1)] * inVol[RelP(0, 1, 1)] + hz[SUB2IDX(1, 1, 1)] * inVol[RelP(1, 1, 1)];
			if (normalize)
			{
				mag = sqrt(gx*gx + gy*gy + gz*gz);
				if (mag < UNDEFMAGTOL)
				{
					gx = 0;
					gy = 0;
					gz = 0;
				}
				else
				{
					gx /= mag;
					gy /= mag;
					gz /= mag;
				}
			}
		}
		if (reverse)
		{
			vfield[p * 3 + 0] = -gx;
			vfield[p * 3 + 1] = -gy;
			vfield[p * 3 + 2] = -gz;

		} else {
			vfield[p * 3 + 0] = gx;
			vfield[p * 3 + 1] = gy;
			vfield[p * 3 + 2] = gz;
		}
	}
}
template <class InType, class OutType>
__global__ void sobelFilter2D(const InType * __restrict__ inVol, const uint64_t inX, const uint64_t inY, bool inNormalize, bool inReverse, OutType * vfield)
{
	int64_t X = inX, Y = inY, XY = X*Y;
	int64_t p = blockIdx.x* blockDim.x + threadIdx.x, x, y, stride = gridDim.x*blockDim.x;
	bool normalize = inNormalize, reverse = inReverse;
	double mag, gx, gy;

	while(p < XY)
	{
		y = p / X;
		x = p - y*X;
		gx = 0;
		gy = 0;
		if (x != 0 && x != (X - 1) && y != 0 && y != (Y - 1))
		{
			gx = -inVol[p - X - 1] - 2 * inVol[p - 1] - inVol[p + X - 1] + inVol[p - X + 1] + 2 * inVol[p + 1] + inVol[p + X + 1];
			gy = -inVol[p - X - 1] - 2 * inVol[p - X] - inVol[p - X + 1] + inVol[p + X - 1] + 2 * inVol[p + X] + inVol[p + X + 1];
			if (normalize)
			{
				mag = sqrt(gx*gx + gy*gy);
				if (mag < UNDEFMAGTOL)
				{
					gx = 0;
					gy = 0;
				}
				else
				{
					gx /= mag;
					gy /= mag;
				}
			}
		}
		if (reverse)
		{
			vfield[p * 3 + 0] = -gx;
			vfield[p * 3 + 1] = -gy;
			vfield[p * 3 + 2] = 0;
		}
		else {
			vfield[p * 3 + 0] = gx;
			vfield[p * 3 + 1] = gy;
			vfield[p * 3 + 2] = 0;
		}
		p += stride;
	}
}
template <class InType,class OutType>
__global__ void sobelFilter2DTexture(const InType * __restrict__ inVol, const uint64_t inX, const uint64_t inY, bool inNormalize, bool inReverse, OutType * vfield)
{
	int64_t X = inX, Y = inY, XY = X*Y, x, y, stride = gridDim.x*blockDim.x;
	int64_t p = blockIdx.x* blockDim.x + threadIdx.x;
	bool normalize = inNormalize, reverse = inReverse;
	double mag,gx,gy;
	while (p < XY)
	{
		y = p / X;
		x = p - y*X;
		gx = 0;
		gy = 0;
		if (x != 0 && x != (X - 1) && y != 0 && y != (Y - 1))
		{
			gx = - tex2D(texInVol, x - 1, y - 1) - 2 * tex2D(texInVol, x - 1, y) - tex2D(texInVol, x - 1, y + 1) + tex2D(texInVol, x + 1, y - 1) + 2 * tex2D(texInVol, x + 1, y) + tex2D(texInVol, x + 1, y + 1);
			gy = - tex2D(texInVol, x - 1, y - 1) - 2 * tex2D(texInVol, x, y - 1) - tex2D(texInVol, x + 1, y - 1) + tex2D(texInVol, x - 1, y + 1) + 2 * tex2D(texInVol, x, y + 1) + tex2D(texInVol, x + 1, y + 1);
			if (normalize)
			{
				mag = sqrt(gx*gx + gy*gy);
				if (mag < UNDEFMAGTOL)
				{
					gx = 0;
					gy = 0;
				}
				else
				{
					gx /= mag;
					gy /= mag;
				}
			}
		}
		if (reverse)
		{
			vfield[p * 3 + 0] = -gx;
			vfield[p * 3 + 1] = -gy;
			vfield[p * 3 + 2] = 0;
		}
		else {
			vfield[p * 3 + 0] = gx;
			vfield[p * 3 + 1] = gy;
			vfield[p * 3 + 2] = 0;
		}
		p += stride;
	}
}
template<class InType,class OutType>
NDField<OutType> * SobelFilter(Array3D<InType> * inVol, bool normalize , bool reverse)
{
	InType * dev_vol = NULL;
	OutType * dev_vfield = NULL;
	uint64_t nVoxels = inVol->totalsize;
	NDField<OutType> *vfield = new NDField<OutType>();
	vfield->allocate(3, inVol->X, inVol->Y, inVol->Z);
	


	cudaError_t cudaStatus;
	/*
	int nDevices;
	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
	
		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);
		printf("  Memory Clock Rate (KHz): %d\n",
			prop.memoryClockRate);
		printf("  Memory Bus Width (bits): %d\n",
			prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
			2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8) / 1.0e6);
	}*/

	cudaStatus = cudaSetDevice(0); 
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "sobelFilter: cudaSetDevice failed!"); exit(EXIT_FAILURE); }

	cudaStatus = cudaMalloc((void**)&dev_vol, sizeof(InType)*nVoxels);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "sobelFilter: cudaMalloc failed!"); exit(EXIT_FAILURE); }
	
	cudaStatus = cudaMalloc((void**)&dev_vfield, sizeof(OutType)*nVoxels * 3);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "sobelFilter: cudaMalloc failed!"); exit(EXIT_FAILURE); }



	cudaStatus = cudaMemcpy(dev_vol, inVol->data, sizeof(InType)* nVoxels, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "sobelFilter: cudaMemcpy failed!"); exit(EXIT_FAILURE); }

	cudaStatus = cudaMemcpyToSymbol(hx, host_hx, sizeof(int) * 27);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "sobelFilter: cudaMemcpy failed!"); exit(EXIT_FAILURE); }
	
	cudaStatus = cudaMemcpyToSymbol(hy, host_hy, sizeof(int)* 27);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "sobelFilter: cudaMemcpy failed!"); exit(EXIT_FAILURE); }
	
	cudaStatus = cudaMemcpyToSymbol(hz, host_hz, sizeof(int)* 27);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "sobelFilter: cudaMemcpy failed!"); exit(EXIT_FAILURE); }


	uint32_t nThreads = 128;
	uint32_t nBlocks = 128;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	if (USETEXMEM)
	{
		cudaEventRecord(start);
		cudaChannelFormatDesc desc_InType = cudaCreateChannelDesc<InType>();
		cudaBindTexture2D(NULL, texInVol, dev_vol, desc_InType, inVol->X, inVol->Y, sizeof(InType)* inVol->X);
		sobelFilter2DTexture << <nBlocks, nThreads >> >(dev_vol, inVol->X, inVol->Y, normalize, reverse, dev_vfield);
		cudaEventRecord(stop);
	} else {
		cudaEventRecord(start);
		if (inVol->Z == 1)
			sobelFilter2D << <nBlocks, nThreads >> >(dev_vol, inVol->X, inVol->Y, normalize,reverse, dev_vfield);
		else
			sobelFilter3D << <nBlocks, nThreads >> >(dev_vol, inVol->X, inVol->Y, inVol->Z, normalize, reverse, dev_vfield);
		cudaEventRecord(stop);
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "sobelFilter: cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
		exit(EXIT_FAILURE);
	}
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
		cudaStatus = cudaMemcpy(vfield->field, dev_vfield, sizeof(OutType)*nVoxels * 3, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "sobelFilter: output cudaMemcpy failed!\n");
		std::cout << cudaGetErrorString(cudaStatus);
		exit(EXIT_FAILURE);
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "sobelFilter: cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
		exit(EXIT_FAILURE);
	}
	if (USETEXMEM)
		cudaUnbindTexture(texInVol);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	if (dev_vol)     { cudaFree(dev_vol);      dev_vol = NULL; }
	if (dev_vfield)  { cudaFree(dev_vfield);   dev_vfield = NULL; }

	//cudaStatus = cudaDeviceReset();
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaDeviceReset failed!");
	//	exit(EXIT_FAILURE);
	//}


	return vfield;
}

template NDField<double> * SobelFilter(Array3D<double>* inVol, bool normalize,bool reverse);
template NDField<float> *  SobelFilter(Array3D<float>* inVol, bool normalize, bool reverse);