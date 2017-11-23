#ifndef __SOBELFILTER_CUH__
#define __SOBELFILTER_CUH__
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <iostream>
#include <stdio.h>
#include "Utilities.h"
using namespace Utilities;


template<class InType, class OutType>
NDField<OutType> * SobelFilter(Array3D<InType> * inVol, bool normalize, bool reverse);


#endif

