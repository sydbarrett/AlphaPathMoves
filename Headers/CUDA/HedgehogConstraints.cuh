#ifndef __HEADGEHOGCONSTRAINTS_CUH__
#define __HEADGEHOGCONSTRAINTS_CUH__
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_constants.h"
#include "device_functions.h"
#include <iostream>
#include <stdio.h>
#include "Utilities.h"
using namespace Utilities;

template<class VFType, class ShiftType> Array2D<char> * getHedgehogConstraints(NDField<VFType> *, Array2D<ShiftType> *, unsigned long int, double);

#endif