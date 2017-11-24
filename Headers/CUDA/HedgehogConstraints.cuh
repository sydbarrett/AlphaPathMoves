/*
This software approximates Hedgehog Shape prior into pairwise edge constraints.
This is a core part of AlphaPathMoves. The software is provied "AS IS" without any warranty,
please see disclaimer below.

##################################################################

License & disclaimer.

Copyright Hossam Isack 	<isack.hossam@gmal.com>

This software and its modifications can be used and distributed for
research purposes only.Publications resulting from use of this code
must cite publications according to the rules given above.Only
Hossam Isack has the right to redistribute this code, unless expressed
permission is given otherwise.Commercial use of this code, any of
its parts, or its modifications is not permited.The copyright notices
must not be removed in case of any modifications.This Licence
commences on the date it is electronically or physically delivered
to you and continues in effect unless you fail to comply with any of
the terms of the License and fail to cure such breach within 30 days
of becoming aware of the breach, in which case the Licence automatically
terminates.This Licence is governed by the laws of Canada and all
disputes arising from or relating to this Licence must be brought
in Toronto, Ontario.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED.IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES(INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

##################################################################
*/
#ifndef __HEADGEHOGCONSTRAINTS_CUH__
#define __HEADGEHOGCONSTRAINTS_CUH__
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_constants.h"
#include "device_functions.h"
#include <iostream>
#include <stdio.h>
#include <cstdint>
#include "Utilities.h"
using namespace Utilities;

template<class VFType, class ShiftType> Array2D<char> * getHedgehogConstraints(NDField<VFType> *, Array2D<ShiftType> *, uint32_t, double);

#endif