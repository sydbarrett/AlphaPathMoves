/*
This software generates neighbourhood systems for 2D images and 3D Volumes.
This and is provied "AS IS" without any warranty, please see disclaimer below.

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
#ifndef __GRIDNEIGHBOURGENERATOR_H__
#define __GRIDNEIGHBOURGENERATOR_H__
#include<vector>
#include<exception>
#include <cstdint>
#include "Utilities.h"
#define LOCAL_TOLERANCE 0.00000000001
using namespace Utilities;
class GirdNeighboursGenerator
{
public:
	GirdNeighboursGenerator(void) {};
	~GirdNeighboursGenerator(void) {};
	static Array2D<int32_t> * getShifts(uint32_t dims, int32_t size, bool nonsymmetric, bool dense, int32_t max_distance, uint32_t & nNegihbours, double * out_shifts_mag)
	{
	/*	dims:= number of dimensions
		size:= neighbourhood system window size (must be odd). 
		nonsymmetric := If nonsymmetric is true then only 4 shift vectors will be created for a 2D, 3x3 pixels Neighboorhood window size
			and if false then 8 shift  (each one of the 4 shift vectors of the ture will be reveresed and added) vectors will  generated.
			The generated shift vectors are relative to the center pixel, e.g. is {[-1,0],[1,0],[0,-1],[0,1],[1,1],[-1,1],[-1,-1],[1,-1]} a basic symmertic 2D-3x3 Neighbourhood system
		dense := false for each possible oriention only the smallest shift will be returned.
			For example, in 2D5x5 [0,1],[0,2] (althought same they have the same direction only [0,1] will be returned when dense is false).
		max_distance := an optional constraint when used only neighbours in the given neighbourhood at distance <= max_distance (from center pixel) will be returned. 
			Also, max_distance=-1 implies no constraint 
	*/
		if (size%2!=1)
			throw std::exception("Neighbourhood window size must be odd\n");
		std::vector<std::vector<int32_t>> shifts;
		std::vector<int32_t> shift;
		std::vector<double> shifts_mag;
		double shift_mag;
		int32_t size_half=size/2, z_size=size_half;
		if (dims==2)
			z_size=0;

		double cosangle=0;
		bool processed=false;
		//It's enough to process all vectors in the positive quadrant and add the flipped vectors along the y and z planes according to the required density.
		//here we proccessed all of them for simplicty, this funciton runs only once and the neighbhood system is not expected to be large.
		for (int32_t x=-size_half; x<= size_half;++x)
			for (int32_t y=-size_half; y<=size_half;++y)
				for (int32_t z=-z_size; z<= z_size;++z)
				{
					if (x==0 && y==0 && z==0)
						continue;
					shift.clear(); shift.push_back(x); shift.push_back(y); shift.push_back(z);
					shift_mag=std::sqrt((double)(x*x+y*y+z*z));
					if (dense) {
						if (max_distance==-1 || shift_mag<=max_distance)
						{
							shifts.push_back(shift);
							shifts_mag.push_back(shift_mag);
						}
						continue;
					}
					processed=false;
					for (int32_t v=0; v < shifts.size();++v)
					{
						cosangle=(shifts[v][0]*x+shifts[v][1]*y+shifts[v][2]*z)/shifts_mag[v]/shift_mag;
						if (std::abs(cosangle-1)<= LOCAL_TOLERANCE ||std::abs(cosangle+1)<= LOCAL_TOLERANCE){ //same direction
							processed=true;
							if ( (max_distance==-1 || shift_mag<=max_distance) && (shift_mag<shifts_mag[v]) )
							{
								shifts_mag[v]=shift_mag;
								shifts[v]=shift;
							}
							break;
						}
					}
					if (!processed)
					{
						shifts.push_back(shift);
						shifts_mag.push_back(shift_mag);
					}
				}
		if (!nonsymmetric && !dense)
		{
			int32_t curret_size= (int32_t) shifts.size();
			for ( int32_t i = 0; i< curret_size;++i)
			{
				shift.clear(); shift.push_back(-shifts[i][0]); shift.push_back(-shifts[i][1]); shift.push_back(-shifts[i][2]);
				shifts.push_back(shift);
				shifts_mag.push_back(shifts_mag[i]);
			}
		}
		Array2D<int32_t> * outshifts = new Array2D<int32_t>();
		nNegihbours=(uint32_t) shifts.size();
		if (nNegihbours==0)
		{
			outshifts->allocate(dims,0);
			CLNDEL1D(out_shifts_mag);
			out_shifts_mag=nullptr;
			return outshifts;
		}
		outshifts->allocate(dims,(int32_t)shifts.size());
		if (out_shifts_mag!= nullptr)
			delete[] out_shifts_mag;
		out_shifts_mag = new double [shifts_mag.size()];
		for ( int32_t i = 0; i< shifts.size();++i)
		{
			if (out_shifts_mag!=nullptr) 
				out_shifts_mag[i]=shifts_mag[i];
			outshifts->data[i*outshifts->constX]= shifts[i][0];
			outshifts->data[i*outshifts->constX+1]= shifts[i][1];
			if(dims==3) 
				outshifts->data[i*outshifts->constX+2]= shifts[i][2];			
		}
		return outshifts;
	}
};
#endif