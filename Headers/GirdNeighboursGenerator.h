#ifndef __GRIDNEIGHBOURGENERATOR_H__
#define __GRIDNEIGHBOURGENERATOR_H__
#include<vector>
#include<exception>
#include "Utilities.h"
#define LOCAL_TOLERANCE 0.00000000001
using namespace Utilities;
class GirdNeighboursGenerator
{
public:
	GirdNeighboursGenerator(void) {};
	~GirdNeighboursGenerator(void) {};
	static Array2D<long int> * getShifts(unsigned long int dims, long int size, bool nonsymmetric, bool dense, long int max_distance, unsigned long int & nNegihbours, double * out_shifts_mag)
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
		std::vector<std::vector<long int>> shifts;
		std::vector<long int> shift;
		std::vector<double> shifts_mag;
		double shift_mag;
		long int size_half=size/2, z_size=size_half;
		if (dims==2)
			z_size=0;

		double cosangle=0;
		bool processed=false;
		//It's enough to process all vectors in the positive quadrant and add the flipped vectors along the y and z planes according to the required density.
		//here we proccessed all of them for simplicty, this funciton runs only once and the neighbhood system is not expected to be large.
		for (long int x=-size_half; x<= size_half;++x)
			for (long int y=-size_half; y<=size_half;++y)
				for (long int z=-z_size; z<= z_size;++z)
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
					for (long int v=0; v < shifts.size();++v)
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
			long int curret_size= (long int) shifts.size();
			for ( long int i = 0; i< curret_size;++i)
			{
				shift.clear(); shift.push_back(-shifts[i][0]); shift.push_back(-shifts[i][1]); shift.push_back(-shifts[i][2]);
				shifts.push_back(shift);
				shifts_mag.push_back(shifts_mag[i]);
			}
		}
		Array2D<long int> * outshifts = new Array2D<long int>();
		nNegihbours=(unsigned long int) shifts.size();
		if (nNegihbours==0)
		{
			outshifts->allocate(dims,0);
			CLNDEL1D(out_shifts_mag);
			out_shifts_mag=nullptr;
			return outshifts;
		}
		outshifts->allocate(dims,(long int)shifts.size());
		if (out_shifts_mag!= nullptr)
			delete[] out_shifts_mag;
		out_shifts_mag = new double [shifts_mag.size()];
		for ( long int i = 0; i< shifts.size();++i)
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