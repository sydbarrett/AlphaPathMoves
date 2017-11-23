#ifndef __EUCLIDEANDT_H__
#define __EUCLIDEANDT_H__

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <cmath>
#include <cfloat>
#include <algorithm>
#include "Utilities.h"

#define square(q) ((q)*(q))
#define INF FLT_MAX

using namespace Utilities;
using namespace std;
namespace DistanceTransform
{
	const __int16 INSIDE = 1;
	const __int16 OUTSIDE = 2;
	const __int16 SQUARED = 4;

	template <typename T>
	void internal_DT1D(T *f, size_t n, T * d)
	{
		size_t *v = new size_t[n];
		size_t  k = 0;
		T *z = new T[n + 1];

		v[0] = 0;
		z[0] = -INF;
		z[1] = +INF;
		for (auto q = 1; q <= n - 1; q++)
		{
			T s = ((f[q] + square(q)) - (f[v[k]] + square(v[k]))) / (2 * q - 2 * v[k]);
			while (s <= z[k])
			{
				k--;
				s = ((f[q] + square(q)) - (f[v[k]] + square(v[k]))) / (2 * q - 2 * v[k]);
			}
			k++;
			v[k] = q;
			z[k] = s;
			z[k + 1] = +INF;
		}

		k = 0;
		for (auto q = 0; q <= n - 1; q++)
		{
			while (z[k + 1] < q) k++;
			d[q] = square(q - v[k]) + f[v[k]];
		}

		delete[] v;
		delete[] z;
	}

	/* dt of 2d function using squared distance */
	template<typename T>
	void internal_DT2D(Array2D<T> *im) {
		size_t width = im->X;
		size_t height = im->Y;
		T *f = new T[max(width, height)];
		T *d = new T[max(width, height)];

		// transform along columns
		for (auto x = 0; x < width; x++)  {
			for (auto y = 0; y < height; y++)  f[y] = im->data[x + im->constX*y];
			internal_DT1D(f, height, d);
			for (auto y = 0; y < height; y++)  im->data[x + im->constX*y] = d[y];
		}

		// transform along rows
		for (auto y = 0; y < height; y++) {
			for (auto x = 0; x < width; x++)  f[x] = im->data[x + im->constX*y];
			internal_DT1D(f, width, d);
			for (auto x = 0; x < width; x++)  im->data[x + im->constX*y] = d[x];
		}
		delete[] f;
		delete[] d;
	}
	template<typename T>
	void internal_DT3D(Array3D<T> *im) 
	{
		size_t width = im->X;
		size_t height = im->Y;
		size_t depth = im->Z;
		T *f = new T[max(depth, max(width, height))];
		T *d = new T[max(depth, max(width, height))];


		for (auto z = 0; z < depth; z++)
			for (auto x = 0; x < width; x++)
			{
				for (auto y = 0; y < height; y++)  f[y] = im->data[x + im->constX*y + im->constXY*z];
				internal_DT1D(f, height, d);
				for (auto y = 0; y < height; y++)  im->data[x + im->constX*y + im->constXY*z] = d[y];
			}

		for (auto z = 0; z < depth; z++)
			for (auto y = 0; y < height; y++)
			{
				for (auto x = 0; x < width; x++)  f[x] = im->data[x + im->constX*y + im->constXY*z];
				internal_DT1D(f, width, d);
				for (auto x = 0; x < width; x++)  im->data[x + im->constX*y + im->constXY*z] = d[x];
			}
		for (auto y = 0; y < height; y++)
			for (auto x = 0; x < width; x++)
			{
				for (auto z = 0; z < depth; z++)   f[z] = im->data[x + im->constX*y + im->constXY*z];
				internal_DT1D(f, depth, d);
				for (auto z = 0; z < depth; z++)   im->data[x + im->constX*y + im->constXY*z] = d[z];
			}
		CLNDEL1D(f);
		CLNDEL1D(d);
	}



	/* add inside  distance map option
	   add outside distance map option
	   add inside and outside distance map option
	   */
	template<typename inT, typename outT>
	Array2D<outT> *EDT(Array2D<inT> *in_im, inT on_values = 1, __int16 options = 2)
	{
		size_t width = in_im->X;
		size_t height = in_im->Y;

		Array2D<outT> *outside = NULL;
		Array2D<outT> *inside = NULL;
		Array2D<outT> *output = NULL;
		if (options & OUTSIDE)
		{
			outside = new Array2D<outT>();
			outside->allocate(width, height);
			for (auto k = 0; k < in_im->totalsize; k++) (in_im->data[k] == on_values) ? outside->data[k] = 0 : outside->data[k] = INF;
			internal_DT2D<outT>(outside);
			for (auto k = 0; k < in_im->totalsize; k++)  outside->data[k] = sqrt(outside->data[k]);
		}
		if (options & INSIDE)
		{
			inside = new Array2D<outT>();
			inside->allocate(width, height);
			for (auto k = 0; k < in_im->totalsize; k++) (in_im->data[k] == on_values) ? inside->data[k] = INF : inside->data[k] = 0;
			internal_DT2D<outT>(inside);
			for (auto k = 0; k < in_im->totalsize; k++)  inside->data[k] = -sqrt(inside->data[k]);

		}
		if ((options & OUTSIDE) && (options & INSIDE))
		{
			output = new Array2D<outT>();
			output->allocate(width, height);
			for (auto k = 0; k < in_im->totalsize; k++)  output->data[k] = inside->data[k] + outside->data[k];
			delete inside;
			delete outside;
		}
		else if (options & INSIDE) {
			output = inside;
		} else {
			output = outside;
		}

		if (options & SQUARED)
		for (auto k = 0; k < in_im->totalsize; k++)  output->data[k] = output->data[k] * output->data[k];
		return output;
	}


	/* add inside  distance map option
	   add outside distance map option
	   add squared distance option (default returns d not d^2)
	   */
	template<typename inT, typename outT>
	Array3D<outT> *EDT(const Array3D<inT> * const in_im, inT on_values = 1, __int16 options = 2)
	{
		size_t width = in_im->X;
		size_t height = in_im->Y;
		size_t depth = in_im->Z;
		Array3D<outT> *outside = NULL;
		Array3D<outT> *inside = NULL;
		Array3D<outT> *output = NULL;


		if (options & OUTSIDE)
		{
			outside = new Array3D<outT>();
			outside->allocate(width, height, depth);
			for (auto k = 0; k < in_im->totalsize; k++) (in_im->data[k] == on_values) ? outside->data[k] = 0 : outside->data[k] = INF;
			internal_DT3D<outT>(outside);
			for (auto k = 0; k < in_im->totalsize; k++)  outside->data[k] = sqrt(outside->data[k]);
		}
		if (options & INSIDE)
		{
			inside = new Array3D<outT>();
			inside->allocate(width, height, depth);
			for (auto k = 0; k < in_im->totalsize; k++) (in_im->data[k] == on_values) ? inside->data[k] = INF : inside->data[k] = 0;
			internal_DT3D<outT>(inside);
			for (auto k = 0; k < in_im->totalsize; k++)  inside->data[k] = -sqrt(inside->data[k]);
		}
		if ((options & OUTSIDE) && (options & INSIDE))
		{
			output = new Array3D<outT>();
			output->allocate(width, height, depth);
			for (auto k = 0; k < in_im->totalsize; k++)  output->data[k] = inside->data[k] + outside->data[k];
			delete inside;
			delete outside;
		} else if (options & INSIDE) {
			output = inside;
		} else {
			output = outside;
		}
		if (options & SQUARED)
			for (auto k = 0; k < in_im->totalsize; k++) output->data[k] = output->data[k] * output->data[k];
		
		return output;
	}
}
#endif