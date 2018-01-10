/*
This software implements basic 2 and 3 dimentional array wrappers that are used by AlphaPathMoves.
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
#ifndef __UTILITIES_H__
#define __UTILITIES_H__
#include<typeinfo> 
#include<typeindex>
#include<iostream>
#include<fstream>
#include<iostream>
#include<string>
#include<sstream>
#include<iterator>
#include<exception>
#include<vector>
#include<BaseTsd.h>
#include <algorithm>
#include <cstdint>
#include"TS_Logger.h"
#include "MatlabUtils.h"

#define pow2(v) ((v)*(v)) 
#define pow3(v) ((v)*(v)*(v)) 
#define CLNDEL0D(X) if ((X)!=nullptr ) { delete (X); (X)=nullptr;}
#define CLNDEL1D(X) if ((X)!=nullptr ) { delete [] (X); (X)=nullptr;}
#define CLNDEL2D(X,K) if ((X)!=nullptr) {\
for (auto __bjklcore = 0; __bjklcore < (K); ++__bjklcore)\
	CLNDEL1D((X)[__bjklcore]); \
	delete[](X); \
	(X) = nullptr; \
}
using int64_t = SSIZE_T;
using namespace std;


namespace Utilities {
	template<class T>
	class ArrayInOut
	{
		//V2
		//n_dims
		//cols rows slices
		//min_value max_value
		//[elements] 0 1 2 3 ....

		//n_dims
		//cols rows slices
		//[elements] 0 1 2 3 ....


		//[B/A]n  //B binary, A ascii and n is the version number
		//[datatype]   //datatype are two bytes carries info about the data type
		//cols rows slices // each  is of sizeof(int64_t), i.e. 8 bytes
		//[elements] 0 1 2 3 .... // each element size will depend on XYZ, also XYZ much agree with the template type

		//suppported datatypes: assumed number of bits
		//sc char                      8bits
		//uc unsigend char             8bits
		//ss int16_t                16bits 
		//us uint16_t       16bits
		//sl int32_t                 32bits  (MicroSoft int)
		//ul uint32_t        32bits  (MicroSoft unsigned int)
		//sx int64_t int            64bits
		//ux unsigend int64_t       64bits
		//fs float                    32bits
		//fd double                   64bits

		T *lcl_array;
		int64_t  *dims, n_dims;  //dims[0] is cols, dims[1] is rows, dims[2] is slices
		int64_t size;
		void local_init(int64_t  * in_dims, int64_t in_n_dims)
		{
			this->n_dims = in_n_dims;
			this->dims = new int64_t[this->n_dims];
			memcpy(this->dims, in_dims, sizeof(int64_t)*this->n_dims);
			this->size = 1;
			for (decltype(n_dims) i = 0; i < this->n_dims; ++i)
				this->size *= this->dims[i];
			this->lcl_array = new T[this->size];
		}
		void load_matlab_file(std::string input_filename)
		{
			int32_t * tmp_dims = nullptr, tmp_n_dims;
			MatlabUtils<T>::Load(input_filename, lcl_array, tmp_dims, tmp_n_dims);
			this->n_dims = tmp_n_dims;
			this->dims = new int64_t[this->n_dims];
			for (int32_t i = 0; i < tmp_n_dims; ++i)
				this->dims[i] = tmp_dims[i];
			if (tmp_dims)
				delete[]tmp_dims;
		}
		void savematlab_file(std::string output_fname)
		{
			int32_t * tmp_dims = nullptr;
			if (n_dims > 0)
				tmp_dims = new int32_t[n_dims];
			else
				throw std::runtime_error("Cannot save a dimension less array");
			for (int32_t i = 0; i < n_dims; ++i)
				tmp_dims[i] = this->dims[i];
			MatlabUtils<T>::Save(output_fname, lcl_array, tmp_dims, n_dims);
			if (tmp_dims)
				delete[]tmp_dims;
		}

		void load_binary_v4(std::string input_filename)
		{
			ifstream input_file(input_filename.c_str(), ios::in | ios::binary);
			char version[4], datatype[4];
			input_file.read(version, sizeof(char) * 3);
			if (version[0] != 'B' && version[1] != '4')
			{
				input_file.close();
				throw std::runtime_error("Parsing failed, old .dat versions are no longer supported");
			}
			input_file.read(datatype, sizeof(char) * 3);
			std::string dtype(datatype, 2);
			std::string data_element_typename = "";
			uint64_t data_element_size_bytes;
			if (dtype == "sc")
			{
				data_element_typename = "signed char";
				data_element_size_bytes = 1;
			}
			else if (dtype == "uc")
			{
				data_element_typename = "uint8_t";
				data_element_size_bytes = 1;
			}
			else if (dtype == "ss")
			{
				data_element_typename = "int16_t";
				data_element_size_bytes = 2;
			}
			else if (dtype == "us")
			{
				data_element_typename = "uint16_t";
				data_element_size_bytes = 2;
			}
			else if (dtype == "sl")
			{
				data_element_typename = "int32_t";
				data_element_size_bytes = 4;
			}
			else if (dtype == "ul")
			{
				data_element_typename = "uint32_t";
				data_element_size_bytes = 4;
			}
			else if (dtype == "sx")
			{
				data_element_typename = "int64_t";
				data_element_size_bytes = 8;
			}
			else if (dtype == "ux")
			{
				data_element_typename = "uint64_t";
				data_element_size_bytes = 8;
			}
			else if (dtype == "fs")
			{
				data_element_typename = "float";
				data_element_size_bytes = 4;
			}
			else if (dtype == "fd")
			{
				data_element_typename = "double";
				data_element_size_bytes = 8;
			}

			if (sizeof(T) != data_element_size_bytes)
			{
				input_file.close();
				bgn_log << LogType::ERROR << "The template type size does not match the saved array of type " << data_element_typename << ".\n" << end_log;
				throw std::exception(("Template type size does not match saved array " + data_element_typename + " size").c_str());
			}
			input_file.read(reinterpret_cast<char *>(&n_dims), std::streamsize(sizeof(decltype(n_dims))));
			this->dims = new int64_t[this->n_dims];
			input_file.read(reinterpret_cast<char *>(this->dims), sizeof(decltype(*dims))*this->n_dims);
			this->size = 1;
			for (decltype(n_dims) i = 0; i < this->n_dims; ++i)
				this->size *= this->dims[i];
			this->lcl_array = new T[this->size];
			input_file.read(reinterpret_cast<char *>(this->lcl_array), sizeof(T)*this->size);
			input_file.close();
		}
		void save_binary_v4(std::string output_fname)
		{
			std::string data_element_type = std::string();
			if (type_index(typeid(char)) == type_index(typeid(T)))
				data_element_type = "sc";
			else if (type_index(typeid(uint8_t)) == type_index(typeid(T)))
				data_element_type = "uc";
			else if (type_index(typeid(int16_t)) == type_index(typeid(T)))
				data_element_type = "ss";
			else if (type_index(typeid(uint16_t)) == type_index(typeid(T)))
				data_element_type = "us";
			else if (type_index(typeid(int32_t)) == type_index(typeid(T)))
				data_element_type = "sl";
			else if (type_index(typeid(uint32_t)) == type_index(typeid(T)))
				data_element_type = "ul";
			else if (type_index(typeid(int64_t)) == type_index(typeid(T)))
				data_element_type = "sx";
			else if (type_index(typeid(uint64_t)) == type_index(typeid(T)))
				data_element_type = "ux";
			else if (type_index(typeid(float)) == type_index(typeid(T)))
				data_element_type = "fs";
			else if (type_index(typeid(double)) == type_index(typeid(T)))
				data_element_type = "fd";
			if (data_element_type.empty())
			{
				bgn_log << LogType::ERROR << "Unsupported template type\n" << end_log;
				throw std::exception("Unsupported template type");
			}

			ofstream output_file(output_fname.c_str(), ios::out | ios::binary);
			output_file.write("B4\n", sizeof(char) * 3);
			output_file.write((data_element_type + "\n").c_str(), sizeof(char) * 3);
			output_file.write(reinterpret_cast<const char*>(&n_dims), std::streamsize(sizeof(decltype(n_dims))));
			output_file.write(reinterpret_cast<const char*>(this->dims), sizeof(decltype(*dims))*this->n_dims);
			output_file.write(reinterpret_cast<const char*>(this->lcl_array), sizeof(T)*this->size);
			output_file.close();
		};
	public:
		explicit ArrayInOut(std::string input_fname)
		{
			this->n_dims = 0;
			this->dims = nullptr;
			this->lcl_array = nullptr;
			ifstream input_file(input_fname.c_str(), ios::in);
			if (!input_file)
			{
				bgn_log << LogType::ERROR << "File " + input_fname + " not found.\n" << end_log;
				throw std::exception("File not found\n");
			}
			input_file.close();
			std::string file_extension = input_fname.substr(input_fname.find_last_of(".") + 1);
			std::transform(file_extension.begin(), file_extension.end(), file_extension.begin(), ::tolower);
			if (file_extension == "dat")
				load_binary_v4(input_fname);
			else if (file_extension == "mat")
				load_matlab_file(input_fname);
			else
				throw std::exception("Unsupported format\n");
		};

		void save(std::string output_fname)
		{
			std::string file_extension = output_fname.substr(output_fname.find_last_of(".") + 1);
			std::transform(file_extension.begin(), file_extension.end(), file_extension.begin(), ::tolower);
			if (file_extension == "dat")
				this->save_binary_v4(output_fname);
			else if (file_extension == "mat")
				savematlab_file(output_fname);
			else
				throw std::exception("This library only supports reading and writing .dat and .mat files\n");
		}

		explicit ArrayInOut(int64_t * in_dims, int64_t in_n_dims)
		{
			this->local_init(in_dims, in_n_dims);
		};
		explicit ArrayInOut(T* in_array, int64_t * in_dims, int64_t in_n_dims)
		{
			this->local_init(in_dims, in_n_dims);
			memcpy(this->lcl_array, in_array, sizeof(T)*this->size);
		};
		explicit ArrayInOut()
		{
			this->n_dims = 0;
			this->dims = nullptr;
			this->lcl_array = nullptr;
		};
		ArrayInOut(const ArrayInOut &) = delete;
		ArrayInOut & operator=(const ArrayInOut &) = delete;
		~ArrayInOut() {
			CLNDEL1D(this->dims);
			CLNDEL1D(this->lcl_array);
		};

		T * clone() {
			if (lcl_array != nullptr && size != 0)
			{
				T * r_array = new T[this->size];
				memcpy(r_array, lcl_array, this->size * sizeof(T)); return r_array;
			}
			return nullptr;
		}
		T * getArray()
		{
			return this->lcl_array;
		}
		int64_t getNDims() { return this->n_dims; }
		int64_t getSize() { return this->size; }
		int64_t getDim(int64_t idx) { return this->dims[idx]; }
		int64_t * getDimsClone()
		{
			int64_t * r_dims = nullptr;
			if (this->dims != nullptr && this->n_dims >0)
			{
				r_dims = new int64_t[this->n_dims];
				memcpy(r_dims, this->dims, sizeof(int64_t)*this->n_dims);
			}
			return r_dims;
		}
		void loseOwnerShip()
		{
			this->n_dims = 0;
			CLNDEL1D(dims);
			this->lcl_array = nullptr;
		}
	};

	template <typename T>
	struct Array2D {
	public:
		T *data;
		int64_t constX, X, Y, totalsize;
		explicit Array2D() : data(nullptr), totalsize(0), X(0), Y(0), constX(0) { }
		explicit Array2D(T * in_array, int64_t X, int64_t Y)
		{
			this->X = X;
			this->Y = Y;
			this->totalsize = X*Y;
			this->data = in_array;
			constX = X;
		}
		explicit Array2D(std::string src_file)
		{
			this->X = 0;
			this->Y = 0;
			this->totalsize = 0;
			this->data = nullptr;
			ArrayInOut<T> in_array(src_file);
			if (in_array.getNDims() != 2)
			{
				bgn_log << LogType::ERROR << "Couldn't load the array as it is not 2D\n" << end_log;
				throw std::exception("Can not parse that Array as 2D array\n");
			}


			this->X = in_array.getDim(0);
			this->Y = in_array.getDim(1);

			this->totalsize = X*Y;
			this->constX = X;
			this->data = in_array.getArray();
			in_array.loseOwnerShip();
		}
		~Array2D() {
			CLNDEL1D(data);
		}
		Array2D(const Array2D &) = delete;
		Array2D & operator=(const Array2D &) = delete;
		void allocate(int64_t X, int64_t Y)
		{
			if (this->X == X && this->Y == Y && totalsize != 0)
				return;
			this->X = X;
			this->Y = Y;
			totalsize = X*Y;
			constX = X;
			CLNDEL1D(data);
			if (totalsize != 0)
				data = new T[totalsize];
		}
		bool IsEqual(const Array2D<T> & b_array)
		{
			if (this->X != b_array.X || this->Y != b_array.Y)
				return false;
			for (decltype(totalsize) idx = 0; idx < this->totalsize; idx++)
				if (this->data[idx] != b_array.data[idx])
					return false;
			return true;
		}
		Array2D * clone()
		{
			Array2D *r_array = new Array2D();
			r_array->X = this->X;
			r_array->Y = this->Y;
			r_array->totalsize = this->totalsize;
			r_array->constX = this->constX;
			r_array->data = nullptr;
			if (totalsize != 0)
			{
				r_array->data = new T[r_array->totalsize];
				memcpy(r_array->data, this->data, sizeof(T)*this->totalsize);
			}
			return r_array;
		}
		void print()
		{
			for (int64_t y = 0; y < Y; ++y)
			{
				for (int64_t x = 0; x < X; ++x)
					std::cout << data[x + y*constX] << ' ';
				std::cout << '\n';
			}
		}
		void saveToFile(std::string filename)
		{
			int64_t dims[2]; dims[0] = X; dims[1] = Y;
			ArrayInOut<T> tmp(this->data, dims, 2);
			tmp.save(filename);
		}
		__forceinline void fill(T val) { std::fill_n(data, totalsize, val); }
		//__forceinline T access(int64_t x, int64_t y) { return data[x + y*constX]; } 
		struct YProxy
		{
			T *data;
			int64_t constX;
			YProxy(int64_t in_constX, T* in_data) : constX(in_constX), data(in_data) {}
			T& operator[](std::uint64_t y) {
				return data[y*constX];
			}
		};
		YProxy operator[](std::uint64_t x) {
			return YProxy(this->constX, this->data + x);
		}

		__forceinline void ScalerMultplication(double scale) { for (uint64_t i = 0; i < this->totalsize; ++i) this->data[i] *= scale; }
		__forceinline void RoundArray() { for (auto i = 0; i < this->totalsize; ++i) this->data[i] = std::round(this->data[i]); }

		template<class outT>
		void convertTo(Array2D<outT> & out_array)
		{
			out_array.allocate(X, Y);
			for (auto k = 0; k < totalsize; k++)
				out_array.data[k] = (outT)data[k];
		}

	};

	template <typename T>
	struct Array3D {


	public:
		T *data;
		int64_t totalsize, constX, constXY, X, Y, Z;
		explicit Array3D() : data(nullptr), totalsize(0), X(0), Y(0), Z(0), constX(0), constXY(0) {}
		explicit Array3D(T * in_array, int64_t X, int64_t Y, int64_t Z)
		{
			this->X = X;
			this->Y = Y;
			this->Y = Z;
			totalsize = X*Y*Z;
			constX = X;
			constXY = X*Y;
			this->data = in_array;
		}
		explicit Array3D(std::string src_file) {
			this->X = 0;
			this->Y = 0;
			this->Z = 0;
			this->totalsize = 0;
			this->data = nullptr;
			ArrayInOut<T> in_array(src_file);
			this->X = in_array.getDim(0);
			this->Y = in_array.getDim(1);
			if (in_array.getNDims() == 2)
				this->Z = 1;
			else if (in_array.getNDims() == 3)
				this->Z = in_array.getDim(2);
			else
				throw std::exception("Can not parse that Array as 3D array\n");
			this->totalsize = X*Y*Z;
			this->constX = X;
			this->constXY = X*Y;
			this->data = in_array.getArray();
			in_array.loseOwnerShip();
		}
		~Array3D() { CLNDEL1D(data); }
		Array3D(const Array3D &) = delete;
		Array3D & operator=(const Array3D &) = delete;
		void allocate(int64_t X, int64_t Y, int64_t Z)
		{
			if (this->X == X && this->Y == Y && this->Z == Z && totalsize != 0)
				return;
			this->X = X;
			this->Y = Y;
			this->Z = Z;
			totalsize = X*Y*Z;
			constX = X;
			constXY = X*Y;
			CLNDEL1D(data);
			data = new T[totalsize];
		}
		bool IsEqual(const Array3D<T> & b_array)
		{
			if (this->X != b_array.X || this->Y != b_array.Y || this->Z != b_array.Z)
				return false;
			for (decltype(totalsize) idx = 0; idx < this->totalsize; idx++)
				if (this->data[idx] != b_array.data[idx])
					return false;
			return true;
		}

		Array3D * clone()
		{
			Array3D *r_array = new Array3D();
			r_array->X = this->X;
			r_array->Y = this->Y;
			r_array->Z = this->Z;
			r_array->totalsize = this->totalsize;
			r_array->constX = this->constX;
			r_array->constXY = this->constXY;
			r_array->data = nullptr;
			if (totalsize != 0)
			{
				r_array->data = new T[r_array->totalsize];
				memcpy(r_array->data, this->data, sizeof(T)*this->totalsize);
			}
			return r_array;
		}
		void print()
		{
			for (int64_t z = 0; z < Z; ++z)
			{
				for (int64_t y = 0; y < Y; ++y)
				{
					for (int64_t x = 0; x < X; ++x)
						std::cout << data[x + y*constX + z*constXY] << ' ';
					std::cout << '\n';
				}
				std::cout << "\n\n";
			}
		}
		void saveToFile(std::string filename)
		{
			int64_t dims[3]; dims[0] = X; dims[1] = Y, dims[2] = Z;
			ArrayInOut<T> tmp(this->data, dims, 3);
			tmp.save(filename);
		}
		__forceinline void fill(T val) { std::fill_n(data, totalsize, val); }
		//__forceinline T access(int64_t x, int64_t y, int64_t z) { return data[x + y*constX+z*constXY]; }
		struct ZProxy
		{
			T *data;
			int64_t constXY;
			ZProxy(int64_t in_constXY, T* in_data) : constXY(in_constXY), data(in_data) {}
			T& operator[](std::uint64_t z) {
				return data[z*constXY];
			}
		};
		struct YProxy
		{
			T *data;
			int64_t constX;
			int64_t constXY;
			YProxy(int64_t in_constX, int64_t in_constXY, T* in_data) : constX(in_constX), constXY(in_constXY), data(in_data) {}
			ZProxy operator[](std::uint64_t y) {
				return ZProxy(this->constXY, this->data + y*constX);
			}
		};
		YProxy operator[](std::uint64_t x) {
			return YProxy(this->constX, this->constXY, this->data + x);
		}

		__forceinline void ScalerMultplication(double scale) { for (auto i = 0; i < this->totalsize; ++i) this->data[i] *= scale; }
		__forceinline void RoundArray() { for (auto i = 0; i < this->totalsize; ++i) this->data[i] = std::round(this->data[i]); }

		template<class outT>
		void convertTo(Array3D<outT> & out_array)
		{
			out_array.allocate(X, Y, Z);
			for (decltype(totalsize) k = 0; k < totalsize; k++)
				out_array.data[k] = (outT)data[k];
		}
	};

	template<typename T>
	struct NDField
	{
		T *field;
		int64_t c1, c2, c3, X, Y, Z, n_elements, totalsize;
		explicit NDField() : field(nullptr), totalsize(0) {}
		explicit NDField(string fname)
		{
			ifstream input_file(fname.c_str(), ios::in);
			if (!input_file)
				throw std::exception("File not found\n");

			string ver_str;
			input_file >> ver_str;
			if (ver_str.compare("V1") != 0)
				throw exception("Unsupported .dat version");
			input_file >> this->n_elements;
			input_file >> this->X;
			input_file >> this->Y;
			input_file >> this->Z;
			this->totalsize = this->n_elements*this->X*this->Y*this->Z;
			this->field = new T[this->totalsize];

			std::vector<T> lc_parsed;
			lc_parsed.reserve(this->totalsize);
			lc_parsed.assign(std::istream_iterator<T>(input_file), std::istream_iterator<T>());
			if (lc_parsed.empty())
				throw exception("Failed to parse file");
			memcpy(this->field, &(lc_parsed[0]), sizeof(T)*this->totalsize);
			input_file.close();
		}
		~NDField()
		{
			CLNDEL1D(this->field);
		}
		NDField & operator=(const NDField &) = delete;
		NDField(const NDField &) = delete;
		void allocate(int64_t n_elements, int64_t X, int64_t Y, int64_t Z)
		{
			this->X = X;
			this->Y = Y;
			this->Z = Z;
			this->n_elements = n_elements;
			totalsize = X*Y*Z*this->n_elements;
			c1 = this->n_elements;
			c2 = this->n_elements*X;
			c3 = this->n_elements*X*Y;
			CLNDEL1D(field);
			if (totalsize != 0)
				field = new T[totalsize];
		}
		void saveToFile(std::string output_fname)
		{
			ofstream output_file(output_fname.c_str(), ios::out);
			output_file << "VF1\n";
			output_file << this->n_elements << " " << this->X << " " << this->Y << " " << this->Z << "\n";
			for (decltype(totalsize) running_id = 0; running_id < this->totalsize; running_id++)
				output_file << this->field[running_id] << " ";
			output_file.close();
		}
		__forceinline void fill(T val) { std::fill_n(field, totalsize, val); }
		__forceinline T* access(int64_t x = 0, int64_t y = 0, int64_t z = 0) { return field + x*c1 + y*c2 + z*c3; }
	};

}
#endif