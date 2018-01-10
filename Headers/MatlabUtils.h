/*
This software implements reads/writes numerical matrcies  in/from Matlab files (.mat) level 5.
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

#ifndef __MATLABUTILS_H_
#define __MATLABUTILS_H_
#include <iostream>
#include <fstream>
#include <typeinfo> 
#include <typeindex>
#include <cstdint>

template <class T >
class MatlabUtils
{
private:
	enum class MatDataType : int32_t {
		miINT8 = 1, miUINT8 = 2, miINT16 = 3, miUINT16 = 4, miINT32 = 5, miUINT32 = 6, \
		miSINGLE = 7, miDOUBLE = 9, miINT64 = 12, miUINT64 = 13, miMATRIX = 14
	};
	enum class ArrayDataType : int32_t {
		mxCELL_CLASS = 1, mxSTRUCT_CLAS = 2, mxOBJECT_CLASS = 3, mxCHAR_CLASS = 4, \
		mxSPARSE_CLASS = 5, mxDOUBLE_CLASS = 6, mxSINGLE_CLASS = 7, mxINT8_CLASS = 8, \
		mxUINT8_CLASS = 9, mxINT16_CLASS = 10, mxUINT16_CLASS = 11, mxINT32_CLASS = 12, \
		mxUINT32_CLASS = 13, mxINT64_CLASS = 14, mxUINT64_CLASS = 15
	};

	static void setMatFileHeader(uint8_t *& header, int32_t & header_size);
	static void setMatrixTag(int32_t array_flags_tag_size, int32_t dims_tag_size, int32_t array_name_tag_size, int32_t data_tag_size, \
		uint8_t * & tag, int32_t & tag_size);
	static void setArrayFlagsTag(T * data, int32_t n_elements, uint8_t* & tag, int32_t & tag_size);
	static void setDimsTag(int32_t * dims, int32_t n_dims, uint8_t * & tag, int32_t & tag_size);
	static void setNameTag(std::string array_name, uint8_t* & tag, int32_t & tag_size);
	static void setDataTag(T * data, int32_t n_elements, uint8_t* & tag, int32_t & tag_size);

	static void readTag(uint8_t * tag, int32_t & tag_type, int32_t & tag_size);

public:

	static bool Save(std::string output_fname, T * in_array, int32_t * dims, int32_t n_dims, std::string mat_name = "no_name");
	static bool Load(std::string output_fname, T *& in_array, int32_t *& dims, int32_t & n_dims);
};

template<class T>
void MatlabUtils<T>::readTag(uint8_t * tag, int32_t & tag_type, int32_t & tag_size)
{
	tag_type = int32_t(tag[0] | tag[1] << 8 | tag[2] << 16 | tag[3] << 24);
	tag_size = int32_t(tag[4] | tag[5] << 8 | tag[6] << 16 | tag[7] << 24);
}

template<class T>
bool MatlabUtils<T>::Load(std::string output_fname, T *& in_array, int32_t *& dims, int32_t & n_dims)
{
	if (in_array)
	{
		delete[] in_array;
		in_array = nullptr;
	}

	if (dims)
	{
		delete[] dims;
		dims = nullptr;
	}

	std::ifstream input_file;
	uint8_t* file_rawbytes = nullptr;
	uint64_t n_bytes = 0;
	try
	{
		input_file.open(output_fname.c_str(), std::ios::in | std::ios::binary);
		input_file.seekg(0, ios::end);
		n_bytes = input_file.tellg();

		file_rawbytes = new uint8_t[n_bytes];
		memset(file_rawbytes, 0, n_bytes);
		input_file.seekg(0, ios::beg);
		input_file.read(reinterpret_cast<char *>(file_rawbytes), n_bytes);
		input_file.close();
	}
	catch (exception e)
	{
		if (file_rawbytes) delete[]file_rawbytes;
		throw std::runtime_error("Could not load .mat file : " + std::string(e.what()));
	}
	if (n_bytes <= 128)
	{
		if (file_rawbytes) delete[]file_rawbytes;
		throw std::runtime_error("Could not parse the .mat file: it can not be less than 128 bytes");
	}
	//check that the first 4 bytes are not zeros
	if (file_rawbytes[0] == 0 && file_rawbytes[1] == 0 && \
		file_rawbytes[2] == 0 && file_rawbytes[3] == 0)
	{
		if (file_rawbytes) delete[]file_rawbytes;
		throw std::runtime_error("This is not a Level 5 .mat file");
	}

	//Ignore the first 125 bytes 
	//validate that  byte[126]=I and byte[127]='M'
	if (file_rawbytes[126] != 'I' && file_rawbytes[127] != 'M')
	{
		if (file_rawbytes) delete[]file_rawbytes;
		throw std::runtime_error("This library only supports little endian systems\n");
	}
	int32_t offset = 128;
	int32_t matrix_tag_type, matrix_tag_size;
	readTag(file_rawbytes + offset, matrix_tag_type, matrix_tag_size);
	offset += 8;
	if (matrix_tag_type != int32_t(MatDataType::miMATRIX))
	{
		if (file_rawbytes) delete[]file_rawbytes;
		throw std::runtime_error("This library only reads matrices of signed/unsigend numerical types");
	}

	int32_t array_flags_tag_type, array_flags_tag_size;
	readTag(file_rawbytes + offset, array_flags_tag_type, array_flags_tag_size);
	char matrix_class = file_rawbytes[offset + 8];
	offset += 16; //ignore the actual flags
	if (array_flags_tag_type != int32_t(MatDataType::miUINT32) && array_flags_tag_size != 8)
	{
		if (file_rawbytes) delete[]file_rawbytes;
		throw std::runtime_error("Could not parse the file as a Level 5 .mat file");
	}
	if (matrix_class == int8_t(ArrayDataType::mxCELL_CLASS) ||
		matrix_class == int8_t(ArrayDataType::mxSTRUCT_CLAS) ||
		matrix_class == int8_t(ArrayDataType::mxOBJECT_CLASS) ||
		matrix_class == int8_t(ArrayDataType::mxCHAR_CLASS) ||
		matrix_class == int8_t(ArrayDataType::mxSPARSE_CLASS))
	{
		if (file_rawbytes) delete[]file_rawbytes;
		throw std::runtime_error("This library only reads matrices of signed/unsigend numerical types");
	}


	int32_t dims_tag_type, dims_tag_size;
	readTag(file_rawbytes + offset, dims_tag_type, dims_tag_size);
	offset += 8;
	if (dims_tag_type != int32_t(MatDataType::miINT32))
	{
		if (file_rawbytes) delete[]file_rawbytes;
		throw std::runtime_error("Could not parse the file as a Level 5 .mat file");
	}
	n_dims = dims_tag_size / sizeof(int32_t);
	int32_t dims_zero_padding = dims_tag_size % 8;
	if (dims_zero_padding != 0)
		dims_zero_padding = 8 - dims_zero_padding;
	dims = new int32_t[n_dims];
	memcpy(dims, file_rawbytes + offset, dims_tag_size);
	offset += dims_tag_size;
	offset += dims_zero_padding;

	int32_t name_tag_type, name_tag_size;
	readTag(file_rawbytes + offset, name_tag_type, name_tag_size);
	offset += 8;
	if (name_tag_type != int32_t(MatDataType::miINT8) || name_tag_size <= 4 || name_tag_size > 8)
	{
		if (file_rawbytes) delete[]file_rawbytes;
		if (dims) delete[]dims;
		throw std::runtime_error("Could not parse the file as a Level 5 .mat file. Also, the matrix name should be between 5 and 8 letters, e.g name it no_name.\nSave your array A in Matlab as follows\n\tno_name=A;\n\tsave('filename.mat','no_name','-v6');.\n");
	}
	int32_t name_zero_padding = name_tag_size % 8;
	if (name_zero_padding != 0)
		name_zero_padding = 8 - name_zero_padding;
	offset += name_tag_size;
	offset += name_zero_padding;

	int32_t data_tag_type, data_tag_size;
	readTag(file_rawbytes + offset, data_tag_type, data_tag_size);
	offset += 8;

	if ((data_tag_type == int32_t(MatDataType::miINT8) && type_index(typeid(int8_t)) != type_index(typeid(T))) ||
		(data_tag_type == int32_t(MatDataType::miUINT8) && type_index(typeid(uint8_t)) != type_index(typeid(T))) ||
		(data_tag_type == int32_t(MatDataType::miINT16) && type_index(typeid(int16_t)) != type_index(typeid(T))) ||
		(data_tag_type == int32_t(MatDataType::miUINT16) && type_index(typeid(uint16_t)) != type_index(typeid(T))) ||
		(data_tag_type == int32_t(MatDataType::miINT32) && type_index(typeid(int32_t)) != type_index(typeid(T))) ||
		(data_tag_type == int32_t(MatDataType::miUINT32) && type_index(typeid(uint32_t)) != type_index(typeid(T))) ||
		(data_tag_type == int32_t(MatDataType::miINT64) && type_index(typeid(int64_t)) != type_index(typeid(T))) ||
		(data_tag_type == int32_t(MatDataType::miUINT64) && type_index(typeid(uint64_t)) != type_index(typeid(T))) ||
		(data_tag_type == int32_t(MatDataType::miSINGLE) && type_index(typeid(float)) != type_index(typeid(T))) ||
		(data_tag_type == int32_t(MatDataType::miDOUBLE) && type_index(typeid(double)) != type_index(typeid(T))))
	{
		if (file_rawbytes) delete[]file_rawbytes;
		if (dims) delete[]dims;
		throw std::runtime_error("Could not parse the file. The file and array data types not match");
	}

	in_array = new T[data_tag_size];
	memcpy(in_array, file_rawbytes + offset, data_tag_size);

	if (file_rawbytes)
		delete[]file_rawbytes;
	return true;
}
template<class T>
bool MatlabUtils<T>::Save(std::string output_fname, T * in_array, int32_t * dims, int32_t n_dims, std::string mat_name)
{

	int32_t n_elements = 1;
	for (int32_t i = 0; i < n_dims; i++)
		n_elements *= dims[i];
	int32_t mat_header_size;
	uint8_t* mat_header = nullptr;
	setMatFileHeader(mat_header, mat_header_size);


	int32_t array_flags_tag_size;
	uint8_t *array_flags_tag = nullptr;
	setArrayFlagsTag(in_array, n_elements, array_flags_tag, array_flags_tag_size);


	int32_t dims_tag_size;
	uint8_t * dims_tag = nullptr;
	setDimsTag(dims, n_dims, dims_tag, dims_tag_size);

	int32_t array_name_tag_size;
	uint8_t *array_name_tag = nullptr;
	setNameTag(mat_name, array_name_tag, array_name_tag_size);


	int32_t data_tag_size;
	uint8_t *data_tag = nullptr;
	setDataTag(in_array, n_elements, data_tag, data_tag_size);

	int32_t matrix_tag_size;
	uint8_t *matrix_tag = nullptr;
	setMatrixTag(array_flags_tag_size, dims_tag_size, array_name_tag_size, data_tag_size, matrix_tag, matrix_tag_size);

	try
	{
		std::ofstream output_file(output_fname.c_str(), std::ios::out | std::ios::binary);
		output_file.write(reinterpret_cast<const char*>(mat_header), mat_header_size);
		output_file.write(reinterpret_cast<const char*>(matrix_tag), matrix_tag_size);
		output_file.write(reinterpret_cast<const char*>(array_flags_tag), array_flags_tag_size);
		output_file.write(reinterpret_cast<const char*>(dims_tag), dims_tag_size);
		output_file.write(reinterpret_cast<const char*>(array_name_tag), array_name_tag_size);
		output_file.write(reinterpret_cast<const char*>(data_tag), data_tag_size);
		output_file.close();
	}
	catch (exception e)
	{
		
		if (mat_header) delete[] mat_header;
		if (matrix_tag) delete[] matrix_tag;
		if (array_flags_tag) delete[] array_flags_tag;
		if (dims_tag) delete[] dims_tag;
		if (array_name_tag) delete[] array_name_tag;
		if (data_tag) delete[] data_tag;
		throw std::runtime_error("An error occured while trying to save the array :" + std::string(e.what()));
	}

	if (mat_header) delete[] mat_header;
	if (matrix_tag) delete[] matrix_tag;
	if (array_flags_tag) delete[] array_flags_tag;
	if (dims_tag) delete[] dims_tag;
	if (array_name_tag) delete[] array_name_tag;
	if (data_tag) delete[] data_tag;
	return true;
}

template<class T>
void MatlabUtils<T>::setMatFileHeader(uint8_t *& header, int32_t & header_size)
{
	std::string title = "MATLAB 5.0 MAT-file via MatUtils";
	header_size = 128;
	header = new uint8_t[header_size];

	std::fill_n(header, sizeof(uint8_t) * 128, 0);
	std::fill_n(header, sizeof(uint8_t) * 116, ' ');
	int16_t version = 0x0100;
	std::string endian_indicator = "IM";

	memcpy(header, title.c_str(), sizeof(char)*title.length());
	memcpy(header + 124, &version, sizeof(int16_t));
	memcpy(header + 126, endian_indicator.c_str(), sizeof(char)*2);
}

template<class T>
void MatlabUtils<T>::setMatrixTag(int32_t array_flags_tag_size, int32_t dims_tag_size, int32_t array_name_tag_size, int32_t data_tag_size, \
	uint8_t * & tag, int32_t & tag_size)
{
	if (tag)
		delete[] tag;
	tag_size = 8;
	tag = new uint8_t[tag_size];
	memset(tag, 0, tag_size);
	int32_t matrix_type = int32_t(MatDataType::miMATRIX);
	int32_t matrix_data_size = array_flags_tag_size + dims_tag_size + array_name_tag_size + data_tag_size;
	memset(tag, 0, tag_size);
	memcpy(tag, &matrix_type, sizeof(int32_t));
	memcpy(tag + 4, &matrix_data_size, sizeof(int32_t));
}

template<class T>
void MatlabUtils<T>::setArrayFlagsTag(T * data, int32_t n_elements, uint8_t* & tag, int32_t & tag_size)
{
	if (tag)
		delete[] tag;

	int32_t array_flags_type = int32_t(MatDataType::miUINT32);
	int32_t array_flags_data_size = 8;
	int32_t array_data_class = 0;


	if (type_index(typeid(int8_t)) == type_index(typeid(T)))
		array_data_class = int32_t(ArrayDataType::mxINT8_CLASS);
	else if (type_index(typeid(uint8_t)) == type_index(typeid(T)))
		array_data_class = int32_t(ArrayDataType::mxUINT8_CLASS);
	else if (type_index(typeid(int16_t)) == type_index(typeid(T)))
		array_data_class = int32_t(ArrayDataType::mxINT16_CLASS);
	else if (type_index(typeid(uint16_t)) == type_index(typeid(T)))
		array_data_class = int32_t(ArrayDataType::mxUINT16_CLASS);
	else if (type_index(typeid(int32_t)) == type_index(typeid(T)))
		array_data_class = int32_t(ArrayDataType::mxINT32_CLASS);
	else if (type_index(typeid(uint32_t)) == type_index(typeid(T)))
		array_data_class = int32_t(ArrayDataType::mxUINT32_CLASS);
	else if (type_index(typeid(int64_t)) == type_index(typeid(T)))
		array_data_class = int32_t(ArrayDataType::mxINT64_CLASS);
	else if (type_index(typeid(uint64_t)) == type_index(typeid(T)))
		array_data_class = int32_t(ArrayDataType::mxUINT64_CLASS);
	else if (type_index(typeid(float)) == type_index(typeid(T)))
		array_data_class = int32_t(ArrayDataType::mxSINGLE_CLASS);
	else if (type_index(typeid(double)) == type_index(typeid(T)))
		array_data_class = int32_t(ArrayDataType::mxDOUBLE_CLASS);

	uint8_t array_flags[8];
	memset(array_flags, 0, 8);
	array_flags[0] = array_data_class;

	tag_size = 16;
	tag = new uint8_t[tag_size];
	memset(tag, 0, tag_size);
	memcpy(tag, &array_flags_type, sizeof(int32_t));
	memcpy(tag + 4, &array_flags_data_size, sizeof(int32_t));
	memcpy(tag + 8, array_flags, 8);
}

template<class T>
void MatlabUtils<T>::setDimsTag(int32_t * dims, int32_t n_dims, uint8_t * & tag, int32_t & tag_size)
{
	if (tag)
		delete[] tag;
	int32_t array_dims_type = int32_t(MatDataType::miINT32);
	int32_t array_dims_size = n_dims * sizeof(int32_t);

	uint32_t zero_padding_in_bytes = array_dims_size % 8;
	if (zero_padding_in_bytes != 0)
		zero_padding_in_bytes = 8 - zero_padding_in_bytes;
	tag_size = 8 + array_dims_size + zero_padding_in_bytes;

	tag = new uint8_t[tag_size];
	memset(tag, 0, tag_size);
	memcpy(tag, &array_dims_type, sizeof(int32_t));
	memcpy(tag + 4, &array_dims_size, sizeof(int32_t));
	memcpy(tag + 8, reinterpret_cast<char *>(dims), array_dims_size);

}

template<class T>
void MatlabUtils<T>::setNameTag(std::string array_name, uint8_t* & tag, int32_t & tag_size)
{
	if (tag)
		delete[] tag;
	int32_t array_name_type = int32_t(MatDataType::miINT8);
	int32_t array_name_size = int32_t(array_name.size());
	uint32_t zero_padding_in_bytes = array_name_size % 8;
	if (zero_padding_in_bytes != 0)
		zero_padding_in_bytes = 8 - zero_padding_in_bytes;
	int32_t padded_size = array_name_size + zero_padding_in_bytes;

	tag_size = 8 + padded_size; //tag_header+array_name_size+zero_padding
	tag = new uint8_t[tag_size];
	memset(tag, 0, tag_size);
	memcpy(tag, &array_name_type, sizeof(int32_t));
	memcpy(tag + 4, &array_name_size, sizeof(int32_t));
	memcpy(tag + 8, array_name.c_str(), array_name_size);
}

template<class T>
void MatlabUtils<T>::setDataTag(T * data, int32_t n_elements, uint8_t* & tag, int32_t & tag_size)
{
	if (tag)
		delete[] tag;

	int32_t data_type = 0;
	if (type_index(typeid(int8_t)) == type_index(typeid(T)))
		data_type = int32_t(MatDataType::miINT8);
	else if (type_index(typeid(uint8_t)) == type_index(typeid(T)))
		data_type = int32_t(MatDataType::miUINT8);
	else if (type_index(typeid(int16_t)) == type_index(typeid(T)))
		data_type = int32_t(MatDataType::miINT16);
	else if (type_index(typeid(uint16_t)) == type_index(typeid(T)))
		data_type = int32_t(MatDataType::miUINT16);
	else if (type_index(typeid(int32_t)) == type_index(typeid(T)))
		data_type = int32_t(MatDataType::miINT32);
	else if (type_index(typeid(uint32_t)) == type_index(typeid(T)))
		data_type = int32_t(MatDataType::miUINT32);
	else if (type_index(typeid(int64_t)) == type_index(typeid(T)))
		data_type = int32_t(MatDataType::miINT64);
	else if (type_index(typeid(uint64_t)) == type_index(typeid(T)))
		data_type = int32_t(MatDataType::miUINT64);
	else if (type_index(typeid(float)) == type_index(typeid(T)))
		data_type = int32_t(MatDataType::miSINGLE);
	else if (type_index(typeid(double)) == type_index(typeid(T)))
		data_type = int32_t(MatDataType::miDOUBLE);

	if (data_type == 0)
		throw ("Unsupported data type");

	int32_t data_size = n_elements * sizeof(T);
	uint32_t zero_padding_in_bytes = data_size % 8;
	if (zero_padding_in_bytes != 0)
		zero_padding_in_bytes = 8 - zero_padding_in_bytes;
	tag_size = 8 + data_size + zero_padding_in_bytes;

	tag = new uint8_t[tag_size];
	memset(tag, 0, tag_size);
	memcpy(tag, &data_type, sizeof(int32_t));
	memcpy(tag + 4, &data_size, sizeof(int32_t));
	memcpy(tag + 8, data, data_size);
}



#endif