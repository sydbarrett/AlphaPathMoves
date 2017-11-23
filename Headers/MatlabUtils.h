#ifndef __MATLABUTILS_H_
#define __MATLABUTILS_H_
#include <iostream>
#include <fstream>
#include <typeinfo> 
#include <typeindex>

template <class T >
class MatlabUtils
{
private:
	enum class MatDataType : long int {
		miINT8 = 1, miUINT8 = 2, miINT16 = 3, miUINT16 = 4, miINT32 = 5, miUINT32 = 6, \
		miSINGLE = 7, miDOUBLE = 9, miINT64 = 12, miUINT64 = 13, miMATRIX = 14
	};
	enum class ArrayDataType : long int {
		mxCELL_CLASS = 1, mxSTRUCT_CLAS = 2, mxOBJECT_CLASS = 3, mxCHAR_CLASS = 4, \
		mxSPARSE_CLASS = 5, mxDOUBLE_CLASS = 6, mxSINGLE_CLASS = 7, mxINT8_CLASS = 8, \
		mxUINT8_CLASS = 9, mxINT16_CLASS = 10, mxUINT16_CLASS = 11, mxINT32_CLASS = 12, \
		mxUINT32_CLASS = 13, mxINT64_CLASS = 14, mxUINT64_CLASS = 15
	};

	static void setMatFileHeader(unsigned char *& header, long int & header_size);
	static void setMatrixTag(long int array_flags_tag_size, long int dims_tag_size, long int array_name_tag_size, long int data_tag_size, \
		unsigned char * & tag, long int & tag_size);
	static void setArrayFlagsTag(T * data, long int n_elements, unsigned char* & tag, long int & tag_size);
	static void setDimsTag(long int * dims, long int n_dims, unsigned char * & tag, long int & tag_size);
	static void setNameTag(std::string array_name, unsigned char* & tag, long int & tag_size);
	static void setDataTag(T * data, long int n_elements, unsigned char* & tag, long int & tag_size);

	static void readTag(unsigned char * tag, long int & tag_type, long int & tag_size);

public:

	static bool Save(std::string output_fname, T * in_array, long int * dims, long int n_dims, std::string mat_name = "no_name");
	static bool Load(std::string output_fname, T *& in_array, long int *& dims, long int & n_dims);
};

template<class T>
void MatlabUtils<T>::readTag(unsigned char * tag, long int & tag_type, long int & tag_size)
{
	tag_type = long int(tag[0] | tag[1] << 8 | tag[2] << 16 | tag[3] << 24);
	tag_size = long int(tag[4] | tag[5] << 8 | tag[6] << 16 | tag[7] << 24);
}

template<class T>
bool MatlabUtils<T>::Load(std::string output_fname, T *& in_array, long int *& dims, long int & n_dims)
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
	unsigned char* file_rawbytes = nullptr;
	size_t n_bytes = 0;
	try
	{
		input_file.open(output_fname.c_str(), std::ios::in | std::ios::binary);
		input_file.seekg(0, ios::end);
		n_bytes = input_file.tellg();

		file_rawbytes = new unsigned char[n_bytes];
		memset(file_rawbytes, 0, n_bytes);
		input_file.seekg(0, ios::beg);
		input_file.read(reinterpret_cast<char *>(file_rawbytes), n_bytes);
		input_file.close();
	}
	catch (exception e)
	{
		if (file_rawbytes) delete[]file_rawbytes;
		throw("Could not load .mat file : " + std::string(e.what()));
	}
	if (n_bytes <= 128)
	{
		if (file_rawbytes) delete[]file_rawbytes;
		throw("Could not parse the .mat file: it can not be less than 128 bytes");
	}
	//check that the first 4 bytes are not zeros
	if (file_rawbytes[0] == 0 && file_rawbytes[1] == 0 && \
		file_rawbytes[2] == 0 && file_rawbytes[3] == 0)
	{
		if (file_rawbytes) delete[]file_rawbytes;
		throw("This is not a Level 5 .mat file");
	}

	//Ignore the first 125 bytes 
	//validate that  byte[126]=I and byte[127]='M'
	if (file_rawbytes[126] != 'I' && file_rawbytes[127] != 'M')
	{
		if (file_rawbytes) delete[]file_rawbytes;
		throw("This library only supports little endian systems\n");
	}
	long int offset = 128;
	long int matrix_tag_type, matrix_tag_size;
	readTag(file_rawbytes + offset, matrix_tag_type, matrix_tag_size);
	offset += 8;
	if (matrix_tag_type != long int(MatDataType::miMATRIX))
	{
		if (file_rawbytes) delete[]file_rawbytes;
		throw("This library only reads matrices of signed/unsigend numerical types");
	}

	long int array_flags_tag_type, array_flags_tag_size;
	readTag(file_rawbytes + offset, array_flags_tag_type, array_flags_tag_size);
	char matrix_class = file_rawbytes[offset + 8];
	offset += 16; //ignore the actual flags
	if (array_flags_tag_type != long int(MatDataType::miUINT32) && array_flags_tag_size != 8)
	{
		if (file_rawbytes) delete[]file_rawbytes;
		throw("Could not parse the file as a Level 5 .mat file");
	}
	if (matrix_class == _int8(ArrayDataType::mxCELL_CLASS) ||
		matrix_class == _int8(ArrayDataType::mxSTRUCT_CLAS) ||
		matrix_class == _int8(ArrayDataType::mxOBJECT_CLASS) ||
		matrix_class == _int8(ArrayDataType::mxCHAR_CLASS) ||
		matrix_class == _int8(ArrayDataType::mxSPARSE_CLASS))
	{
		if (file_rawbytes) delete[]file_rawbytes;
		throw("This library only reads matrices of signed/unsigend numerical types");
	}


	long int dims_tag_type, dims_tag_size;
	readTag(file_rawbytes + offset, dims_tag_type, dims_tag_size);
	offset += 8;
	if (dims_tag_type != long int(MatDataType::miINT32))
	{
		if (file_rawbytes) delete[]file_rawbytes;
		throw("Could not parse the file as a Level 5 .mat file");
	}
	n_dims = dims_tag_size / sizeof(long int);
	long int dims_zero_padding = dims_tag_size % 8;
	if (dims_zero_padding != 0)
		dims_zero_padding = 8 - dims_zero_padding;
	dims = new long int[n_dims];
	memcpy(dims, file_rawbytes + offset, dims_tag_size);
	offset += dims_tag_size;
	offset += dims_zero_padding;

	long int name_tag_type, name_tag_size;
	readTag(file_rawbytes + offset, name_tag_type, name_tag_size);
	offset += 8;
	if (name_tag_type != long int(MatDataType::miINT8))
	{
		if (file_rawbytes) delete[]file_rawbytes;
		if (dims) delete[]dims;
		throw("Could not parse the file as a Level 5 .mat file");
	}
	long int name_zero_padding = name_tag_size % 8;
	if (name_zero_padding != 0)
		name_zero_padding = 8 - name_zero_padding;
	offset += name_tag_size;
	offset += name_zero_padding;

	long int data_tag_type, data_tag_size;
	readTag(file_rawbytes + offset, data_tag_type, data_tag_size);
	offset += 8;

	if ((data_tag_type == long int(MatDataType::miINT8) && type_index(typeid(signed char)) != type_index(typeid(T))) ||
		(data_tag_type == long int(MatDataType::miUINT8) && type_index(typeid(unsigned char)) != type_index(typeid(T))) ||
		(data_tag_type == long int(MatDataType::miINT16) && type_index(typeid(signed int)) != type_index(typeid(T))) ||
		(data_tag_type == long int(MatDataType::miUINT16) && type_index(typeid(unsigned int)) != type_index(typeid(T))) ||
		(data_tag_type == long int(MatDataType::miINT32) && type_index(typeid(signed long int)) != type_index(typeid(T))) ||
		(data_tag_type == long int(MatDataType::miUINT32) && type_index(typeid(unsigned long int)) != type_index(typeid(T))) ||
		(data_tag_type == long int(MatDataType::miINT64) && type_index(typeid(signed long long)) != type_index(typeid(T))) ||
		(data_tag_type == long int(MatDataType::miUINT64) && type_index(typeid(unsigned long long)) != type_index(typeid(T))) ||
		(data_tag_type == long int(MatDataType::miSINGLE) && type_index(typeid(float)) != type_index(typeid(T))) ||
		(data_tag_type == long int(MatDataType::miDOUBLE) && type_index(typeid(double)) != type_index(typeid(T))))
	{
		if (file_rawbytes) delete[]file_rawbytes;
		if (dims) delete[]dims;
		throw("Could not parse the file. The file and array data types not match");
	}

	in_array = new T[data_tag_size];
	memcpy(in_array, file_rawbytes + offset, data_tag_size);

	if (file_rawbytes)
		delete[]file_rawbytes;
	return true;
}
template<class T>
bool MatlabUtils<T>::Save(std::string output_fname, T * in_array, long int * dims, long int n_dims, std::string mat_name)
{

	long int n_elements = 1;
	for (long int i = 0; i < n_dims; i++)
		n_elements *= dims[i];
	long int mat_header_size;
	unsigned char* mat_header = nullptr;
	setMatFileHeader(mat_header, mat_header_size);


	long int array_flags_tag_size;
	unsigned char *array_flags_tag = nullptr;
	setArrayFlagsTag(in_array, n_elements, array_flags_tag, array_flags_tag_size);


	long int dims_tag_size;
	unsigned char * dims_tag = nullptr;
	setDimsTag(dims, n_dims, dims_tag, dims_tag_size);

	long int array_name_tag_size;
	unsigned char *array_name_tag = nullptr;
	setNameTag(mat_name, array_name_tag, array_name_tag_size);


	long int data_tag_size;
	unsigned char *data_tag = nullptr;
	setDataTag(in_array, n_elements, data_tag, data_tag_size);

	long int matrix_tag_size;
	unsigned char *matrix_tag = nullptr;
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
		throw("An error occured while trying to save the array :" + std::string(e.what()));
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
void MatlabUtils<T>::setMatFileHeader(unsigned char *& header, long int & header_size)
{
	std::string title = "MATLAB 5.0 MAT-file via MatUtils";
	header_size = 128;
	header = new unsigned char[header_size];

	std::fill_n(header, sizeof(unsigned char) * 128, 0);
	std::fill_n(header, sizeof(unsigned char) * 116, ' ');
	short int version = 0x0100;
	std::string endian_indicator = "IM";

	memcpy(header, title.c_str(), sizeof(char)*title.length());
	memcpy(header + 124, &version, sizeof(short int));
	memcpy(header + 126, endian_indicator.c_str(), sizeof(char)*2);
}

template<class T>
void MatlabUtils<T>::setMatrixTag(long int array_flags_tag_size, long int dims_tag_size, long int array_name_tag_size, long int data_tag_size, \
	unsigned char * & tag, long int & tag_size)
{
	if (tag)
		delete[] tag;
	tag_size = 8;
	tag = new unsigned char[tag_size];
	memset(tag, 0, tag_size);
	long int matrix_type = long int(MatDataType::miMATRIX);
	long int matrix_data_size = array_flags_tag_size + dims_tag_size + array_name_tag_size + data_tag_size;
	memset(tag, 0, tag_size);
	memcpy(tag, &matrix_type, sizeof(long int));
	memcpy(tag + 4, &matrix_data_size, sizeof(long int));
}

template<class T>
void MatlabUtils<T>::setArrayFlagsTag(T * data, long int n_elements, unsigned char* & tag, long int & tag_size)
{
	if (tag)
		delete[] tag;

	long int array_flags_type = long int(MatDataType::miUINT32);
	long int array_flags_data_size = 8;
	long int array_data_class = 0;


	if (type_index(typeid(signed char)) == type_index(typeid(T)))
		array_data_class = long int(ArrayDataType::mxINT8_CLASS);
	else if (type_index(typeid(unsigned char)) == type_index(typeid(T)))
		array_data_class = long int(ArrayDataType::mxUINT8_CLASS);
	else if (type_index(typeid(signed int)) == type_index(typeid(T)))
		array_data_class = long int(ArrayDataType::mxINT16_CLASS);
	else if (type_index(typeid(unsigned int)) == type_index(typeid(T)))
		array_data_class = long int(ArrayDataType::mxUINT16_CLASS);
	else if (type_index(typeid(signed long int)) == type_index(typeid(T)))
		array_data_class = long int(ArrayDataType::mxINT32_CLASS);
	else if (type_index(typeid(unsigned long int)) == type_index(typeid(T)))
		array_data_class = long int(ArrayDataType::mxUINT32_CLASS);
	else if (type_index(typeid(signed long long)) == type_index(typeid(T)))
		array_data_class = long int(ArrayDataType::mxINT64_CLASS);
	else if (type_index(typeid(unsigned long long)) == type_index(typeid(T)))
		array_data_class = long int(ArrayDataType::mxUINT64_CLASS);
	else if (type_index(typeid(float)) == type_index(typeid(T)))
		array_data_class = long int(ArrayDataType::mxSINGLE_CLASS);
	else if (type_index(typeid(double)) == type_index(typeid(T)))
		array_data_class = long int(ArrayDataType::mxDOUBLE_CLASS);

	unsigned char array_flags[8];
	memset(array_flags, 0, 8);
	array_flags[0] = array_data_class;

	tag_size = 16;
	tag = new unsigned char[tag_size];
	memset(tag, 0, tag_size);
	memcpy(tag, &array_flags_type, sizeof(long int));
	memcpy(tag + 4, &array_flags_data_size, sizeof(long int));
	memcpy(tag + 8, array_flags, 8);
}

template<class T>
void MatlabUtils<T>::setDimsTag(long int * dims, long int n_dims, unsigned char * & tag, long int & tag_size)
{
	if (tag)
		delete[] tag;
	long int array_dims_type = long int(MatDataType::miINT32);
	long int array_dims_size = n_dims * sizeof(long int);

	unsigned long int zero_padding_in_bytes = array_dims_size % 8;
	if (zero_padding_in_bytes != 0)
		zero_padding_in_bytes = 8 - zero_padding_in_bytes;
	tag_size = 8 + array_dims_size + zero_padding_in_bytes;

	tag = new unsigned char[tag_size];
	memset(tag, 0, tag_size);
	memcpy(tag, &array_dims_type, sizeof(long int));
	memcpy(tag + 4, &array_dims_size, sizeof(long int));
	memcpy(tag + 8, reinterpret_cast<char *>(dims), array_dims_size);

}

template<class T>
void MatlabUtils<T>::setNameTag(std::string array_name, unsigned char* & tag, long int & tag_size)
{
	if (tag)
		delete[] tag;
	long int array_name_type = long int(MatDataType::miINT8);
	long int array_name_size = long int(array_name.size());
	unsigned long int zero_padding_in_bytes = array_name_size % 8;
	if (zero_padding_in_bytes != 0)
		zero_padding_in_bytes = 8 - zero_padding_in_bytes;
	long int padded_size = array_name_size + zero_padding_in_bytes;

	tag_size = 8 + padded_size; //tag_header+array_name_size+zero_padding
	tag = new unsigned char[tag_size];
	memset(tag, 0, tag_size);
	memcpy(tag, &array_name_type, sizeof(long int));
	memcpy(tag + 4, &array_name_size, sizeof(long int));
	memcpy(tag + 8, array_name.c_str(), array_name_size);
}

template<class T>
void MatlabUtils<T>::setDataTag(T * data, long int n_elements, unsigned char* & tag, long int & tag_size)
{
	if (tag)
		delete[] tag;

	long int data_type = 0;
	if (type_index(typeid(signed char)) == type_index(typeid(T)))
		data_type = long int(MatDataType::miINT8);
	else if (type_index(typeid(unsigned char)) == type_index(typeid(T)))
		data_type = long int(MatDataType::miUINT8);
	else if (type_index(typeid(signed int)) == type_index(typeid(T)))
		data_type = long int(MatDataType::miINT16);
	else if (type_index(typeid(unsigned int)) == type_index(typeid(T)))
		data_type = long int(MatDataType::miUINT16);
	else if (type_index(typeid(signed long int)) == type_index(typeid(T)))
		data_type = long int(MatDataType::miINT32);
	else if (type_index(typeid(unsigned long int)) == type_index(typeid(T)))
		data_type = long int(MatDataType::miUINT32);
	else if (type_index(typeid(signed long long)) == type_index(typeid(T)))
		data_type = long int(MatDataType::miINT64);
	else if (type_index(typeid(unsigned long long)) == type_index(typeid(T)))
		data_type = long int(MatDataType::miUINT64);
	else if (type_index(typeid(float)) == type_index(typeid(T)))
		data_type = long int(MatDataType::miSINGLE);
	else if (type_index(typeid(double)) == type_index(typeid(T)))
		data_type = long int(MatDataType::miDOUBLE);

	if (data_type == 0)
		throw ("Unsupported data type");

	long int data_size = n_elements * sizeof(T);
	unsigned long int zero_padding_in_bytes = data_size % 8;
	if (zero_padding_in_bytes != 0)
		zero_padding_in_bytes = 8 - zero_padding_in_bytes;
	tag_size = 8 + data_size + zero_padding_in_bytes;

	tag = new unsigned char[tag_size];
	memset(tag, 0, tag_size);
	memcpy(tag, &data_type, sizeof(long int));
	memcpy(tag + 4, &data_size, sizeof(long int));
	memcpy(tag + 8, data, data_size);
}



#endif