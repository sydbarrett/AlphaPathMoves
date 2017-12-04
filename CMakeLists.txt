
cmake_minimum_required(VERSION 3.9.4)
project(PathMoves LANGUAGES C;CXX;CUDA)
find_package(CUDA QUIET REQUIRED)


set(CMAKE_SUPPRESS_REGENERATION true) 
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11" )


include(CheckTypeSize)
macro(ValidateExists ${validation_var})
	check_type_size(${validation_var} VAR_SIZE)
	if(${VAR_SIZE}=="0" OR ${VAR_SIZE}=="")
		message( FATAL_ERROR "${validation_var} data type is missing" )
	endif()
endmacro()

#check_type_size("void *" VOID_SIZE)
#message(${VOID_SIZE})
#ValidateExists("void *")
set( validation_var "void *")
check_type_size("void *" VAR_SIZE)
if(${VAR_SIZE}=="0" OR ${VAR_SIZE}=="")
	message( FATAL_ERROR "${validation_var} data type is missing" )
endif()
set(CMAKE_SUPPRESS_REGENERATION true) 


file(GLOB SOURCES_LIST
    "${PathMoves_SOURCE_DIR}/Sources/*.cpp"
	"${PathMoves_SOURCE_DIR}/Sources/QPBO/*.cpp"
	"${PathMoves_SOURCE_DIR}/Sources/BK/*.cpp"
	"${PathMoves_SOURCE_DIR}/Sources/CUDA/*.cu")
	
file(GLOB HEADERS_LIST
    "${PathMoves_SOURCE_DIR}/Headers/*.h"
	"${PathMoves_SOURCE_DIR}/Headers/CUDA/*.cuh"
	"${PathMoves_SOURCE_DIR}/Headers/IBFS/*.h"
	"${PathMoves_SOURCE_DIR}/Headers/QPBO/*.h"
	"${PathMoves_SOURCE_DIR}/Headers/BK/*.h")

include_directories( "${PathMoves_SOURCE_DIR}/Headers"
					 "${PathMoves_SOURCE_DIR}/Headers/CUDA"
					 "${PathMoves_SOURCE_DIR}/Headers/IBFS"
					 "${PathMoves_SOURCE_DIR}/Headers/QPBO"
					 "${PathMoves_SOURCE_DIR}/Headers/BK")

add_executable(PathMoves ${SOURCES_LIST} ${HEADERS_LIST})
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT PathMoves)


foreach(source IN LISTS SOURCES_LIST)
    get_filename_component(source_path "${source}" PATH)
    string(REPLACE "${PathMoves_SOURCE_DIR}/" "" source_path_msvc "${source_path}")
	string(REPLACE "/" "\\" source_path_msvc "${source_path_msvc}")
    source_group("${source_path_msvc}" FILES "${source}")
endforeach()

foreach(header IN LISTS HEADERS_LIST)
    get_filename_component(header_path "${header}" PATH)
	string(REPLACE "${PathMoves_SOURCE_DIR}/" "" header_path_msvc "${header_path}")
    string(REPLACE "/" "\\" header_path_msvc "${header_path_msvc}")
    source_group("${header_path_msvc}" FILES "${header}")
endforeach()