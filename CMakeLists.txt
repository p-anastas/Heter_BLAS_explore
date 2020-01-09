cmake_minimum_required (VERSION 3.10)
#set(CMAKE_VERBOSE_MAKEFILE on)
set ( CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" )
set ( CMAKE_CXX_COMPILER  "/opt/intel/bin/icc" )
project (MatrixSplit LANGUAGES CXX CUDA)

set ( PROJECT_SOURCE_DIR ${PROJECT_SOURCE_DIR}/src )
message( "PROJECT_SOURCE_DIR: " ${PROJECT_SOURCE_DIR} )
message( "PROJECT_BINARY_DIR: " ${PROJECT_BINARY_DIR} )
message( "CMAKE_CURRENT_SOURCE_DIR: " ${CMAKE_CURRENT_SOURCE_DIR} )
message( "CMAKE_CURRENT_BINARY_DIR: " ${CMAKE_CURRENT_BINARY_DIR} )

# C++ compile flags
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -qopenmp")
message( "CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS})

# NVCC compile flags
set (CMAKE_CUDA_FLAGS "-O3 -w")
message( "CMAKE_CUDA_FLAGS: " ${CMAKE_CUDA_FLAGS})

## CUDA libs and includes
set (CUDA_PREFIX "/usr/local/cuda")
set (CUDA_INCLUDE_DIRS ${CUDA_PREFIX}/include ${CUDA_PREFIX}/samples/common/inc)
include_directories(${CUDA_INCLUDE_DIRS} )
message( "CUDA_INCLUDE_DIRS: " ${CUDA_INCLUDE_DIRS} )
set (CUDA_LD ${CUDA_PREFIX}/lib)
set (CUDA_LD1 ${CUDA_PREFIX}/lib64)
set (CUDA_LINK "-L${CUDA_LD} -L${CUDA_LD1} -lcudart -lcublas -lcusparse ")
message( "CUDA_LINK: " ${CUDA_LINK} )

#MKL libs and includes 
set (MKL_PREFIX "/opt/intel/mkl") # "/various/common_tools/intel_parallel_studio/mkl" for cslab
set (MKL_INCLUDE_DIRS ${MKL_PREFIX}/include)
include_directories(${MKL_INCLUDE_DIRS} )
message( "MKL_INCLUDE_DIRS: " ${MKL_INCLUDE_DIRS} )
set (MKL_LD ${MKL_PREFIX}/lib/intel64)
set (MKL_LINK  "-L${MKL_LD} -lmkl_rt  -lpthread -lm" ) #-lnuma
message( "MKL_LINK: " ${MKL_LINK} )

#Intel Parallel Studio libs and includes
set (INTEL_PREFIX "/opt/intel/parallel_studio_xe_2019") #"/various/common_tools/intel_parallel_studio" for cslab
set (INTEL_INCLUDE_DIRS ${INTEL_PREFIX}/include)
include_directories(${INTEL_INCLUDE_DIRS} )
message( "INTEL_INCLUDE_DIRS: " ${INTEL_INCLUDE_DIRS} )
set (INTEL_LD ${INTEL_PREFIX}/lib/intel64)
set (INTEL_LINK  "-L${INTEL_LD} -xHost" )
message( "INTEL_LINK: " ${INTEL_LINK} )

# My header files 
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/inc)
message( "My header files: " ${CMAKE_CURRENT_SOURCE_DIR}/inc )


file(GLOB SOURCE_FILES ${PROJECT_SOURCE_DIR}/*.c* )
message( "SOURCE_FILES: " ${SOURCE_FILES} )

add_executable (dgemm_runner ${SOURCE_FILES})

set (LINK_LIB "${INTEL_LINK} ${MAGMA_LINK} ${MKL_LINK} ${CUDA_LINK} -lrt -ldl")
message( "LINK_LIB: " ${LINK_LIB} )
target_link_libraries(dgemm_runner ${LINK_LIB})
