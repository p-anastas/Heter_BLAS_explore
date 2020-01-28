///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief Some CUDA function calls with added error-checking
///
#ifndef BENCH_FUNCS_H
#define BENCH_FUNCS_H

#include "Zawardo_defines.hpp"

#include <cuda.h>

double Dvec_init(void ** ptr, size_t N_bytes, int loc, size_t itterations);
double check_bandwidth(size_t N_bytes, void * dest, int to, void * src, int from, size_t itterations);

#endif



