///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief Some CUDA function calls with added error-checking
///
#ifndef GPU_UTILS_H
#define GPU_UTILS_H

#include "Zawardo_defines.hpp"

#include <cuda.h>

typedef struct gpu_timer {
  cudaEvent_t start;
  cudaEvent_t stop;
  float ms = 0;
} * gpu_timer_p;

gpu_timer_p gpu_timer_init();
void gpu_timer_start(gpu_timer_p timer);
void gpu_timer_stop(gpu_timer_p timer);
float gpu_timer_get(gpu_timer_p timer);

/// Test memory bandwidth for various copies of lenght 'bytes'
void test_bandwidth(size_t bytes);

/// Print all available CUDA devices and their basic info
void print_devices();

/// Check if there are CUDA errors on the stack
void cudaCheckErrors();

/// Allocate 'count' bytes of CUDA device memory (+errorcheck)
void* gpu_malloc(size_t count);
void* pin_malloc(size_t count);

/// Free the CUDA device  memory pointed by 'gpuptr' (+errorcheck)
void gpu_free(void* gpuptr);
void pin_free(void* gpuptr);

/// Print Free/Total CUDA device memory along with 'message' (+errorcheck)
void gpu_showMem(char* message);

float* Svec_init_pinned(size_t size, float val);
float* Svec_transfer_gpu(float* host_vec, size_t size);

double* Dvec_init_pinned(size_t size, double val);
double* Dvec_transfer_gpu(double* host_vec, size_t size);

#endif
