///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief Some CUDA function calls with added error-checking
///

#include <float.h>
#include <cstdio>
#include "cpu_utils.hpp"
#include "gpu_utils.hpp"
#include "benchmark_functions.hpp"

double Dvec_init(void ** ptr, size_t N_bytes, int loc, size_t itterations){
  double total_t = 0; 
  int count = 666;

gpu_timer_p cuda_timer;
  cudaGetDeviceCount(&count);

  if (-1 == loc) {
    fprintf(stderr, "Allocating %d bytes to host...", N_bytes);
  cuda_timer = gpu_timer_init();
  gpu_timer_start(cuda_timer);
     for (int it = 0; it < itterations; it++) {
		if (*ptr) pin_free(*ptr);
		*ptr = pin_malloc(N_bytes);
    }

  } else if (loc >= count || loc < 0)
    error("Dvec_init: Invalid device");
  else {
    fprintf(stderr, "Allocating %d bytes to device(%d)...", N_bytes, loc);
    cudaSetDevice(loc);
  cuda_timer = gpu_timer_init();
  gpu_timer_start(cuda_timer);
     for (int it = 0; it < itterations; it++) {
		if (*ptr) gpu_free(*ptr);
		*ptr = gpu_malloc(N_bytes);
    }
  }
  gpu_timer_stop(cuda_timer);
  total_t = gpu_timer_get(cuda_timer) / 1000;
  fprintf(stderr,
          "Allocation(%d) benchmarked successfully t = %lf ms ( %.3lf Gb/s "
          "%.15lf s/byte)\n",
          N_bytes, total_t * 1000 / itterations,
          1e-9 / (total_t / N_bytes / itterations),
          total_t / N_bytes / itterations);
  //fprintf(stdout, "%d,%d,%.15lf\n", N_bytes, loc,
  //        total_t / itterations);
  return total_t;

}

double check_bandwidth(size_t N_bytes, void * dest, int to, void * src, int from, size_t itterations) {
  double total_t = 0; 
  int count = 666;
  cudaGetDeviceCount(&count);

  if (-1 == from) {
    fprintf(stderr, "Copying %d bytes from host...", N_bytes);
  } else if (from >= count || from < 0)
    error("check_bandwidth: Invalid source device");
  else {
    fprintf(stderr, "Copying %d bytes from device(%d)...", N_bytes, from);
    cudaSetDevice(from);
  }

  if (-1 == to) {
    fprintf(stderr, "to host -> ", N_bytes);
  } else if (to >= count || to < 0)
    error("check_bandwidth: Invalid destination device");
  else {
    fprintf(stderr, "to device(%d) -> ", to);
    cudaSetDevice(to);
  }

  gpu_timer_p cuda_timer = gpu_timer_init();
  gpu_timer_start(cuda_timer);
  if (-2 == from + to)
    for (int it = 0; it < itterations; it++) memcpy(dest, src, N_bytes);
  if (-1 == from)
    for (int it = 0; it < itterations; it++)
      cudaMemcpy(dest, src, N_bytes, cudaMemcpyHostToDevice);
  if (-1 == to)
    for (int it = 0; it < itterations; it++)
      cudaMemcpy(dest, src, N_bytes, cudaMemcpyDeviceToHost);
  else
    for (int it = 0; it < itterations; it++)
      cudaMemcpy(dest, src, N_bytes, cudaMemcpyDeviceToDevice);
  gpu_timer_stop(cuda_timer);
  total_t = gpu_timer_get(cuda_timer) / 1000;
  fprintf(stderr,
          "bandwidth(%d) benchmarked successfully t = %lf ms ( %.3lf Gb/s "
          "%.15lf s/byte)\n",
          N_bytes, total_t * 1000 / itterations,
          1e-9 / (total_t / N_bytes / itterations),
          total_t / N_bytes / itterations);
  fprintf(stdout, "%d,%d,%d,%.15lf\n", N_bytes, from, to,
          total_t / itterations);
  return total_t;
}
