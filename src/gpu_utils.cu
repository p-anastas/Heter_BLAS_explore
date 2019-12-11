///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief Some CUDA function calls with added error-checking
///

#include <cstdio>
#include "cpu_utils.hpp"
#include "gpu_utils.hpp"

void print_devices() {
  debug(" -> print_devices()");
  cudaDeviceProp properties;
  int nDevices = 0;
  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaGetDeviceProperties(&properties, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", properties.name);
    printf("  Memory Clock Rate (MHz): %d\n",
           properties.memoryClockRate / 1024);
    printf("  Memory Bus Width (bits): %d\n", properties.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n",
           2.0 * properties.memoryClockRate * (properties.memoryBusWidth / 8) /
               1.0e6);
    if (properties.major >= 3)
      printf("  Unified Memory support: YES\n\n");
    else
      printf("  Unified Memory support: NO\n\n");
  }
  debug(" <- print_devices()");
}

void test_bandwidth(size_t bytes) {
  debug(" -> test_bandwidth(bytes)");
  double exc_timer = 0, *host_vector = (double *)malloc(bytes),
         *host_vector1 = (double *)malloc(bytes);
  double *pinvector;
  cudaHostAlloc(&pinvector, bytes, cudaHostAllocDefault);
  double *devector;
  cudaMalloc(&devector, bytes);
  // double * univector; cudaMallocManaged(&univector, bytes);
  printf("Running Transaction benchmarks...\n\n");
  printf("Timing memcpy(%.3lf Mb)...", bytes * 1.0 / 1024 / 1024);
  exc_timer = csecond() - exc_timer;
  for (int i = 0; i < NR_ITER; i++) memcpy(host_vector, host_vector1, bytes);
  exc_timer = csecond() - exc_timer;
  printf("\n -- Host to Host\t: ");
  report_bandwidth(exc_timer, bytes);
  free(host_vector1);
  /*
  exc_timer = 0;
    for (int i = 0; i < NR_ITER; i++) {
      cudaMemPrefetchAsync(univector, bytes, 0);
      cudaDeviceSynchronize();
      exc_timer = csecond() - exc_timer;
      memcpy(host_vector, univector, bytes);
      exc_timer = csecond() - exc_timer;
    }
  printf("\n -- Unified to Host\t: ");
  report_bandwidth(exc_timer,bytes);
  exc_timer = 0;
    for (int i = 0; i < NR_ITER; i++) {
      cudaMemPrefetchAsync(univector, bytes, 0);
      cudaDeviceSynchronize();
      exc_timer = csecond() - exc_timer;
      memcpy(univector, host_vector, bytes);
      exc_timer = csecond() - exc_timer;
    }
  printf("\n -- Host to Unified\t: ");
  report_bandwidth(exc_timer,bytes);
  */
  exc_timer = 0;
  exc_timer = csecond() - exc_timer;
  for (int i = 0; i < NR_ITER; i++) memcpy(host_vector, pinvector, bytes);
  exc_timer = csecond() - exc_timer;
  printf("\n -- Pinned H. to Host\t: ");
  report_bandwidth(exc_timer, bytes);
  exc_timer = 0;
  exc_timer = csecond() - exc_timer;
  for (int i = 0; i < NR_ITER; i++) memcpy(pinvector, host_vector, bytes);
  exc_timer = csecond() - exc_timer;
  printf("\n -- Host to Pinned H.\t: ");
  report_bandwidth(exc_timer, bytes);
  printf("\n\n");
  printf("Timing cudaMemcpy(%.3lf Mb)...", bytes * 1.0 / 1024 / 1024);
  exc_timer = 0;
  exc_timer = csecond() - exc_timer;
  for (int i = 0; i < NR_ITER; i++) {
    cudaMemcpy(devector, devector, bytes, cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
  }
  exc_timer = csecond() - exc_timer;
  printf("\n -- Device to Device\t: ");
  report_bandwidth(exc_timer, bytes);
  exc_timer = 0;
  exc_timer = csecond() - exc_timer;
  for (int i = 0; i < NR_ITER; i++)
    cudaMemcpy(host_vector, devector, bytes, cudaMemcpyDeviceToHost);
  exc_timer = csecond() - exc_timer;
  printf("\n -- Device to Host\t: ");
  report_bandwidth(exc_timer, bytes);
  exc_timer = 0;
  exc_timer = csecond() - exc_timer;
  for (int i = 0; i < NR_ITER; i++)
    cudaMemcpy(devector, host_vector, bytes, cudaMemcpyHostToDevice);
  exc_timer = csecond() - exc_timer;
  printf("\n -- Host to Device\t: ");
  report_bandwidth(exc_timer, bytes);
  exc_timer = 0;
  exc_timer = csecond() - exc_timer;
  for (int i = 0; i < NR_ITER; i++)
    cudaMemcpy(devector, pinvector, bytes, cudaMemcpyHostToDevice);
  exc_timer = csecond() - exc_timer;
  printf("\n -- Pinned H. to Device\t: ");
  report_bandwidth(exc_timer, bytes);
  exc_timer = 0;
  exc_timer = csecond() - exc_timer;
  for (int i = 0; i < NR_ITER; i++)
    cudaMemcpy(pinvector, devector, bytes, cudaMemcpyDeviceToHost);
  exc_timer = csecond() - exc_timer;
  printf("\n -- Device to Pinned H.\t: ");
  report_bandwidth(exc_timer, bytes);
  /*
  exc_timer = 0;
  exc_timer = csecond() - exc_timer;
    for (int i = 0; i < NR_ITER; i++)
      cudaMemcpy(univector, pinvector, bytes,
                 cudaMemcpyHostToDevice);
      exc_timer = csecond() - exc_timer;
  printf("\n -- Pinned H. to Uni.\t: ");
  report_bandwidth(exc_timer,bytes);
  exc_timer = 0;
  exc_timer = csecond() - exc_timer;
    for (int i = 0; i < NR_ITER; i++)
      cudaMemcpy(pinvector, univector, bytes,
                 cudaMemcpyDeviceToHost);
      exc_timer = csecond() - exc_timer;
  printf("\n -- Uni. to Pinned H.\t: ");
  report_bandwidth(exc_timer,bytes);
  */
  printf("\n\n");
  free(host_vector);
  cudaFreeHost(pinvector);
  gpu_free(devector);

  debug(" <- test_bandwidth(bytes)");
}

void cudaCheckErrors() {
  cudaError_t errSync = cudaGetLastError();
  cudaError_t errAsync = cudaDeviceSynchronize();
  if (errSync != cudaSuccess)
    printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
  if (errAsync != cudaSuccess)
    printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
}

void *gpu_alloc(size_t count) {
  void *ret;
  massert(cudaMalloc(&ret, count) == cudaSuccess,
          cudaGetErrorString(cudaGetLastError()));
  return ret;
}

void gpu_free(void *gpuptr) {
  massert(cudaFree(gpuptr) == cudaSuccess,
          cudaGetErrorString(cudaGetLastError()));
}

void gpu_showMem(char *message) {
  size_t free, total;
  massert(cudaMemGetInfo(&free, &total) == cudaSuccess,
          cudaGetErrorString(cudaGetLastError()));
  printf("showMem(%s): %u free host_vector of %u MB \n", message,
         free / (1024 * 1024), total / (1024 * 1024));
}

gpu_timer_p gpu_timer_init() {
  gpu_timer_p timer = (gpu_timer_p)malloc(sizeof(struct gpu_timer));
  cudaEventCreate(&timer->start);
  cudaEventCreate(&timer->stop);
  return timer;
}

void gpu_timer_start(gpu_timer_p timer) { cudaEventRecord(timer->start); }

void gpu_timer_stop(gpu_timer_p timer) { cudaEventRecord(timer->stop); }

float gpu_timer_get(gpu_timer_p timer) {
  cudaEventSynchronize(timer->stop);
  cudaEventElapsedTime(&timer->ms, timer->start, timer->stop);
  return timer->ms;
}

float *Svec_init_pinned(size_t size, float val) {
  float *vec;
  cudaMallocHost(&vec, size * sizeof(float));
  for (size_t i = 0; i < size; i++) vec[i] = val;  //(float) Drandom(-1,1);
  return vec;
}

float *Svec_transfer_gpu(float *host_vec, size_t size) {
  float *dev_vec;
  cudaMalloc(&dev_vec, size * sizeof(float));
  cudaMemcpy(dev_vec, host_vec, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors();
  return dev_vec;
}
