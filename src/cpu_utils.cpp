///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief Some CUDA function calls with added error-checking
///

#include "cpu_utils.hpp"

#include <float.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>

void debug(char* string) {
#ifdef DEBUG
  printf("DEBUG: %s\n", string);
#endif
}

void massert(bool condi, const char* msg) {
  if (!condi) {
    printf("Error: %s\n", msg);
    exit(1);
  }
}

double csecond(void) {
  struct timespec tms;

  if (clock_gettime(CLOCK_REALTIME, &tms)) {
    return (0.0);
  }
  /// seconds, multiplied with 1 million
  int64_t micros = tms.tv_sec * 1000000;
  /// Add full microseconds
  micros += tms.tv_nsec / 1000;
  /// round up if necessary
  if (tms.tv_nsec % 1000 >= 500) {
    ++micros;
  }
  return ((double)micros / 1000000.0);
}

void warning(char* string) { printf("WARNING ( %s )\n", string); }

void error(char* string) {
  printf("ERROR ( %s ) halting execution\n", string);
  exit(1);
}

void report_results(double timer, long flops, long bytes) {
  double time = timer / NR_ITER;
  double Gflops = flops / (time * 1e9);
  double Gbytes = bytes / (time * 1e9);
  printf("%lf ms ( %.2lf Gflops/s %.2lf Gbytes/s)", 1000.0 * time, Gflops,
         Gbytes);
}

void report_bandwidth(double timer, size_t bytes) {
  double time = timer / NR_ITER;
  double Gbytes = bytes / (time * 1e9);
  printf("%lf ms ( %.2lf Gbytes/s)", 1000.0 * time, Gbytes);
}

double Drandom(double min, double max) {
  return (max - min) * ((double)rand() / (double)RAND_MAX) + min;
}

size_t Svec_diff(float* a, float* b, size_t size) {
  size_t failed = 0;
  for (size_t i = 0; i < size; i++)
    if (!Sequals(a[i], b[i])) failed++;
  return failed;
}

int Sequals(float a, float b) {
  float absA = abs(a);
  float absB = abs(b);
  float diff = abs(a - b);

  if (a == b) {  // shortcut, handles infinities
    return 1;
  } else if (a == 0 || b == 0 || (absA + absB < FLT_MIN)) {
    // a or b is zero or both are extremely close to it
    // relative error is less meaningful here
    if (diff < (FLT_EPSILON * FLT_MIN))
      return 1;
    else
      return 0;
  } else {  // use relative error
    if (diff / std::min((absA + absB), FLT_MIN) < FLT_MIN)
      return 1;
    else
      return 0;
  }
}

size_t Dvec_diff(double* a, double* b, size_t size, double eps) {
  size_t failed = 0;
  for (size_t i = 0; i < size; i++)
    if (!Dequals(a[i], b[i], eps)) failed++;
  return failed;
}

int Dequals(double a, double b, double eps) {
  double absA = abs(a);
  double absB = abs(b);
  double diff = abs(a - b);

  if (a == b) {  // shortcut, handles infinities
    return 1;
  } else if (a == 0 || b == 0 || (absA + absB < DBL_MIN)) {
    // a or b is zero or both are extremely close to it
    // relative error is less meaningful here
    if (diff < (eps * DBL_MIN))
      return 1;
    else
      return 0;
  } else {  // use relative error
    if (diff /* /std::min((absA + absB), DBL_MIN)*/ < eps)
      return 1;
    else
      return 0;
  }
}

float* Svec_init_host(size_t size, float val) {
  float* vec;
  vec = (float*)malloc(size * sizeof(float));
  for (size_t i = 0; i < size; i++) vec[i] = val;  //(float) Drandom(-1,1);
  return vec;
}

double* Dvec_init_host(size_t size, double val) {
  double* vec;
  vec = (double*)malloc(size * sizeof(double));
  if (Dequals(val, 42, DBL_EPSILON))
    for (size_t i = 0; i < size; i++) vec[i] = Drandom(-1, 1);
  else
    for (size_t i = 0; i < size; i++) vec[i] = val;
  return vec;
}

void Stranspose(float* vec, size_t dim1, size_t dim2) {
  debug("Ignoring transposing for now");
  /*
  float swap = 0;
    for (size_t i = 0; i < dim1; i++) for (size_t j = 0; j < dim2; j++) {
                  swap = vec[dim1*j + i];
                  vec[dim1*j + i] = vec[dim2*i + j];
                  vec[dim2*i + j] = swap;
  }
  */
}

double* Dtranspose(double* vec, size_t dim1, size_t dim2) {
  // debug("Ignoring transposing for now");
  double swap = 0;
  double* buffer = (double*)malloc(dim1 * dim2 * sizeof(double));
  for (size_t i = 0; i < dim1; i++) {
#pragma omp parallel for
    for (size_t j = 0; j < dim2; j++) buffer[dim1 * j + i] = vec[dim2 * i + j];
  }
  return buffer;
}

void Dvec_copy(double* dest, double* src, size_t size) {
  /*
  #pragma omp parallel for
   for (size_t i = 0; i < size; i++) dest[i] = src[i];
  */
  memcpy(dest, src, size * sizeof(double));
}

void Dtest_equality(double* C_comp, double* C, size_t size) {
  size_t acc = 0, failed;
  double eps = 0.1;
  failed = Dvec_diff(C_comp, C, size, eps);
  while (eps > DBL_MIN && !failed) {
    eps *= 0.1;
    acc++;
    failed = Dvec_diff(C_comp, C, size, eps);
  }
  if (!acc) {
    printf("Test failed %d times\n", failed);
  } else
    printf("Test passed(Accuracy= %d digits, %d/%d breaking for %d)\n", acc,
           failed, size, acc + 1);
  for (int i = 0; i < 10; i++)
    if (!Dequals(C_comp[i], C[i], eps))
      printf("CPU vs GPU: %.15lf vs %.15lf\n", C_comp[i], C[i]);
}
