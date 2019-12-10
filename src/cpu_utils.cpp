///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief Some CUDA function calls with added error-checking
///

#include "cpu_utils.hpp"

#include <stdlib.h>
#include <time.h>
#include <float.h>
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


double Drandom(double min, double max){
return (max - min) * ( (double)rand() / (double)RAND_MAX ) + min;
}

size_t Svec_diff(float* a, float* b, size_t size){
	size_t failed = 0; 
	for (size_t i = 0; i < size; i++) if (!Sequals(a[i], b[i])) failed++;
	return failed;
}

int Sequals(float a, float b) {
		float absA = abs(a);
		float absB = abs(b);
		float diff = abs(a - b);

		if (a == b) { // shortcut, handles infinities
			return 1;
    } else if (a == 0 || b == 0 || (absA + absB < FLT_MIN)) {
			// a or b is zero or both are extremely close to it
			// relative error is less meaningful here
			if (diff < ( FLT_EPSILON * FLT_MIN)) return 1;
			else return 0; 
		} else { // use relative error
			if(diff / std::min((absA + absB), FLT_MIN) < FLT_MIN) return 1;
			else return 0; 
		}
	}


size_t Dvec_diff(double* a, double* b, size_t size){
	size_t failed = 0; 
	for (size_t i = 0; i < size; i++) if (!Dequals(a[i], b[i])) failed++;
	return failed;
}

int Dequals(double a, double b) {
		double absA = abs(a);
		double absB = abs(b);
		double diff = abs(a - b);

		if (a == b) { // shortcut, handles infinities
			return 1;
    } else if (a == 0 || b == 0 || (absA + absB < DBL_MIN)) {
			// a or b is zero or both are extremely close to it
			// relative error is less meaningful here
			if (diff < ( DBL_EPSILON * DBL_MIN)) return 1;
			else return 0; 
		} else { // use relative error
			if(diff / std::min((absA + absB), DBL_MIN) < DBL_EPSILON) return 1;
			else return 0; 
		}
	}

