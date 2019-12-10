///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief Some CUDA function calls with added error-checking
///
#ifndef CPU_UTILS_Η
#define CPU_UTILS_Η

#include "Zawardo_defines.hpp"
#include <stdio.h>

void debug(char* string);
void massert(bool condi, const char* msg);
double csecond();
void warning(char* string);
void error(char* string);
void report_results(double timer, long flops, long bytes);
void report_bandwidth(double timer, size_t bytes);
double Drandom(double min, double max);
size_t Svec_diff(float* a, float* b, size_t size);
int Sequals(float a, float b);
size_t Dvec_diff(double* a, double* b, size_t size);
int Dequals(double a, double b);

#endif