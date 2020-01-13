///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The start of Zawarudo
///

#include <cuda.h>
#include <mkl.h>
#include "cublas_v2.h"

#include "cpu_utils.hpp"
#include "gpu_utils.hpp"

int main(const int argc, const char *argv[]) {

  double alpha, *src, *dest;

  size_t N_bytes, from, to, itterations = 10;

  switch (argc) {
    case (4):
      N_bytes = atoi(argv[1]);
      from = atoi(argv[2]);
      to = atoi(argv[3]);
      break;
    default:
      error("Incorrect input arguments");
  }


  double total_t = 0;

int count = 666;
cudaGetDeviceCount(&count);

if(-1 == from){
		fprintf(stderr, "Copying %d bytes from host...", N_bytes);
		src = Dvec_init_pinned(N_bytes, 42);
}
else if (from >=  count || from < 0)
		error("Invalid source device");
else{
	fprintf(stderr, "Copying %d bytes from device(%d)...", N_bytes, from);
	cudaSetDevice(from);
	cudaMalloc(&src, N_bytes);
}

if(-1 == to){
		fprintf(stderr, "to host\n", N_bytes);
		dest = Dvec_init_pinned(N_bytes, 0);
}
else if (to >= count || to < 0)
		error("Invalid destination device");
else{
	fprintf(stderr, "to device(%d)\n", to);
	cudaSetDevice(to);
	cudaMalloc(&dest, N_bytes);
}

  fprintf(stdout,"%d,%d,%d,",  N_bytes, from, to);

total_t = csecond();
if (-2 == from + to) 
	for(int it =0; it < itterations; it++) memcpy(dest,src, N_bytes);
if (-1 == from)
	for(int it =0; it < itterations; it++) cudaMemcpy(dest, src, N_bytes, cudaMemcpyHostToDevice);
if (-1 == to) 
	for(int it =0; it < itterations; it++) cudaMemcpy(dest, src, N_bytes, cudaMemcpyDeviceToHost);
else 
	for(int it =0; it < itterations; it++) cudaMemcpy(dest, src, N_bytes, cudaMemcpyDefault);
total_t = csecond() - total_t;
fprintf(stderr, "bandwidth(%d) benchmarked sucsessfully t = %lf ms ( %.3lf Gb/s %.15lf s/byte)\n", N_bytes, total_t*1000/itterations, 1e-9/(total_t/N_bytes/itterations) ,total_t/N_bytes/itterations);
fprintf(stdout,"%.15lf\n", total_t/itterations);
 return 0;
}
