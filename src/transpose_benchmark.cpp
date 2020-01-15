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
  double alpha;

  size_t N, M, itterations = 1;

  N = M = 300;

  switch (argc) {
    case (3):
      N = atoi(argv[1]);
      M = atoi(argv[2]);
      break;
    default:
      error("Incorrect input arguments");
  }

  double total_t = 0;

  double *x, *y;

  x = Dvec_init_pinned(N * M, 42);
  y = Dvec_init_pinned(N * M, 0);

  itterations = 1000;

  total_t = csecond();
  for (int it = 0; it < itterations; it++) Dtranspose(y, x, N, M);
  total_t = csecond() - total_t;
  fprintf(stderr,
          "transpose(%d,%d) benchmarked sucsessfully t = %lf ms ( %.15lf "
          "s/double)\n",
          N, M, total_t * 1000 / itterations, total_t / N * M / itterations);
  fprintf(stdout, "%d,%d,%.15lf\n", Ν, Μ, total_t / itterations);
  return 0;
}
