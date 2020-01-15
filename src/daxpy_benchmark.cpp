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

  size_t N, itterations = 1, incx, incy;

  N = 300;
  alpha = 1.1;
  incx = incy = 1;

  switch (argc) {
    case (5):
      incx = atoi(argv[3]);
      incy = atoi(argv[4]);
    case (3):
      N = atoi(argv[1]);
      alpha = atof(argv[2]);
      break;
    default:
      error("Incorrect input arguments");
  }

  double total_t = 0;

  double *x, *y;

  x = Dvec_init_pinned(N, 42);
  y = Dvec_init_pinned(N, 0);

  itterations = 1000;

  total_t = csecond();
  for (int it = 0; it < itterations; it++)
    cblas_daxpy(N, alpha, x, incx, y, incy);
  total_t = csecond() - total_t;
  fprintf(stderr,
          "daxpy(%d) benchmarked sucsessfully t = %lf ms ( %.15lf s/double)\n",
          N, total_t * 1000 / itterations, total_t / N / itterations);
  fprintf(stdout, "%d,%lf,%d,%d,%.15lf\n", N, alpha, incx, incy,
          total_t / itterations);
  return 0;
}
