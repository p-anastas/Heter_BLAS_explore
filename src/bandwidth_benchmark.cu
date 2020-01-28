///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The start of Zawarudo
///

#include <cuda.h>
#include <mkl.h>
#include "cublas_v2.h"

#include "benchmark_functions.hpp"
#include "cpu_utils.hpp"
#include "gpu_utils.hpp"


int main(const int argc, const char *argv[]) {
  void *src = NULL, *dest = NULL;
  double timer, prev_timer, Gbytes;
  
  size_t i, N_bytes, from, to, itterations = 1000, convergence = 0, area_bounds[3] = {0,0,0}, bound_step = 10;

  switch (argc) {
    case (4):
      N_bytes = atoi(argv[1]);
      from = atoi(argv[2]);
      to = atoi(argv[3]);
      break;
    default:
      error("Incorrect input arguments");
  }

timer = Dvec_init(&src, N_bytes, from, 10);
timer = Dvec_init(&dest, N_bytes, to, 10);
timer = check_bandwidth(N_bytes, dest, to, src, from, 10);
i = 128;
timer = 0;
prev_timer = 0;
while (!convergence && i <= N_bytes)
{
  timer = check_bandwidth(i, dest, to, src, from, itterations);
  Gbytes = i / (timer * 1e9);

if( Derror(timer,prev_timer) < 0.1 || i== 128) {
fprintf(stderr, "Constant area (%d-%d) timer = %lf prev_timer = %lf error= %lf\n", i/2,i, timer, prev_timer, Derror(timer,prev_timer));
prev_timer = timer;
}
else if(!area_bounds[0]){
fprintf(stderr, "\nFirst bound area (%d-%d) timer = %lf prev_timer = %lf error= %lf\n", i/2,i, timer, prev_timer, Derror(timer,prev_timer));
	timer = prev_timer;
	int ctr = i/2;
	while ( ctr < i && Derror(timer,prev_timer) < 0.01) {
		ctr = ctr + (i-i/2)/16;
		timer = check_bandwidth(ctr, dest, to, src, from, itterations);
	}
	area_bounds[0] = ctr;
        fprintf(stderr, "Constant area bound found:  %d timer = %lf prev_timer = %lf error= %lf\n\n", area_bounds[0],timer, prev_timer, Derror(timer,prev_timer));
        fprintf(stdout, "%d,%d,%d,%.15lf,bound_0\n", ctr, from, to, timer / itterations);
}

else if (Derror(timer/2,prev_timer) > 0.1) {
fprintf(stderr, "Non-linear area (%d-%d) timer/2 = %lf prev_timer = %lf error= %lf\n", i/2,i, timer/2, prev_timer, Derror(timer/2,prev_timer));
prev_timer = timer;
}
else if(!area_bounds[1]) {
fprintf(stderr, "\nSecond bound area (%d-%d) timer/2 = %lf prev_timer = %lf error= %lf\n", i/2,i, timer/2, prev_timer, Derror(timer/2,prev_timer));
	timer = prev_timer;
        double temp_prev_timer = prev_timer;
	int ctr = i/2;
	while ( ctr < i && Derror(timer*(ctr - (i-i/2)/16)/ctr,temp_prev_timer) > 0.01) {
		ctr = ctr + (i-i/2)/16;
                temp_prev_timer = timer;
		timer = check_bandwidth(ctr, dest, to, src, from, itterations);
                fprintf(stderr, "Checking Non-Linear area bound: timer_normalized = %lf prev_timer = %lf error= %lf\n\n", timer*(ctr - (i-i/2)/16)/ctr, temp_prev_timer, Derror(timer*(ctr - (i-i/2)/16)/ctr,temp_prev_timer));
	}
	area_bounds[1] = ctr;
        fprintf(stderr, "Non-Linear area bound found:  %d timer_normalized = %lf prev_timer = %lf error= %lf\n\n", area_bounds[1],timer*(ctr - (i-i/2)/16)/ctr, temp_prev_timer, Derror(timer*(i/2)/ctr,temp_prev_timer));
        fprintf(stdout, "%d,%d,%d,%.15lf,bound_1\n", ctr, from, to, timer / itterations);
}
else if (Derror(timer/2,prev_timer) > 0.01){
fprintf(stderr, "Sub-Linear area (%d-%d) timer/2 = %lf prev_timer = %lf error= %lf\n", i/2,i, timer/2, prev_timer, Derror(timer/2,prev_timer));
prev_timer = timer;
}
else{
	area_bounds[2] = i/2;
fprintf(stderr, "Linear area (%d-%d) timer/2 = %lf prev_timer = %lf error= %lf\n", i/2,i, timer/2, prev_timer, Derror(timer/2,prev_timer));
        fprintf(stdout, "%d,%d,%d,%.15lf,bound_2\n", i/2, from, to, prev_timer / itterations);
  convergence = 1;
}
    i=i*2; 

}
  return 0;
}
