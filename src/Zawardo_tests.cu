///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief The start of Zawarudo
///

#include <cuda.h>
#include <mkl.h>
#include "cublas_v2.h"

#include "gpu_utils.hpp"
#include "cpu_utils.hpp"

int main(void)
{

print_devices();

  int M,K, N=K=M = 1.5e4;
float alpha = 1.0 , beta = 1.0;

test_bandwidth(N*sizeof(float));

double cpu_timer = csecond();
float gpu_alloc_t, gpu_transa_t, gpu_comp_t;
gpu_timer_p cuda_timer = gpu_timer_init();
cudaStream_t stream1, stream2;
cudaStreamCreate (&stream1);
cudaStreamCreate (&stream2);

  float *A, *B, *C, *C_comp, *d_A, *d_B, *d_C;
  cudaMallocHost(&A, M*K*sizeof(float));
  cudaMallocHost(&B, K*N*sizeof(float));
  cudaMallocHost(&C, M*N*sizeof(float));
  C_comp =  (float*)malloc(M*N*sizeof(float));

 memset(A, 0, M*K*sizeof(float));
 memset(B, 0, K*N*sizeof(float));
 memset(C, 0, M*N*sizeof(float));

for (int i = 0; i< M*K; i++) A[i] = 3;//(float) Drandom(-1,1);
for (int i = 0; i< K*N; i++) B[i] = 1;//(float) Drandom(-1,1);
for (int i = 0; i< M*N; i++) C[i] = C_comp[i] = 2;//(float) Drandom(-1,1);

size_t failed = Svec_diff(C_comp,C, M*N);
if(failed) printf("Test failed %d times\n", failed);
else printf("Test passed(C)\n");

cpu_timer = csecond() - cpu_timer;
printf("Initializing Arrays on host (M=%d, N=%d, K=%d) t_init = %lf ms\n",M, N, K, cpu_timer*1000); 
cpu_timer = csecond();
for (int i = 0; i< NR_ITER; i++)  cblas_sgemm (CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
                           alpha,
                           A, M,
                           B, K,
                           beta,
                           C_comp, M);
cpu_timer = csecond() - cpu_timer;
printf("MKL Sgemm(M=%d, N=%d, K=%d) ",M, N, K);
report_results(cpu_timer, (long) M*K*(2*N+1), (long)(M*K+K*N+M*N*2)*sizeof(float));
printf("\n");

  gpu_timer_start(cuda_timer);
  cudaMalloc(&d_A, M*K*sizeof(float)); 
  cudaMalloc(&d_B, K*N*sizeof(float));
  cudaMalloc(&d_C, M*N*sizeof(float));
  gpu_timer_stop(cuda_timer);
gpu_alloc_t = gpu_timer_get(cuda_timer);

  gpu_timer_start(cuda_timer);
  cudaMemcpy(d_A, A, M*K*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, K*N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C, M*N*sizeof(float), cudaMemcpyHostToDevice);
  gpu_timer_stop(cuda_timer);
gpu_transa_t = gpu_timer_get(cuda_timer);

cublasHandle_t handle;
cublasStatus_t stat = cublasCreate(&handle);
cublasSetStream(handle, stream1);
  gpu_timer_start(cuda_timer);


for (int i = 0; i< NR_ITER; i++) 
{stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
			M, N, K,
                           &alpha,
                           d_A, M,
                           d_B, K,
                           &beta,
                           d_C, M);
cudaDeviceSynchronize();
}

  gpu_timer_stop(cuda_timer);
cudaCheckErrors();
  gpu_comp_t = gpu_timer_get(cuda_timer);
  gpu_timer_start(cuda_timer);
  cudaMemcpy(C, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);
  gpu_timer_stop(cuda_timer);
gpu_transa_t += gpu_timer_get(cuda_timer);

printf("Device overhead(M=%d, N=%d, K=%d) t_alloc = %.3f ms, t_transa = %.3f ms\n",M, N, K, gpu_alloc_t, gpu_transa_t); 
  printf("CUDA Sgemm(M=%d, N=%d, K=%d) ",M, N, K);
report_results((double)gpu_comp_t/1000.0, (long) M*K*(2*N+1), (long)(M*K+K*N+M*N*2)*sizeof(float)); //(M*N+(long)M*K*(3*N+1))
printf("\n");

failed = Svec_diff(C_comp,C, M*N);
if(failed) {
printf("Test failed %d times\n", failed);
for (int i = 0; i< 10; i++) printf("CPU vs GPU: %f vs %f\n", C_comp[i], C[i]);
}
else printf("Test passed(C)\n");
}

