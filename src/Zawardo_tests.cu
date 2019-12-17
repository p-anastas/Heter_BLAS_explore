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

int main(int argc, char *argv[]) {
  // print_devices();

  /*
  double *test, *test_T;

  test = Dvec_init_pinned(25, 42);
  test_T = Dvec_init_pinned(25, 0);

  int s1 = 5, s2 = 2;
  Dtranspose(test_T, test, s1, s2);

  for (int i = 0; i <s1; i++){
          for (int j = 0; j <s2; j++) printf("%0.3lf ", test[s2*i +j]);
          printf("\n");
  }
  printf("\n");
  for (int i = 0; i <s2; i++){
          for (int j = 0; j <s1; j++) printf("%0.3lf ", test_T[s1*i +j]);
          printf("\n");
  }

  exit(1);
  */

  int M = 1500, K = 2000, N = 1000;
  double alpha = 1.0, beta = 1.0;

  double transpose_timer, cpu_timer = csecond();
  float gpu_preproc_t = 0, gpu_comp_t = 0, gpu_reduce_t = 0;
  gpu_timer_p cuda_timer = gpu_timer_init();

  // cudaStream_t stream1, stream2;
  // cudaStreamCreate(&stream1);
  // cudaStreamCreate (&stream2);

  double *A, *B, *C, *C_comp, *C_buffer, *d_A, *d_B, *d_C, *C_T;

  A = Dvec_init_pinned(M * K, 42);
  B = Dvec_init_pinned(K * N, 42);
  C = Dvec_init_pinned(M * N, 42);
  C_buffer = Dvec_init_host(M * N, 0);
  C_comp = Dvec_init_host(M * N, 0);

  Dvec_copy(C_comp, C, M * N);
  Dvec_copy(C_buffer, C, M * N);

  cpu_timer = csecond() - cpu_timer;
  printf("Initializing Arrays on host (M=%d, N=%d, K=%d) t_init = %lf ms\n", M,
         N, K, cpu_timer * 1000);
  cpu_timer = csecond();
  for (int i = 0; i < 10; i++)
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, K,
                B, N, beta, C_comp, N);
  cpu_timer = csecond() - cpu_timer;
  printf("MKL Sgemm(M=%d, N=%d, K=%d) ", M, N, K);
  report_results(cpu_timer / 10, (long)M * K * (2 * N + 1),
                 (long)(M * K + K * N + M * N * 2) * sizeof(double));
  printf("\n");

  transpose_timer = csecond();
  C_T = (double *)pin_malloc(M * N * sizeof(double));
  Dtranspose(C_T, C, M, N);
  transpose_timer = csecond() - transpose_timer;

  gpu_timer_start(cuda_timer);
  d_A = Dvec_transfer_gpu(A, M * K);
  d_B = Dvec_transfer_gpu(B, K * N);
  d_C = Dvec_transfer_gpu(C_T, M * N);
  gpu_timer_stop(cuda_timer);
  gpu_preproc_t = gpu_timer_get(cuda_timer);

  cublasHandle_t handle;
  cublasStatus_t stat = cublasCreate(&handle);
  // cublasSetStream(handle, stream1);
  gpu_timer_start(cuda_timer);

  for (int i = 0; i < 10; i++) {
    stat = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, d_A,
                       K, d_B, N, &beta, d_C, M);
    cudaDeviceSynchronize();
  }

  gpu_timer_stop(cuda_timer);
  cudaCheckErrors();
  gpu_comp_t = gpu_timer_get(cuda_timer);
  gpu_timer_start(cuda_timer);
  cudaMemcpy(C_T, d_C, M * N * sizeof(double), cudaMemcpyDeviceToHost);
  gpu_timer_stop(cuda_timer);
  transpose_timer = csecond() - transpose_timer;
  Dtranspose(C, C_T, N, M);
  transpose_timer = csecond() - transpose_timer;
  gpu_reduce_t = gpu_timer_get(cuda_timer);

  printf(
      "Device overhead(M=%d, N=%d, K=%d) transpose = %lf ms, t_preproc = %lf "
      "ms, t_reduce = %lf ms\n",
      M, N, K, 1000 * transpose_timer, gpu_preproc_t, gpu_reduce_t);

  printf("CUDA Sgemm(M=%d, N=%d, K=%d) ", M, N, K);
  report_results((double)gpu_comp_t / 1000.0 / 10, (long)M * K * (2 * N + 1),
                 (long)(M * K + K * N + M * N * 2) *
                     sizeof(double));  //(M*N+(long)M*K*(3*N+1))
  printf("\n");

  Dtest_equality(C_comp, C, M * N);

  gpu_free(d_A);
  gpu_free(d_B);
  gpu_free(d_C);
  pin_free(C);
  pin_free(C_T);

  C = Dvec_init_pinned(M * N, 0);
  Dvec_copy(C, C_buffer, M * N);
  Dvec_copy(C_comp, C_buffer, M * N);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, K,
              B, N, beta, C_comp, N);

  // TODO: Stuff to predict
  // test_bandwidth(N*sizeof(float));

  /// FIXME: For now use only 2 devices/execution units

  /// For now use only 2 possible splits. If Dim_split = 0 then Dim equal to
  /// host.
  int M_split = 0, N_split = N * 2 / 5, K_split = 0;

  /// Flag regarding how copies will be handled. If = 0 then all data copied
  /// before op, otherwise chunks of size asynch_trans
  int asynch_trans = 0;

  /// Flag regarding 'cheating' the C = aAB + bC operation to  C = aAB +
  /// ghost_betaC in the device in order to  refer from copying the C matrix
  ///   to the device. Requires reduce at return.
  int ghost_beta, reduce = 0;

  if (argc > 1)
    ghost_beta = atoi(argv[1]);
  else
    ghost_beta = beta;

return_point:

  if (!M_split + !N_split + !K_split < 2)
    error("split more than one dim for 2 devices.");
  if (asynch_trans)
    debug(
        "asynch transactions not implemented yet, ignoring (nice try,though)");

  if (M_split) {
    if (M_split == M) {
      debug("executing solely on GPU");
      if (ghost_beta) {
        debug("executing with ghost_beta !=0");
        debug(
            "this is literally offloading naivelly the whole thing on CUBLAS");

        transpose_timer = csecond();
        C_T = (double *)pin_malloc(M * N * sizeof(double));
        Dtranspose(C_T, C, M, N);
        transpose_timer = csecond() - transpose_timer;

        gpu_timer_start(cuda_timer);
        d_A = Dvec_transfer_gpu(A, M * K);
        d_B = Dvec_transfer_gpu(B, K * N);
        d_C = Dvec_transfer_gpu(C_T, M * N);
        gpu_timer_stop(cuda_timer);
        gpu_preproc_t = gpu_timer_get(cuda_timer);

        cublasHandle_t handle;
        cublasStatus_t stat = cublasCreate(&handle);
        // cublasSetStream(handle, stream1);
        gpu_timer_start(cuda_timer);

        stat = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha,
                           d_A, K, d_B, N, &beta, d_C, M);

        cudaCheckErrors();
        gpu_timer_stop(cuda_timer);
        gpu_comp_t = gpu_timer_get(cuda_timer);
        gpu_timer_start(cuda_timer);
        cudaMemcpy(C_T, d_C, M * N * sizeof(double), cudaMemcpyDeviceToHost);
        gpu_timer_stop(cuda_timer);
        transpose_timer = csecond() - transpose_timer;
        Dtranspose(C, C_T, N, M);
        transpose_timer = csecond() - transpose_timer;
        gpu_reduce_t = gpu_timer_get(cuda_timer);

        printf(
            "Device overhead(M=%d, N=%d, K=%d) transpose = %lf ms, t_preproc = "
            "%lf ms, t_reduce = %lf ms\n",
            M, N, K, 1000 * transpose_timer, gpu_preproc_t, gpu_reduce_t);
        printf("CUDA Sgemm(M=%d, N=%d, K=%d) ", M, N, K);
        report_results((double)gpu_comp_t / 1000.0, (long)M * K * (2 * N + 1),
                       (long)(M * K + K * N + M * N * 2) *
                           sizeof(double));  //(M*N+(long)M*K*(3*N+1))
        printf("\n");
        printf("Total Sgemm(M=%d, N=%d, K=%d) ", M, N, K);
        report_results((double)gpu_comp_t / 1000.0 + transpose_timer +
                           (double)gpu_preproc_t / 1000 +
                           (double)gpu_reduce_t / 1000,
                       (long)M * K * (2 * N + 1),
                       (long)(M * K + K * N + M * N * 2) *
                           sizeof(double));  //(M*N+(long)M*K*(3*N+1))
        printf("\n");

        Dtest_equality(C_comp, C, M * N);

        gpu_free(d_A);
        gpu_free(d_B);
        gpu_free(d_C);
        pin_free(C);
        pin_free(C_T);

      } else {
        debug("executing with ghost_beta = 0");
        double dev_beta = 0.0;
        double *reduce_C;
        gpu_timer_start(cuda_timer);
        d_A = Dvec_transfer_gpu(A, M * K);
        d_B = Dvec_transfer_gpu(B, K * N);
        d_C = (double *)gpu_malloc(M * N * sizeof(double));
        gpu_timer_stop(cuda_timer);
        gpu_preproc_t = gpu_timer_get(cuda_timer);

        cublasHandle_t handle;
        cublasStatus_t stat = cublasCreate(&handle);
        // cublasSetStream(handle, stream1);
        gpu_timer_start(cuda_timer);

        stat = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha,
                           d_A, K, d_B, N, &dev_beta, d_C, M);
        // Hiding host computations etc here
        reduce_C = (double *)pin_malloc(M * N * sizeof(double));
        cblas_dscal(M * N, beta, C, 1);
        cudaCheckErrors();

        gpu_timer_stop(cuda_timer);

        gpu_comp_t = gpu_timer_get(cuda_timer);
        gpu_timer_start(cuda_timer);
        cudaMemcpy(reduce_C, d_C, M * N * sizeof(double),
                   cudaMemcpyDeviceToHost);
        gpu_timer_stop(cuda_timer);
        transpose_timer = csecond();
        Dtranspose_add(C, reduce_C, N, M);
        transpose_timer = csecond() - transpose_timer;
        gpu_reduce_t = gpu_timer_get(cuda_timer);

        printf(
            "Device overhead(M=%d, N=%d, K=%d) transpose = %lf ms, t_preproc = "
            "%lf ms, t_reduce = %lf ms\n",
            M, N, K, 1000 * transpose_timer, gpu_preproc_t, gpu_reduce_t);

        printf("CUDA Sgemm(M=%d, N=%d, K=%d) ", M, N, K);
        report_results((double)gpu_comp_t / 1000.0, (long)M * K * (2 * N + 1),
                       (long)(M * K + K * N + M * N * 2) *
                           sizeof(double));  //(M*N+(long)M*K*(3*N+1))
        printf("\n");
        printf("Total Sgemm(M=%d, N=%d, K=%d) ", M, N, K);
        report_results((double)gpu_comp_t / 1000.0 + transpose_timer +
                           (double)gpu_preproc_t / 1000 +
                           (double)gpu_reduce_t / 1000,
                       (long)M * K * (2 * N + 1),
                       (long)(M * K + K * N + M * N * 2) *
                           sizeof(double));  //(M*N+(long)M*K*(3*N+1))
        printf("\n");
        Dtest_equality(C_comp, C, M * N);

        gpu_free(d_A);
        gpu_free(d_B);
        gpu_free(d_C);
        pin_free(C);
        pin_free(reduce_C);
      }
    } else {
      debug("executing hybrid GPU-CPU (M)");
      size_t M_gpu = M_split, M_cpu = M - M_split;
      if (ghost_beta) {
        debug("executing with ghost_beta !=0");

        transpose_timer = csecond();
        double *A_T = (double *)pin_malloc(M_gpu * K * sizeof(double));
        Dtranspose(A_T, A, M_gpu, K);
        double *C_T = (double *)pin_malloc(M_gpu * N * sizeof(double));
        Dtranspose(C_T, C, M_gpu, N);
        transpose_timer = csecond() - transpose_timer;

        gpu_timer_start(cuda_timer);
        d_A = Dvec_transfer_gpu(A_T, M_gpu * K);
        d_B = Dvec_transfer_gpu(B, K * N);
        d_C = Dvec_transfer_gpu(C_T, M_gpu * N);
        gpu_timer_stop(cuda_timer);
        gpu_preproc_t = gpu_timer_get(cuda_timer);

        // cublasHandle_t handle;
        // cublasStatus_t stat = cublasCreate(&handle);
        // cublasSetStream(handle, stream1);
        gpu_timer_start(cuda_timer);

        stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, M_gpu, N, K,
                           &alpha, d_A, M_gpu, d_B, N, &beta, d_C, M_gpu);
        cpu_timer = csecond();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M_cpu, N, K,
                    alpha, &(A[M_gpu * K]), K, B, N, beta, &(C[M_gpu * N]), N);
        cpu_timer = csecond() - cpu_timer;
        cudaCheckErrors();
        gpu_timer_stop(cuda_timer);

        gpu_comp_t = gpu_timer_get(cuda_timer);
        gpu_timer_start(cuda_timer);
        cudaMemcpy(C_T, d_C, M_gpu * N * sizeof(double),
                   cudaMemcpyDeviceToHost);
        gpu_timer_stop(cuda_timer);
        gpu_reduce_t = gpu_timer_get(cuda_timer);
        transpose_timer = csecond() - transpose_timer;
        Dtranspose(C, C_T, N, M_gpu);
        transpose_timer = csecond() - transpose_timer;

        printf(
            "Device overhead(M=%d, N=%d, K=%d) transpose = %lf ms, t_preproc = "
            "%lf ms, t_reduce = %lf ms\n",
            M, N, K, 1000 * transpose_timer, gpu_preproc_t, gpu_reduce_t);

        printf("Hybrid Sgemm(M_cpu=%d, N=%d, K=%d) CPU time = ", M_cpu, N, K);
        report_results((double)cpu_timer, (long)M_cpu * K * (2 * N + 1),
                       (long)(M_cpu * K + K * N + M_cpu * N * 2) *
                           sizeof(double));  //(M*N+(long)M*K*(3*N+1))
        printf("\n");

        printf("Total Sgemm(M=%d, N=%d, K=%d) ", M, N, K);
        report_results((double)gpu_comp_t / 1000.0 + transpose_timer +
                           (double)gpu_preproc_t / 1000 +
                           (double)gpu_reduce_t / 1000,
                       (long)M * K * (2 * N + 1),
                       (long)(M * K + K * N + M * N * 2) *
                           sizeof(double));  //(M*N+(long)M*K*(3*N+1))
        printf("\n");

        Dtest_equality(C_comp, C, M * N);

        gpu_free(d_A);
        gpu_free(d_B);
        gpu_free(d_C);
        pin_free(C);
        pin_free(C_T);
        pin_free(A_T);
      } else {
        debug("executing with ghost_beta = 0");
        gpu_timer_start(cuda_timer);
        double dev_beta = 0.0;
        double *reduce_C;
        transpose_timer = csecond();
        double *A_T = (double *)pin_malloc(M_gpu * K * sizeof(double));
        Dtranspose(A_T, A, M_gpu, K);
        transpose_timer = csecond() - transpose_timer;

        gpu_timer_start(cuda_timer);
        d_A = Dvec_transfer_gpu(A_T, M_gpu * K);
        d_B = Dvec_transfer_gpu(B, K * N);
        d_C = (double *)gpu_malloc(M_gpu * N * sizeof(double));
        gpu_timer_stop(cuda_timer);
        gpu_preproc_t = gpu_timer_get(cuda_timer);

        cublasHandle_t handle;
        cublasStatus_t stat = cublasCreate(&handle);
        // cublasSetStream(handle, stream1);
        gpu_timer_start(cuda_timer);

        stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, M_gpu, N, K,
                           &alpha, d_A, M_gpu, d_B, N, &dev_beta, d_C, M_gpu);
        reduce_C = (double *)pin_malloc(M_gpu * N * sizeof(double));
        cblas_dscal(M_gpu * N, beta, C, 1);
        cpu_timer = csecond();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M_cpu, N, K,
                    alpha, &(A[M_gpu * K]), K, B, N, beta, &(C[M_gpu * N]), N);
        cpu_timer = csecond() - cpu_timer;
        cudaCheckErrors();

        gpu_timer_stop(cuda_timer);
        gpu_comp_t = gpu_timer_get(cuda_timer);
        gpu_timer_start(cuda_timer);
        cudaMemcpy(reduce_C, d_C, M_gpu * N * sizeof(double),
                   cudaMemcpyDeviceToHost);
        gpu_timer_stop(cuda_timer);
        gpu_reduce_t = gpu_timer_get(cuda_timer);

        transpose_timer = csecond() - transpose_timer;
        Dtranspose_add(C, reduce_C, N, M_gpu);
        transpose_timer = csecond() - transpose_timer;

        printf(
            "Device overhead(M=%d, N=%d, K=%d) transpose = %lf ms, t_preproc = "
            "%lf ms, t_reduce = %lf ms\n",
            M, N, K, 1000 * transpose_timer, gpu_preproc_t, gpu_reduce_t);

        printf("Hybrid Sgemm(M_cpu=%d, N=%d, K=%d) CPU time = ", M_cpu, N, K);
        report_results((double)cpu_timer, (long)M_cpu * K * (2 * N + 1),
                       (long)(M_cpu * K + K * N + M_cpu * N * 2) *
                           sizeof(double));  //(M*N+(long)M*K*(3*N+1))
        printf("\n");

        printf("Total Sgemm(M=%d, N=%d, K=%d) ", M, N, K);
        report_results((double)gpu_comp_t / 1000.0 + transpose_timer +
                           (double)gpu_preproc_t / 1000 +
                           (double)gpu_reduce_t / 1000,
                       (long)M * K * (2 * N + 1),
                       (long)(M * K + K * N + M * N * 2) *
                           sizeof(double));  //(M*N+(long)M*K*(3*N+1))
        printf("\n");

        Dtest_equality(C_comp, C, M * N);

        gpu_free(d_A);
        gpu_free(d_B);
        gpu_free(d_C);
        pin_free(C);
        pin_free(reduce_C);
        pin_free(A_T);
      }
    }
  } else if (N_split) {
    if (N_split == N) {
      debug("executing solely on GPU but with N spliting (?)");
      debug("Not gonna copy the same code, a goto is preffered");
      N_split = 0;
      M_split = M;
      goto return_point;
    } else {
      debug("executing hybrid GPU-CPU (N)");
      size_t N_gpu = N_split, N_cpu = N - N_split;
      if (ghost_beta) {
        debug("executing with ghost_beta !=0");

        transpose_timer = csecond();
        double *B_T = (double *)pin_malloc(K * N * sizeof(double));
        Dtranspose(B_T, B, K, N);
        double *C_T = (double *)pin_malloc(M * N * sizeof(double));
        Dtranspose(C_T, C, M, N);
        transpose_timer = csecond() - transpose_timer;

        gpu_timer_start(cuda_timer);
        d_A = Dvec_transfer_gpu(A, M * K);
        d_B = Dvec_transfer_gpu(B_T, K * N_gpu);
        d_C = Dvec_transfer_gpu(C_T, M * N_gpu);
        gpu_timer_stop(cuda_timer);
        gpu_preproc_t = gpu_timer_get(cuda_timer);

        gpu_timer_start(cuda_timer);

        stat = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N_gpu, K,
                           &alpha, d_A, K, d_B, K, &beta, d_C, M);
        cpu_timer = csecond();
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, M, N_cpu, K, alpha,
                    A, K, &(B_T[N_gpu * K]), K, beta, &(C_T[M * N_gpu]), M);
        cpu_timer = csecond() - cpu_timer;
        cudaCheckErrors();

        gpu_timer_stop(cuda_timer);
        gpu_comp_t = gpu_timer_get(cuda_timer);
        gpu_timer_start(cuda_timer);
        cudaMemcpy(C_T, d_C, M * N_gpu * sizeof(double),
                   cudaMemcpyDeviceToHost);
        gpu_timer_stop(cuda_timer);
        gpu_reduce_t = gpu_timer_get(cuda_timer);

        transpose_timer = csecond() - transpose_timer;
        Dtranspose(C, C_T, N, M);
        transpose_timer = csecond() - transpose_timer;

        printf(
            "Device overhead(M=%d, N=%d, K=%d) transpose = %lf ms, t_preproc = "
            "%lf ms, t_reduce = %lf ms\n",
            M, N, K, 1000 * transpose_timer, gpu_preproc_t, gpu_reduce_t);

        printf("Hybrid Sgemm(M=%d, N_cpu=%d, K=%d) CPU time = ", M, N_cpu, K);
        report_results((double)cpu_timer, (long)M * K * (2 * N_cpu + 1),
                       (long)(M * K + K * N_cpu + M * N_cpu * 2) *
                           sizeof(double));  //(M*N+(long)M*K*(3*N+1))
        printf("\n");

        printf("Total Sgemm(M=%d, N=%d, K=%d) ", M, N, K);
        report_results((double)gpu_comp_t / 1000.0 + transpose_timer +
                           (double)gpu_preproc_t / 1000 +
                           (double)gpu_reduce_t / 1000,
                       (long)M * K * (2 * N + 1),
                       (long)(M * K + K * N + M * N * 2) *
                           sizeof(double));  //(M*N+(long)M*K*(3*N+1))
        printf("\n");

        Dtest_equality(C_comp, C, M * N);

        gpu_free(d_A);
        gpu_free(d_B);
        gpu_free(d_C);
      } else {
        debug("executing with ghost_beta = 0");
        gpu_timer_start(cuda_timer);
        double dev_beta = 0.0;

        transpose_timer = csecond();
        double *B_T = (double *)pin_malloc(K * N * sizeof(double));
        Dtranspose(B_T, B, K, N);
        double *C_T = (double *)pin_malloc(M * N * sizeof(double));
        Dtranspose(C_T, C, M, N);
        transpose_timer = csecond() - transpose_timer;

        gpu_timer_start(cuda_timer);
        d_A = Dvec_transfer_gpu(A, M * K);
        d_B = Dvec_transfer_gpu(B_T, K * N_gpu);
        d_C = (double *)gpu_malloc(M * N_gpu * sizeof(double));
        gpu_timer_stop(cuda_timer);
        gpu_preproc_t = gpu_timer_get(cuda_timer);

        cublasHandle_t handle;
        cublasStatus_t stat = cublasCreate(&handle);
        // cublasSetStream(handle, stream1);
        gpu_timer_start(cuda_timer);

        stat = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N_gpu, K,
                           &alpha, d_A, K, d_B, K, &dev_beta, d_C, M);
        cblas_dscal(M * N, beta, C, 1);
        cpu_timer = csecond();
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, M, N_cpu, K, alpha,
                    A, K, &(B_T[N_gpu * K]), K, 0.0, &(C_T[M * N_gpu]), M);
        cpu_timer = csecond() - cpu_timer;
        cudaCheckErrors();

        gpu_timer_stop(cuda_timer);
        gpu_comp_t = gpu_timer_get(cuda_timer);
        gpu_timer_start(cuda_timer);
        cudaMemcpy(C_T, d_C, M * N_gpu * sizeof(double),
                   cudaMemcpyDeviceToHost);
        gpu_timer_stop(cuda_timer);
        gpu_reduce_t = gpu_timer_get(cuda_timer);

        transpose_timer = csecond() - transpose_timer;
        Dtranspose_add(C, C_T, N, M);
        transpose_timer = csecond() - transpose_timer;

        printf(
            "Device overhead(M=%d, N_gpu=%d, K=%d) t_preproc = %lf ms, "
            "t_reduce = "
            "%lf ms\n",
            M, N_gpu, K, gpu_preproc_t, gpu_reduce_t);

        printf("Hybrid Sgemm(M=%d, N_gpu=%d, K=%d) CPU time = ", M, N_cpu, K);
        report_results((double)cpu_timer, (long)M * K * (2 * N_cpu + 1),
                       (long)(M * K + K * N_cpu + M * N_cpu * 2) *
                           sizeof(double));  //(M*N+(long)M*K*(3*N+1))
        printf("\n");

        printf("Total Sgemm(M=%d, N=%d, K=%d) ", M, N, K);
        report_results((double)gpu_comp_t / 1000.0 + transpose_timer +
                           (double)gpu_preproc_t / 1000 +
                           (double)gpu_reduce_t / 1000,
                       (long)M * K * (2 * N + 1),
                       (long)(M * K + K * N + M * N * 2) *
                           sizeof(double));  //(M*N+(long)M*K*(3*N+1))
        printf("\n");

        Dtest_equality(C_comp, C, M * N);

        gpu_free(d_A);
        gpu_free(d_B);
        gpu_free(d_C);
      }
    }
  } else if (K_split) {
  } else {
    debug("Not spliting at all, execute the whole on host");

    cpu_timer = csecond();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, K,
                B, N, beta, C, N);
    cpu_timer = csecond() - cpu_timer;
    printf("MKL Sgemm(M=%d, N=%d, K=%d) ", M, N, K);
    report_results(cpu_timer, (long)M * K * (2 * N + 1),
                   (long)(M * K + K * N + M * N * 2) * sizeof(double));
    printf("\n");
    Dtest_equality(C_comp, C, M * N);
  }
}
