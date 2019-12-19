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

/// Inputs: A, B, C, alpha, beta, M, N, K, store format (Colmajor OR Rowmajor)
/// for A, B, C
/// TODO: Add lda, ldb, ldc of initial call in order to also predict cases for
/// sub-matrix gemm ( if ld_dim > dim etc)

/// Predictor inputs: M_split (OR) N_split (OR) K_split, <<cblas Layout (for
/// CPU), >>, C_add device
/// TODO: Also add asynch trans

/// Output: A, B intact, C dgemm result
/// TODO: No mem leaks (CPU and GPU)

/// Extras: Precise debug and parameter error checking
/// TODO: More than two devices

enum mem_layout { COL_MAJOR, ROW_MAJOR };

typedef struct control_str {
  // For now assume device 0 is always the host
  size_t num_devices;
  mem_layout A_mem;
  mem_layout B_mem;
  mem_layout C_mem;
  double pin_alloc_t = 0, scatter_t = 0, transpose_t = 0, cpu_ex_t = 0,
         gpu_ex_t = 0, gather_t = 0, reduce_t = 0;

} * control_p;

typedef struct pred_str {
  size_t M_split;
  size_t N_split;
  size_t K_split;
  CBLAS_LAYOUT cblas_target;
  size_t Cadd_device_id;
  // TODO: Not implemented
  int asynch_trans = 0;

} * predict_p;

double *hybrid_dgemm_Msplit(control_p ctrl, predict_p pred, size_t M, size_t N,
                            size_t K, double alpha, double *A, double *B,
                            double beta, double *C) {
  debug("-> hybrid_dgemm_Msplit()\n");
  if (ctrl->num_devices < 1)
    error(
        "hybrid_dgemm_Msplit -> 0 or less devices? What are you trying to "
        "do...");
  else if (ctrl->num_devices > 2)
    error(
        "hybrid_dgemm_Msplit -> Max 1 GPU + 1 CPU implemented (nice "
        "try,though).");

  if (pred->asynch_trans)
    error(
        "hybrid_dgemm_Msplit -> asynch transactions not implemented yet "
        "(nice try,though).");

  if (!pred->M_split || pred->M_split >= M)
    error("hybrid_dgemm_Msplit -> Full CPU/GPU versions do not belong here");
  if (!A) error("hybrid_dgemm_Msplit -> A is not malloc'ed correctly");
  if (!B) error("hybrid_dgemm_Msplit -> B is not malloc'ed correctly");
  if (beta != 0 && !C)
    error("hybrid_dgemm_Msplit -> C is not malloc'ed correctly");
  debug(
      "hybrid_dgemm_Msplit -> Trying your Matrix bounds (incomming "
      "segfaults)...");
  double test = A[M * K - 1];
  test = B[K * N - 1];
  if (beta != 0) test = C[M * N - 1];
  debug("hybrid_dgemm_Msplit -> Passed.");
  double *C_out, local_t;

  double *A_cpu, *A_gpu, *B_cpu, *B_gpu, *C_cpu, *C_gpu;

  double *A_T, *B_T, *C_T, *C_buffer, cpu_beta = 0, gpu_beta = 0;

  size_t M_gpu = pred->M_split, M_cpu = M - pred->M_split, ldA = 0, ldB = 0,
         ldC = 0, d_ldA = 0, d_ldB = 0, d_ldC = 0;
  cublasOperation_t gpu_op_A, gpu_op_B;  // CUBLAS_OP_N, CUBLAS_OP_T
  CBLAS_TRANSPOSE cpu_op_A, cpu_op_B;    // CblasNoTrans, CblasTrans

  cublasHandle_t handle;
  cublasStatus_t stat = cublasCreate(&handle);

  gpu_timer_p cuda_timer = gpu_timer_init();

  /// Setup A parts on host and device

  switch (ctrl->A_mem) {
    case (ROW_MAJOR): 
      local_t = csecond();
      A_gpu = Dvec_transfer_gpu(A, M_gpu * K);
      local_t = csecond() - local_t;
      ctrl->scatter_t += local_t;
      gpu_op_A = CUBLAS_OP_T;
      d_ldA = K;
      A_cpu = &(A[M_gpu * K]);
      ldA = K;
      if (pred->cblas_target == CblasRowMajor)
        cpu_op_A = CblasNoTrans;

      else if (pred->cblas_target == CblasColMajor) {
        debug(
            "hybrid_dgemm_Msplit -> pred->cblas_target == CblasColMajor "
            "Untested "
            "diamond");
        cpu_op_A = CblasTrans;
      }
     break;
    case (COL_MAJOR): 
      error(
          "hybrid_dgemm_Msplit -> ctrl->A_mem == COL_MAJOR Unimplemented "
          "diamond");
     break;
    default:
      error("hybrid_dgemm_Msplit -> Unreachable default reached ");
  }

  /// Setup B parts on host and device
  B_cpu = B;
  local_t = csecond();
  B_gpu = Dvec_transfer_gpu(B, K * N);
  local_t = csecond() - local_t;
  ctrl->scatter_t += local_t;

  switch (ctrl->B_mem) {
    case (ROW_MAJOR):
      gpu_op_B = CUBLAS_OP_T;
      d_ldB = N;
      ldB = N;
      if (pred->cblas_target == CblasRowMajor)
        cpu_op_B = CblasNoTrans;
      else if (pred->cblas_target == CblasColMajor)
        cpu_op_B = CblasTrans;
     break;
    case (COL_MAJOR): 
      gpu_op_B = CUBLAS_OP_N;
      d_ldB = K;
      ldB = K;
      if (pred->cblas_target == CblasRowMajor)
        cpu_op_B = CblasTrans;
      else if (pred->cblas_target == CblasColMajor)
        cpu_op_B = CblasNoTrans;
     break;
    default:
      error("hybrid_dgemm_Msplit -> Unreachable default reached ");
  }

  /// Setup C parts on host and device

  if (!beta) {
    local_t = csecond();
    C_gpu = (double *)gpu_malloc(M_gpu * N * sizeof(double));
    if (!C)
      C_cpu = (double *)pin_malloc(M_gpu * N * sizeof(double));
    else
      C_cpu = &(C[M_gpu * N]);
    local_t = csecond() - local_t;
    ctrl->pin_alloc_t += local_t;
    d_ldC = M_gpu;
    if (pred->cblas_target == CblasRowMajor)
      ldC = N;
    else if (pred->cblas_target == CblasColMajor)
      ldC = M_cpu;

  } else if (pred->Cadd_device_id == -1) {
    cpu_beta = gpu_beta = beta;
    if (ctrl->C_mem == ROW_MAJOR) {
      C_cpu = &(C[M_gpu * N]);
      if (pred->cblas_target == CblasRowMajor)
        ldC = N;
      else if (pred->cblas_target == CblasColMajor) {
        error(
            "hybrid_dgemm_Msplit -> pred->cblas_target == CblasColMajor "
            "and pred->C_mem == ROW_MAJOR Unimplemented diamond");
      }
      local_t = csecond();
      C_T = (double *)pin_malloc(M_gpu * N * sizeof(double));
      C_buffer = C_T;
      local_t = csecond() - local_t;
      ctrl->pin_alloc_t += local_t;
      local_t = csecond();
      Dtranspose(C_T, C, M_gpu, N);
      local_t = csecond() - local_t;
      ctrl->transpose_t += local_t;
      local_t = csecond();
      C_gpu = Dvec_transfer_gpu(C_T, M_gpu * N);
      local_t = csecond() - local_t;
      ctrl->scatter_t += local_t;
      d_ldC = M_gpu;
    } else if (ctrl->C_mem == COL_MAJOR) {
      C_cpu = &(C[M_gpu]);
      if (pred->cblas_target == CblasRowMajor) {
        error(
            "hybrid_dgemm_Msplit -> pred->cblas_target == CblasRowMajor and "
            "pred->C_mem == COL_MAJOR Unimplemented diamond");
      } else if (pred->cblas_target == CblasColMajor)
        ldC = M;
      debug(
          "hybrid_dgemm_Msplit -> pred->cblas_target == CblasColMajor "
          "Untested diamond");
      local_t = csecond();
      C_gpu = Dvec_chunk_transfer_gpu(C, N, M_gpu, M);
      local_t = csecond() - local_t;
      ctrl->scatter_t += local_t;
      d_ldC = M_gpu;
    }

  } else if (pred->Cadd_device_id == 0) {
    error(
        "hybrid_dgemm_Msplit -> pred->Cadd_device_id == 0 Unimplemented "
        "diamond");
    /*
        gpu_beta = beta;
        switch (pred->C_mem) {
          case (ROW_MAJOR_CPU):
            C_cpu = &(C[M_gpu * N]);
            local_t = csecond();
            C_buffer = (double *)pin_malloc(M_gpu * N * sizeof(double));
            local_t = csecond() - local_t;
            ctrl->pin_alloc_t += local_t;
            local_t = csecond();
            C_gpu = (double *)gpu_malloc(M_gpu * N * sizeof(double));
            local_t = csecond() - local_t;
            ctrl->scatter_t += local_t;
            break;
          case (COL_MAJOR_CPU):
            error(
                "hybrid_dgemm_Msplit -> Using COL_MAJOR_CPU C... not "
                "implemented");
            break;
          case (ROW_MAJOR_GPU):
            error(
                "hybrid_dgemm_Msplit -> Using ROW_MAJOR_GPU C... not "
                "implemented");
            break;
          case (COL_MAJOR_GPU):
            error(
                "hybrid_dgemm_Msplit -> Using COL_MAJOR_GPU C... not "
                "implemented");
            break;
          default:
            error("hybrid_dgemm_Msplit -> A Unknown mem type");
        }
    */
  } else if (pred->Cadd_device_id == 1) {
    error(
        "hybrid_dgemm_Msplit -> pred->Cadd_device_id == 1 Unimplemented "
        "diamond");
    /*
        cpu_beta = beta;
        switch (pred->C_mem) {
          case (ROW_MAJOR_CPU):
            C_cpu = &(C[M_gpu * N]);
            local_t = csecond();
            C_T = (double *)pin_malloc(M_gpu * N * sizeof(double));
            C_buffer = C_T;
            local_t = csecond() - local_t;
            ctrl->pin_alloc_t += local_t;
            local_t = csecond();
            Dtranspose(C_T, C, M_gpu, N);
            local_t = csecond() - local_t;
            ctrl->transpose_t += local_t;
            local_t = csecond();
            C_gpu = Dvec_transfer_gpu(C_T, M_gpu * N);
            local_t = csecond() - local_t;
            ctrl->scatter_t += local_t;
            break;
          case (COL_MAJOR_CPU):
            error(
                "hybrid_dgemm_Msplit -> Using COL_MAJOR_CPU C... not "
                "implemented");
            break;
          case (ROW_MAJOR_GPU):
            error(
                "hybrid_dgemm_Msplit -> Using ROW_MAJOR_GPU C... not "
                "implemented");
            break;
          case (COL_MAJOR_GPU):
            error(
                "hybrid_dgemm_Msplit -> Using COL_MAJOR_GPU C... not "
                "implemented");
            break;
          default:
            error("hybrid_dgemm_Msplit -> A Unknown mem type");
        }
    */
  }

   if (!ldA || !ldB || !ldC || !d_ldA || !d_ldB || !d_ldC)
    error(
        "hybrid_dgemm_Msplit -> Some ld_dim were not defined correctly (=0)");

  gpu_timer_start(cuda_timer);

  stat = cublasDgemm(handle, gpu_op_A, gpu_op_B, M_gpu, N, K, &alpha, A_gpu,
                     d_ldA, B_gpu, d_ldB, &beta, C_gpu, d_ldC);
  gpu_timer_stop(cuda_timer);
  local_t = csecond();

  cblas_dgemm(pred->cblas_target, cpu_op_A, cpu_op_B, M_cpu, N, K, alpha, A_cpu,
              ldA, B_cpu, ldB, beta, C_cpu, ldC);
  local_t = csecond() - local_t;
  ctrl->cpu_ex_t = local_t;
  cudaCheckErrors();

  ctrl->gpu_ex_t = (double)gpu_timer_get(cuda_timer) / 1000;

  if (ctrl->C_mem == ROW_MAJOR) {
    if (pred->cblas_target == CblasRowMajor) {
      local_t = csecond();
      cudaMemcpy(C_buffer, C_gpu, M_gpu * N * sizeof(double),
                 cudaMemcpyDeviceToHost);
      local_t = csecond() - local_t;
      ctrl->gather_t = local_t;
      local_t = csecond();
      Dtranspose(C, C_buffer, N, M_gpu);
      local_t = csecond() - local_t;
      ctrl->transpose_t += local_t;
    } else if (pred->cblas_target == CblasColMajor) {
      error(
          "hybrid_dgemm_Msplit -> pred->cblas_target == CblasColMajor and "
          "pred->C_mem == ROW_MAJOR Unimplemented diamond");
    }
  } else if (ctrl->C_mem == COL_MAJOR) {
    if (pred->cblas_target == CblasRowMajor) {
      error(
          "hybrid_dgemm_Msplit -> pred->cblas_target == CblasRowMajor and "
          "pred->C_mem == COL_MAJOR Unimplemented diamond");
    }
    if (pred->cblas_target == CblasColMajor) {
      local_t = csecond();
      Dvec_chunk_copy_from_gpu(C, C_gpu, N, M_gpu, M);
      local_t = csecond() - local_t;
      ctrl->gather_t = local_t;
    }
  }

  printf(
      "Device overhead(M=%d, N=%d, K=%d) pin_alloc = %lf ms, scatter = %lf "
      "ms, "
      "transpose = %lf ms, gather = %lf ms, reduce = %lf ms\n",
      M, N, K, 1000 * ctrl->pin_alloc_t, 1000 * ctrl->scatter_t,
      1000 * ctrl->transpose_t, 1000 * ctrl->gather_t, 1000 * ctrl->reduce_t);

  printf("Hybrid Sgemm(M_cpu=%d, N=%d, K=%d) CPU time = ", M_cpu, N, K);
  report_results(ctrl->cpu_ex_t, (long)M_cpu * K * (2 * N + 1),
                 (long)(M_cpu * K + K * N + M_cpu * N * 2) *
                     sizeof(double));  //(M*N+(long)M*K*(3*N+1))
  printf("\n");

  printf("Hybrid Sgemm(M_gpu=%d, N=%d, K=%d) GPU time = ", M_gpu, N, K);
  report_results(ctrl->gpu_ex_t, (long)M_gpu * K * (2 * N + 1),
                 (long)(M_gpu * K + K * N + M_gpu * N * 2) *
                     sizeof(double));  //(M*N+(long)M*K*(3*N+1))
  printf("\n");

  pin_free(C_buffer);

  // FIXME: implement correct frees with switches
  gpu_free(A_gpu);
  gpu_free(B_gpu);
  gpu_free(C_gpu);

  C_out = C;
  debug("<- hybrid_dgemm_Msplit()\n");
  return C_out;
}

int main(int argc, char *argv[]) {
  // print_devices();

  /*
    double *test, *test_T;

    test = Dvec_init_pinned(25, 42);
    test_T = Dvec_init_pinned(25, 0);

    for (int i = 0; i <5; i++){
            for (int j = 0; j <5; j++) printf("%0.3lf ", test[5*i +j]);
            printf("\n");
    }
    printf("\n");
    printf("\n");

    int s1 = 2, s2 = 5;
    Dtranspose(test_T, test, s1, s2);

    for (int i = 0; i <s1; i++){
            for (int j = 0; j <s2; j++) printf("%0.3lf ", test[s2*i +j]);
            printf("\n");
    }
    printf("\n");
    for (int i = 0; i <s1*s2; i++)printf("%0.3lf ", test_T[i]);
    printf("\n");


    exit(1);
  */

  int M = 3000, K = 2000, N = 2500;
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
  int M_split = M * 2 / 5, N_split = 0, K_split = 0;

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

  predict_p main_pred = (predict_p)malloc(sizeof(struct pred_str));
  control_p main_ctrl = (control_p)malloc(sizeof(struct control_str));

  main_ctrl->num_devices = 2;
  main_pred->M_split = M_split;
  main_ctrl->A_mem = ROW_MAJOR;
main_ctrl->B_mem = ROW_MAJOR;
main_ctrl->C_mem = ROW_MAJOR;
  main_pred->cblas_target = CblasRowMajor;
  main_pred->Cadd_device_id = -1;
  if (M_split) {
    if (M_split == M) {
      debug("executing solely on GPU");
      if (ghost_beta) {
        debug("executing with ghost_beta !=0");
        debug(
            "this is literally offloading naivelly the whole thing on "
            "CUBLAS");

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
            "Device overhead(M=%d, N=%d, K=%d) transpose = %lf ms, t_preproc "
            "= "
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
            "Device overhead(M=%d, N=%d, K=%d) transpose = %lf ms, t_preproc "
            "= "
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
      C = hybrid_dgemm_Msplit(main_ctrl, main_pred, M, N, K, alpha, A, B, beta,
                              C);

      printf("Total Sgemm(M=%d, N=%d, K=%d) ", M, N, K);
      report_results(fmax(main_ctrl->cpu_ex_t, main_ctrl->gpu_ex_t) +
                         main_ctrl->scatter_t + main_ctrl->pin_alloc_t +
                         main_ctrl->transpose_t + main_ctrl->gather_t +
                         main_ctrl->reduce_t,
                     (long)M * K * (2 * N + 1),
                     (long)(M * K + K * N + M * N * 2) *
                         sizeof(double));  //(M*N+(long)M*K*(3*N+1))
      printf("\n");

      Dtest_equality(C_comp, C, M * N);
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
            "Device overhead(M=%d, N=%d, K=%d) transpose = %lf ms, t_preproc "
            "= "
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
        pin_free(B_T);
        pin_free(C_T);
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
            "Device overhead(M=%d, N=%d, K=%d) transpose = %lf ms, t_preproc "
            "= "
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
        pin_free(B_T);
        pin_free(C_T);
      }
    }
  } else if (K_split) {
    if (K_split == K) {
      debug("executing solely on GPU but with Κ spliting (?)");
      debug("Not gonna copy the same code, a goto is preffered");
      K_split = 0;
      M_split = M;
      goto return_point;
    } else {
      debug("executing hybrid GPU-CPU (K)");
      size_t K_gpu = K_split, K_cpu = K - K_split;
      if (ghost_beta) {
        debug("executing with ghost_beta !=0");
        debug("...you would think but that would calculate C = aAB + 2bC");
        debug("our friend goto will save us");
        goto just_bellow;
      } else {
      just_bellow:
        debug("executing with ghost_beta = 0");
        gpu_timer_start(cuda_timer);
        double dev_beta = 0.0;

        transpose_timer = csecond();
        double *A_T = (double *)pin_malloc(M * K * sizeof(double));
        Dtranspose(A_T, A, M, K);
        double *reduce_C;
        transpose_timer = csecond() - transpose_timer;

        gpu_timer_start(cuda_timer);
        d_A = Dvec_transfer_gpu(A_T, M * K_gpu);
        d_B = Dvec_transfer_gpu(B, K_gpu * N);
        d_C = (double *)gpu_malloc(M * N * sizeof(double));
        gpu_timer_stop(cuda_timer);
        gpu_preproc_t = gpu_timer_get(cuda_timer);

        gpu_timer_start(cuda_timer);

        stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, M, N, K_gpu,
                           &alpha, d_A, M, d_B, N, &dev_beta, d_C, M);
        reduce_C = (double *)pin_malloc(M * N * sizeof(double));
        cpu_timer = csecond();
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, N, K_cpu, alpha,
                    &(A_T[M * K_gpu]), M, &(B[N * K_gpu]), N, beta, C, N);
        cpu_timer = csecond() - cpu_timer;
        cudaCheckErrors();

        gpu_timer_stop(cuda_timer);
        gpu_comp_t = gpu_timer_get(cuda_timer);
        gpu_timer_start(cuda_timer);
        cudaMemcpy(reduce_C, d_C, M * N * sizeof(double),
                   cudaMemcpyDeviceToHost);
        gpu_timer_stop(cuda_timer);
        gpu_reduce_t = gpu_timer_get(cuda_timer);

        transpose_timer = csecond() - transpose_timer;
        Dtranspose_add(C, reduce_C, N, M);
        transpose_timer = csecond() - transpose_timer;

        printf(
            "Device overhead(M=%d, N=%d, K=%d) transpose = %lf ms, t_preproc "
            "= "
            "%lf ms, t_reduce = %lf ms\n",
            M, N, K, 1000 * transpose_timer, gpu_preproc_t, gpu_reduce_t);

        printf("Hybrid Sgemm(M=%d, N=%d, K_cpu=%d) CPU time = ", M, N, K_cpu);
        report_results((double)cpu_timer, (long)M * K_cpu * (2 * N + 1),
                       (long)(M * K_cpu + K_cpu * N + M * N * 2) *
                           sizeof(double));  //(M*N+(long)M*K_cpu*(3*N+1))
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
        pin_free(A_T);
        pin_free(reduce_C);
      }
    }
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
