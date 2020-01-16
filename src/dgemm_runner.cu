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

enum mem_layout { ROW_MAJOR = 0, COL_MAJOR };

const char *print_mem(mem_layout mem) {
  if (mem == ROW_MAJOR)
    return "Row major";
  else if (mem == COL_MAJOR)
    return "Col major";
  else
    return "ERROR";
}

typedef struct control_str {
  // For now assume device 0 is always the host
  size_t num_devices;
  mem_layout A_mem;
  mem_layout B_mem;
  mem_layout C_mem;
  double alloc_t = 0, scatter_t = 0, transpose_t = 0, cpu_ex_t = 0,
         gpu_ex_t = 0, gather_t = 0, reduce_t = 0;

} * control_p;

typedef struct pred_str {
  size_t M_split;
  size_t N_split;
  size_t K_split;
  size_t Cadd_device_id;
  // TODO: Not implemented
  int asynch_trans = 0;

} * predict_p;

double *gpu_dgemm(control_p ctrl, predict_p pred, size_t M, size_t N, size_t K,
                  double alpha, double *A, double *B, double beta, double *C) {
  debug("-> gpu_dgemm()");
  if (ctrl->num_devices < 1)
    error(
        "gpu_dgemm -> 0 or less devices? What are you trying to "
        "do...");
  else if (ctrl->num_devices > 2)
    error(
        "gpu_dgemm -> Max 1 GPU + 1 CPU implemented (nice "
        "try,though).");

  if (pred->asynch_trans)
    error(
        "gpu_dgemm -> asynch transactions not implemented yet "
        "(nice try,though).");

  if (!A) error("gpu_dgemm -> A is not malloc'ed correctly");
  if (!B) error("gpu_dgemm -> B is not malloc'ed correctly");
  if (beta != 0 && !C) error("gpu_dgemm -> C is not malloc'ed correctly");
  debug(
      "gpu_dgemm -> Trying your Matrix bounds (incomming "
      "segfaults)...");
  double test = A[M * K - 1];
  test = B[K * N - 1];
  if (beta != 0) test = C[M * N - 1];
  debug("gpu_dgemm -> Passed.");
  double *C_out, local_t;

  double *A_gpu, *B_gpu, *C_gpu;

  double *C_T, cpu_beta = 0, gpu_beta = 0;

  size_t M_gpu = pred->M_split, ldA = 0, ldB = 0, ldC = 0, d_ldA = 0, d_ldB = 0,
         d_ldC = 0;
  cublasOperation_t gpu_op_A, gpu_op_B;  // CUBLAS_OP_N, CUBLAS_OP_T
  CBLAS_TRANSPOSE cpu_op_A, cpu_op_B;    // CblasNoTrans, CblasTrans

  cublasHandle_t handle;
  cublasStatus_t stat = cublasCreate(&handle);

  gpu_timer_p cuda_timer = gpu_timer_init();

  if (ctrl->A_mem == COL_MAJOR)
    ldA = M;
  else
    ldA = K;
  if (ctrl->B_mem == COL_MAJOR)
    ldB = K;
  else
    ldB = N;

  if (ctrl->C_mem == COL_MAJOR) {
    ldC = M;
    if (ctrl->A_mem == COL_MAJOR)
      gpu_op_A = CUBLAS_OP_N;
    else
      gpu_op_A = CUBLAS_OP_T;
    if (ctrl->B_mem == COL_MAJOR)
      gpu_op_B = CUBLAS_OP_N;
    else
      gpu_op_B = CUBLAS_OP_T;
  } else {
    ldC = N;
    if (ctrl->A_mem == ROW_MAJOR)
      gpu_op_A = CUBLAS_OP_T;
    else
      gpu_op_A = CUBLAS_OP_N;

    if (ctrl->B_mem == ROW_MAJOR)
      gpu_op_B = CUBLAS_OP_T;
    else
      gpu_op_B = CUBLAS_OP_N;
  }

  local_t = csecond();
  A_gpu = Dvec_transfer_gpu(A, M * K);
  B_gpu = Dvec_transfer_gpu(B, K * N);
  local_t = csecond() - local_t;
  ctrl->scatter_t += local_t;

  local_t = csecond();
  C_T = (double *)pin_malloc(M * N * sizeof(double));
  local_t = csecond() - local_t;
  ctrl->alloc_t += local_t;

  if (pred->Cadd_device_id == -1) {
    gpu_beta = beta;
    local_t = csecond();
    if (ctrl->C_mem == ROW_MAJOR) Dtranspose(C_T, C, M, N);

    local_t = csecond() - local_t;
    ctrl->transpose_t += local_t;

    local_t = csecond();
    C_gpu = Dvec_transfer_gpu(C_T, M * N);
    local_t = csecond() - local_t;
    ctrl->scatter_t += local_t;
  } else if (pred->Cadd_device_id == 0) {
    local_t = csecond();
    C_gpu = (double *)gpu_malloc(M * N * sizeof(double));
    local_t = csecond() - local_t;
    ctrl->alloc_t += local_t;
  }

  else if (pred->Cadd_device_id == 1) {
    debug(
        "gpu_dgemm -> pred->Cadd_device_id == 1 is obsolete..all computations "
        "on gpu anyway");

    gpu_beta = beta;
    local_t = csecond();
    if (ctrl->C_mem == ROW_MAJOR) Dtranspose(C_T, C, M, N);

    local_t = csecond() - local_t;
    ctrl->transpose_t += local_t;

    local_t = csecond();
    C_gpu = Dvec_transfer_gpu(C_T, M * N);
    local_t = csecond() - local_t;
    ctrl->scatter_t += local_t;
  }

  gpu_timer_start(cuda_timer);
  stat = cublasDgemm(handle, gpu_op_A, gpu_op_B, M, N, K, &alpha, A_gpu, ldA,
                     B_gpu, ldB, &gpu_beta, C_gpu, M);

  gpu_timer_stop(cuda_timer);

  if (pred->Cadd_device_id == 0) cblas_dscal(M * N, beta, C, 1);

  cudaCheckErrors();
  ctrl->gpu_ex_t += gpu_timer_get(cuda_timer) / 1000;

  if (ctrl->C_mem == ROW_MAJOR) {
    local_t = csecond();
    cudaMemcpy(C_T, C_gpu, M * N * sizeof(double), cudaMemcpyDeviceToHost);
    local_t = csecond() - local_t;
    ctrl->gather_t += local_t;

    local_t = csecond();
    if (pred->Cadd_device_id == 0)
      Dtranspose_add(C, C_T, N, M);
    else
      Dtranspose(C, C_T, N, M);
    local_t = csecond() - local_t;
    ctrl->transpose_t += local_t;
  } else {
    local_t = csecond();
    if (pred->Cadd_device_id == 0) {
      local_t = csecond();
      cudaMemcpy(C_T, C_gpu, M * N * sizeof(double), cudaMemcpyDeviceToHost);
      local_t = csecond() - local_t;
      ctrl->gather_t += local_t;
      local_t = csecond();
      cblas_daxpy(N * M, 1.0, C_T, 1, C, 1);
      local_t = csecond() - local_t;
      ctrl->reduce_t += local_t;
    } else {
      local_t = csecond();
      cudaMemcpy(C, C_gpu, M * N * sizeof(double), cudaMemcpyDeviceToHost);
      local_t = csecond() - local_t;
      ctrl->gather_t += local_t;
    }
  }

  // FIXME: implement correct frees with switches
  gpu_free(A_gpu);
  gpu_free(B_gpu);
  gpu_free(C_gpu);
  pin_free(C_T);

  C_out = C;
  debug("<- gpu_dgemm()\n");
  return C_out;
}

double *hybrid_dgemm_Msplit(control_p ctrl, predict_p pred, size_t M, size_t N,
                            size_t K, double alpha, double *A, double *B,
                            double beta, double *C) {
  debug("-> hybrid_dgemm_Msplit()");
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
  if (beta || C) test = C[M * N - 1];
  debug("hybrid_dgemm_Msplit -> Passed.");
  double *C_out, local_t;

  double *A_cpu, *A_gpu, *B_cpu, *B_gpu, *C_cpu, *C_gpu;

  double *C_buffer, cpu_beta = 0, gpu_beta = 0;

  size_t M_gpu = pred->M_split, M_cpu = M - pred->M_split, ldA = 0, ldB = 0,
         ldC = 0, d_ldA = 0, d_ldB = 0, d_ldC = 0;
  cublasOperation_t gpu_op_A, gpu_op_B;  // CUBLAS_OP_N, CUBLAS_OP_T
  CBLAS_TRANSPOSE cpu_op_A, cpu_op_B;    // CblasNoTrans, CblasTrans

  CBLAS_LAYOUT cblas_target;
  cublasHandle_t handle;
  cublasStatus_t stat = cublasCreate(&handle);

  gpu_timer_p cuda_timer = gpu_timer_init();

  /// Setup A parts on host and device
  local_t = csecond();
  switch (ctrl->A_mem) {
    case (ROW_MAJOR):
      A_gpu = Dvec_transfer_gpu(A, M_gpu * K);
      gpu_op_A = CUBLAS_OP_T;
      d_ldA = K;
      A_cpu = &(A[M_gpu * K]);
      ldA = K;
      if (ctrl->C_mem == ROW_MAJOR)
        cpu_op_A = CblasNoTrans;
      else if (ctrl->C_mem == COL_MAJOR)
        cpu_op_A = CblasTrans;
      break;
    case (COL_MAJOR):
      gpu_op_A = CUBLAS_OP_N;
      d_ldA = M_gpu;
      A_cpu = &(A[M_gpu]);
      ldA = M;
      if (ctrl->C_mem == ROW_MAJOR)
        cpu_op_A = CblasTrans;
      else if (ctrl->C_mem == COL_MAJOR)
        cpu_op_A = CblasNoTrans;
      A_gpu = Dvec_chunk_transfer_gpu(A, K, M_gpu, M);
      break;
    default:
      error("hybrid_dgemm_Msplit -> Unreachable default reached ");
  }
  local_t = csecond() - local_t;
  ctrl->scatter_t += local_t;

  /// Setup B parts on host and device
  local_t = csecond();
  B_cpu = B;
  B_gpu = Dvec_transfer_gpu(B, K * N);
  switch (ctrl->B_mem) {
    case (ROW_MAJOR):
      gpu_op_B = CUBLAS_OP_T;
      d_ldB = N;
      ldB = N;
      if (ctrl->C_mem == ROW_MAJOR)
        cpu_op_B = CblasNoTrans;
      else if (ctrl->C_mem == COL_MAJOR)
        cpu_op_B = CblasTrans;
      break;
    case (COL_MAJOR):
      gpu_op_B = CUBLAS_OP_N;
      d_ldB = K;
      ldB = K;
      if (ctrl->C_mem == ROW_MAJOR)
        cpu_op_B = CblasTrans;
      else if (ctrl->C_mem == COL_MAJOR)
        cpu_op_B = CblasNoTrans;
      break;
    default:
      error("hybrid_dgemm_Msplit -> Unreachable default reached ");
  }
  local_t = csecond() - local_t;
  ctrl->scatter_t += local_t;

  /// Setup C parts on host and device
  d_ldC = M_gpu;

  if (!beta || !pred->Cadd_device_id) {
    cpu_beta = beta;
    local_t = csecond();
    C_gpu = (double *)gpu_malloc(M_gpu * N * sizeof(double));
    C_buffer = (double *)pin_malloc(M_gpu * N * sizeof(double));
    if (!C) C = (double *)malloc(M * N * sizeof(double));

    local_t = csecond() - local_t;
    ctrl->alloc_t += local_t;
  } else if (pred->Cadd_device_id == -1) {
    cpu_beta = gpu_beta = beta;
    if (ctrl->C_mem == ROW_MAJOR) {
      local_t = csecond();
      C_buffer = (double *)pin_malloc(M_gpu * N * sizeof(double));
      local_t = csecond() - local_t;
      ctrl->alloc_t += local_t;
      local_t = csecond();
      Dtranspose(C_buffer, C, M_gpu, N);
      local_t = csecond() - local_t;
      ctrl->transpose_t += local_t;
      local_t = csecond();
      C_gpu = Dvec_transfer_gpu(C_buffer, M_gpu * N);
      local_t = csecond() - local_t;
      ctrl->scatter_t += local_t;
    } else if (ctrl->C_mem == COL_MAJOR) {
      local_t = csecond();
      C_buffer = (double *)pin_malloc(1 * sizeof(double));  /// Dummy for free
      C_gpu = Dvec_chunk_transfer_gpu(C, N, M_gpu, M);
      local_t = csecond() - local_t;
      ctrl->scatter_t += local_t;
    }

  } else if (pred->Cadd_device_id == 1) {
    error("hybrid_dgemm_Msplit -> pred->Cadd_device_id == 1 Unimplemented.");
  }

  if (ctrl->C_mem == ROW_MAJOR) {
    cblas_target = CblasRowMajor;
    C_cpu = &(C[M_gpu * N]);
    ldC = N;
  } else if (ctrl->C_mem == COL_MAJOR) {
    C_cpu = &(C[M_gpu]);
    ldC = M;
    cblas_target = CblasColMajor;
  }

  if (!ldA || !ldB || !ldC || !d_ldA || !d_ldB || !d_ldC)
    error("hybrid_dgemm_Msplit -> Some ld_dim were not defined correctly (=0)");

  gpu_timer_start(cuda_timer);

  stat = cublasDgemm(handle, gpu_op_A, gpu_op_B, M_gpu, N, K, &alpha, A_gpu,
                     d_ldA, B_gpu, d_ldB, &gpu_beta, C_gpu, d_ldC);
  gpu_timer_stop(cuda_timer);

  if (pred->Cadd_device_id == 0) {
    local_t = csecond();
    switch (ctrl->C_mem) {
      case (ROW_MAJOR):
        cblas_dscal(N * M_gpu, beta, C, 1);
        break;
      case (COL_MAJOR):
        for (int i = 0; i < N; i++)
          cblas_daxpy(M_gpu, beta, &C[i * M], 1, &C_buffer[i * M_gpu], 1);
        break;
      default:
        error("hybrid_dgemm_Msplit -> A Unknown mem type");
    }
    local_t = csecond() - local_t;
    ctrl->reduce_t += local_t;
  }

  local_t = csecond();
  cblas_dgemm(cblas_target, cpu_op_A, cpu_op_B, M_cpu, N, K, alpha, A_cpu, ldA,
              B_cpu, ldB, cpu_beta, C_cpu, ldC);
  local_t = csecond() - local_t;
  ctrl->cpu_ex_t += local_t;
  cudaCheckErrors();

  ctrl->gpu_ex_t += (double)gpu_timer_get(cuda_timer) / 1000;

  if (ctrl->C_mem == ROW_MAJOR) {
    local_t = csecond();
    cudaMemcpy(C_buffer, C_gpu, M_gpu * N * sizeof(double),
               cudaMemcpyDeviceToHost);
    local_t = csecond() - local_t;
    ctrl->gather_t += local_t;
    local_t = csecond();
    if (pred->Cadd_device_id == 0)
      Dtranspose_add(C, C_buffer, N, M_gpu);
    else
      Dtranspose(C, C_buffer, N, M_gpu);
    local_t = csecond() - local_t;
    ctrl->transpose_t += local_t;

  } else if (ctrl->C_mem == COL_MAJOR) {
    local_t = csecond();
    Dvec_chunk_copy_from_gpu(C, C_gpu, N, M_gpu, M);

    if (pred->Cadd_device_id == 0)
      for (int i = 0; i < N; i++)
        cblas_daxpy(M_gpu, 1.0, &C_buffer[i * M_gpu], 1, &C[i * M], 1);
    local_t = csecond() - local_t;
    ctrl->gather_t += local_t;
  }

  // FIXME: implement correct frees with switches
  gpu_free(A_gpu);
  gpu_free(B_gpu);
  gpu_free(C_gpu);
  pin_free(C_buffer);

  C_out = C;
  debug("<- hybrid_dgemm_Msplit()\n");
  return C_out;
}

double *hybrid_dgemm_Nsplit(control_p ctrl, predict_p pred, size_t M, size_t N,
                            size_t K, double alpha, double *A, double *B,
                            double beta, double *C) {
  debug("-> hybrid_dgemm_Nsplit()");
  if (ctrl->num_devices < 1)
    error(
        "hybrid_dgemm_Nsplit -> 0 or less devices? What are you trying to "
        "do...");
  else if (ctrl->num_devices > 2)
    error(
        "hybrid_dgemm_Nsplit -> Max 1 GPU + 1 CPU implemented (nice "
        "try,though).");

  if (pred->asynch_trans)
    error(
        "hybrid_dgemm_Nsplit -> asynch transactions not implemented yet "
        "(nice try,though).");

  if (!pred->N_split || pred->N_split >= N)
    error("hybrid_dgemm_Nsplit -> Full CPU/GPU versions do not belong here");
  if (!A) error("hybrid_dgemm_Nsplit -> A is not malloc'ed correctly");
  if (!B) error("hybrid_dgemm_Nsplit -> B is not malloc'ed correctly");
  if (beta != 0 && !C)
    error("hybrid_dgemm_Nsplit -> C is not malloc'ed correctly");
  debug(
      "hybrid_dgemm_Nsplit -> Trying your Matrix bounds (incomming "
      "segfaults)...");
  double test = A[M * K - 1];
  test = B[K * N - 1];
  if (beta || C) test = C[M * N - 1];
  debug("hybrid_dgemm_Nsplit -> Passed.");
  double *C_out, local_t;

  double *A_cpu, *A_gpu, *B_cpu, *B_gpu, *C_cpu, *C_gpu;

  double *C_buffer, cpu_beta = 0, gpu_beta = 0;

  size_t N_gpu = pred->N_split, N_cpu = N - pred->N_split, ldA = 0, ldB = 0,
         ldC = 0, d_ldA = 0, d_ldB = 0, d_ldC = 0;
  cublasOperation_t gpu_op_A, gpu_op_B;  // CUBLAS_OP_N, CUBLAS_OP_T
  CBLAS_TRANSPOSE cpu_op_A, cpu_op_B;    // CblasNoTrans, CblasTrans

  CBLAS_LAYOUT cblas_target;
  cublasHandle_t handle;
  cublasStatus_t stat = cublasCreate(&handle);

  gpu_timer_p cuda_timer = gpu_timer_init();

  /// Setup A parts on host and device
  local_t = csecond();
  A_cpu = A;
  A_gpu = Dvec_transfer_gpu(A, M * K);

  switch (ctrl->A_mem) {
    case (ROW_MAJOR):
      gpu_op_A = CUBLAS_OP_T;
      d_ldA = K;
      ldA = K;
      if (ctrl->C_mem == ROW_MAJOR)
        cpu_op_A = CblasNoTrans;
      else if (ctrl->C_mem == COL_MAJOR)
        cpu_op_A = CblasTrans;
      break;
    case (COL_MAJOR):
      gpu_op_A = CUBLAS_OP_N;
      d_ldA = M;
      ldA = M;
      if (ctrl->C_mem == ROW_MAJOR)
        cpu_op_A = CblasTrans;
      else if (ctrl->C_mem == COL_MAJOR)
        cpu_op_A = CblasNoTrans;
      break;
    default:
      error("hybrid_dgemm_Nsplit -> Unreachable default reached ");
  }
  local_t = csecond() - local_t;
  ctrl->scatter_t += local_t;

  /// Setup B parts on host and device
  local_t = csecond();
  switch (ctrl->B_mem) {
    case (ROW_MAJOR):
      gpu_op_B = CUBLAS_OP_T;
      d_ldB = N_gpu;
      B_cpu = &(B[N_gpu]);
      ldB = N;
      if (ctrl->C_mem == ROW_MAJOR)
        cpu_op_B = CblasNoTrans;
      else if (ctrl->C_mem == COL_MAJOR)
        cpu_op_B = CblasTrans;
      B_gpu = Dvec_chunk_transfer_gpu(B, K, N_gpu, N);
      break;
    case (COL_MAJOR):
      B_gpu = Dvec_transfer_gpu(B, N_gpu * K);
      gpu_op_B = CUBLAS_OP_N;
      d_ldB = K;
      B_cpu = &(B[N_gpu * K]);
      ldB = K;
      if (ctrl->C_mem == ROW_MAJOR)
        cpu_op_B = CblasTrans;
      else if (ctrl->C_mem == COL_MAJOR)
        cpu_op_B = CblasNoTrans;
      break;
    default:
      error("hybrid_dgemm_Nsplit -> Unreachable default reached ");
  }
  local_t = csecond() - local_t;
  ctrl->scatter_t += local_t;

  /// Setup C parts on host and device
  d_ldC = M;

  if (!beta || !pred->Cadd_device_id) {
    cpu_beta = beta;
    local_t = csecond();
    C_gpu = (double *)gpu_malloc(M * N_gpu * sizeof(double));
    C_buffer = (double *)pin_malloc(M * N_gpu * sizeof(double));
    if (!C) C = (double *)malloc(M * N * sizeof(double));

    local_t = csecond() - local_t;
    ctrl->alloc_t += local_t;
  } else if (pred->Cadd_device_id == -1) {
    cpu_beta = gpu_beta = beta;
    if (ctrl->C_mem == ROW_MAJOR) {
      local_t = csecond();
      C_buffer = (double *)pin_malloc(M * N_gpu * sizeof(double));
      local_t = csecond() - local_t;
      ctrl->alloc_t += local_t;
      local_t = csecond();
      Dtranspose_stride_src(C_buffer, C, M, N_gpu, N);
      local_t = csecond() - local_t;
      ctrl->transpose_t += local_t;
      local_t = csecond();
      C_gpu = Dvec_transfer_gpu(C_buffer, M * N_gpu);
      local_t = csecond() - local_t;
      ctrl->scatter_t += local_t;
    } else if (ctrl->C_mem == COL_MAJOR) {
      local_t = csecond();
      C_buffer = (double *)pin_malloc(1 * sizeof(double));  /// Dummy for free;
      C_gpu = Dvec_transfer_gpu(C, N_gpu * M);
      local_t = csecond() - local_t;
      ctrl->scatter_t += local_t;
    }

  } else if (pred->Cadd_device_id == 1) {
    error("hybrid_dgemm_Nsplit -> pred->Cadd_device_id == 1 Unimplemented.");
  }

  if (ctrl->C_mem == ROW_MAJOR) {
    cblas_target = CblasRowMajor;
    C_cpu = &(C[N_gpu]);
    ldC = N;
  } else if (ctrl->C_mem == COL_MAJOR) {
    C_cpu = &(C[N_gpu * M]);
    ldC = M;
    cblas_target = CblasColMajor;
  }

  if (!ldA || !ldB || !ldC || !d_ldA || !d_ldB || !d_ldC)
    error("hybrid_dgemm_Nsplit -> Some ld_dim were not defined correctly (=0)");

  gpu_timer_start(cuda_timer);

  stat = cublasDgemm(handle, gpu_op_A, gpu_op_B, M, N_gpu, K, &alpha, A_gpu,
                     d_ldA, B_gpu, d_ldB, &gpu_beta, C_gpu, d_ldC);
  gpu_timer_stop(cuda_timer);

  if (pred->Cadd_device_id == 0) {
    local_t = csecond();
    switch (ctrl->C_mem) {
      case (ROW_MAJOR):
        for (int i = 0; i < M; i++) cblas_dscal(N_gpu, beta, &C[i * N], 1);
        break;
      case (COL_MAJOR):
        cblas_dscal(N_gpu * M, beta, C, 1);
        break;
      default:
        error("hybrid_dgemm_Nsplit -> A Unknown mem type");
    }
    local_t = csecond() - local_t;
    ctrl->reduce_t += local_t;
  }

  local_t = csecond();
  cblas_dgemm(cblas_target, cpu_op_A, cpu_op_B, M, N_cpu, K, alpha, A_cpu, ldA,
              B_cpu, ldB, cpu_beta, C_cpu, ldC);
  local_t = csecond() - local_t;
  ctrl->cpu_ex_t += local_t;
  cudaCheckErrors();

  ctrl->gpu_ex_t += (double)gpu_timer_get(cuda_timer) / 1000;

  if (ctrl->C_mem == ROW_MAJOR) {
    local_t = csecond();
    cudaMemcpy(C_buffer, C_gpu, M * N_gpu * sizeof(double),
               cudaMemcpyDeviceToHost);
    local_t = csecond() - local_t;
    ctrl->gather_t += local_t;
    local_t = csecond();
    if (pred->Cadd_device_id == 0)
      Dtranspose_stride_dest_add(C, C_buffer, N_gpu, M, N);
    else
      Dtranspose_stride_dest(C, C_buffer, N_gpu, M, N);
    local_t = csecond() - local_t;
    ctrl->transpose_t += local_t;

  } else if (ctrl->C_mem == COL_MAJOR) {
    local_t = csecond();
    if (pred->Cadd_device_id == 0) {
      cudaMemcpy(C_buffer, C_gpu, M * N_gpu * sizeof(double),
                 cudaMemcpyDeviceToHost);
      cblas_daxpy(M * N_gpu, 1.0, C_buffer, 1, C, 1);
    } else
      cudaMemcpy(C, C_gpu, M * N_gpu * sizeof(double), cudaMemcpyDeviceToHost);

    local_t = csecond() - local_t;
    ctrl->gather_t += local_t;
  }

  // FIXME: implement correct frees with switches
  gpu_free(A_gpu);
  gpu_free(B_gpu);
  gpu_free(C_gpu);
  pin_free(C_buffer);

  C_out = C;
  debug("<- hybrid_dgemm_Nsplit()\n");
  return C_out;
}

double *hybrid_dgemm_Ksplit(control_p ctrl, predict_p pred, size_t M, size_t N,
                            size_t K, double alpha, double *A, double *B,
                            double beta, double *C) {
  debug("-> hybrid_dgemm_Ksplit()");
  if (ctrl->num_devices < 1)
    error(
        "hybrid_dgemm_Ksplit -> 0 or less devices? What are you trying to "
        "do...");
  else if (ctrl->num_devices > 2)
    error(
        "hybrid_dgemm_Ksplit -> Max 1 GPU + 1 CPU implemented (nice "
        "try,though).");

  if (pred->asynch_trans)
    error(
        "hybrid_dgemm_Ksplit -> asynch transactions not implemented yet "
        "(nice try,though).");

  if (!pred->K_split || pred->K_split >= K)
    error("hybrid_dgemm_Ksplit -> Full CPU/GPU versions do not belong here");
  if (!A) error("hybrid_dgemm_Ksplit -> A is not malloc'ed correctly");
  if (!B) error("hybrid_dgemm_Ksplit -> B is not malloc'ed correctly");
  if (beta != 0 && !C)
    error("hybrid_dgemm_Ksplit -> C is not malloc'ed correctly");
  debug(
      "hybrid_dgemm_Ksplit -> Trying your Matrix bounds (incomming "
      "segfaults)...");
  double test = A[M * K - 1];
  test = B[K * N - 1];
  if (beta || C) test = C[M * N - 1];
  debug("hybrid_dgemm_Ksplit -> Passed.");
  double *C_out, local_t;

  double *A_cpu, *A_gpu, *B_cpu, *B_gpu, *C_cpu, *C_gpu;

  double *C_buffer, cpu_beta = 0, gpu_beta = 0;

  size_t K_gpu = pred->K_split, K_cpu = K - pred->K_split, ldA = 0, ldB = 0,
         ldC = 0, d_ldA = 0, d_ldB = 0, d_ldC = 0;
  cublasOperation_t gpu_op_A, gpu_op_B;  // CUBLAS_OP_N, CUBLAS_OP_T
  CBLAS_TRANSPOSE cpu_op_A, cpu_op_B;    // CblasNoTrans, CblasTrans

  CBLAS_LAYOUT cblas_target;
  cublasHandle_t handle;
  cublasStatus_t stat = cublasCreate(&handle);

  gpu_timer_p cuda_timer = gpu_timer_init();

  /// Setup A parts on host and device
  local_t = csecond();
  switch (ctrl->A_mem) {
    case (ROW_MAJOR):
      gpu_op_A = CUBLAS_OP_T;
      d_ldA = K_gpu;
      A_cpu = &(A[K_gpu]);
      ldA = K;
      if (ctrl->C_mem == ROW_MAJOR)
        cpu_op_A = CblasNoTrans;
      else if (ctrl->C_mem == COL_MAJOR)
        cpu_op_A = CblasTrans;
      A_gpu = Dvec_chunk_transfer_gpu(A, M, K_gpu, K);
      break;
    case (COL_MAJOR):
      A_gpu = Dvec_transfer_gpu(A, K_gpu * M);
      gpu_op_A = CUBLAS_OP_N;
      d_ldA = M;
      A_cpu = &(A[K_gpu * M]);
      ldA = M;
      if (ctrl->C_mem == ROW_MAJOR)
        cpu_op_A = CblasTrans;
      else if (ctrl->C_mem == COL_MAJOR)
        cpu_op_A = CblasNoTrans;
      break;
    default:
      error("hybrid_dgemm_Ksplit -> Unreachable default reached ");
  }
  local_t = csecond() - local_t;
  ctrl->scatter_t += local_t;

  /// Setup B parts on host and device
  local_t = csecond();
  switch (ctrl->B_mem) {
    case (ROW_MAJOR):
      B_gpu = Dvec_transfer_gpu(B, K_gpu * N);
      gpu_op_B = CUBLAS_OP_T;
      d_ldB = N;
      B_cpu = &(B[K_gpu * N]);
      ldB = N;
      if (ctrl->C_mem == ROW_MAJOR)
        cpu_op_B = CblasNoTrans;
      else if (ctrl->C_mem == COL_MAJOR)
        cpu_op_B = CblasTrans;
      break;
    case (COL_MAJOR):
      gpu_op_B = CUBLAS_OP_N;
      d_ldB = K_gpu;
      B_cpu = &(B[K_gpu]);
      ldB = K;
      if (ctrl->C_mem == ROW_MAJOR)
        cpu_op_B = CblasTrans;
      else if (ctrl->C_mem == COL_MAJOR)
        cpu_op_B = CblasNoTrans;
      B_gpu = Dvec_chunk_transfer_gpu(B, N, K_gpu, K);
      break;
    default:
      error("hybrid_dgemm_Ksplit -> Unreachable default reached ");
  }
  local_t = csecond() - local_t;
  ctrl->scatter_t += local_t;

  /// Setup C parts on host and device
  local_t = csecond();
  d_ldC = M;
  C_buffer = (double *)pin_malloc(M * N * sizeof(double));

  if (!beta || !pred->Cadd_device_id || pred->Cadd_device_id == -1) {
    cpu_beta = beta;
    C_gpu = (double *)gpu_malloc(M * N * sizeof(double));
    if (!C) C = (double *)malloc(M * N * sizeof(double));
  } else if (pred->Cadd_device_id == 1) {
    error("hybrid_dgemm_Ksplit -> pred->Cadd_device_id == 1 Unimplemented.");
  }

  C_cpu = C;
  if (ctrl->C_mem == ROW_MAJOR) {
    cblas_target = CblasRowMajor;
    ldC = N;
  } else if (ctrl->C_mem == COL_MAJOR) {
    ldC = M;
    cblas_target = CblasColMajor;
  }
  local_t = csecond() - local_t;
  ctrl->alloc_t += local_t;

  if (!ldA || !ldB || !ldC || !d_ldA || !d_ldB || !d_ldC)
    error("hybrid_dgemm_Ksplit -> Some ld_dim were not defined correctly (=0)");

  gpu_timer_start(cuda_timer);

  stat = cublasDgemm(handle, gpu_op_A, gpu_op_B, M, N, K_gpu, &alpha, A_gpu,
                     d_ldA, B_gpu, d_ldB, &gpu_beta, C_gpu, d_ldC);
  gpu_timer_stop(cuda_timer);

  local_t = csecond();
  cblas_dgemm(cblas_target, cpu_op_A, cpu_op_B, M, N, K_cpu, alpha, A_cpu, ldA,
              B_cpu, ldB, cpu_beta, C_cpu, ldC);  /// ERROR!!!
  local_t = csecond() - local_t;
  ctrl->cpu_ex_t += local_t;
  cudaCheckErrors();

  ctrl->gpu_ex_t += (double)gpu_timer_get(cuda_timer) / 1000;

  local_t = csecond();
  cudaMemcpy(C_buffer, C_gpu, M * N * sizeof(double), cudaMemcpyDeviceToHost);
  local_t = csecond() - local_t;
  ctrl->gather_t += local_t;

  local_t = csecond();
  if (ctrl->C_mem == ROW_MAJOR)
    Dtranspose_add(C, C_buffer, N, M);
  else if (ctrl->C_mem == COL_MAJOR)
    cblas_daxpy(M * N, 1.0, C_buffer, 1, C, 1);
  local_t = csecond() - local_t;
  ctrl->reduce_t += local_t;

  // FIXME: implement correct frees with switches
  gpu_free(A_gpu);
  gpu_free(B_gpu);
  gpu_free(C_gpu);
  pin_free(C_buffer);

  C_out = C;
  debug("<- hybrid_dgemm_Ksplit()\n");
  return C_out;
}

int main(const int argc, const char *argv[]) {
  // print_devices();

  /*
    double *test, *test_T;

    test = Dvec_init_pinned(25, 42);
    test_T = Dvec_init_pinned(25, 0);

    for (int i = 0; i <5; i++){
            for (int j = 0; j <5; j++) fprintf(stderr,"%0.3lf ", test[5*i +j]);
            fprintf(stderr,"\n");
    }
    fprintf(stderr,"\n");
    fprintf(stderr,"\n");

    int s1 = 2, s2 = 5;
    Dtranspose(test_T, test, s1, s2);

    for (int i = 0; i <s1; i++){
            for (int j = 0; j <s2; j++) fprintf(stderr,"%0.3lf ", test[s2*i
    +j]); fprintf(stderr,"\n");
    }
    fprintf(stderr,"\n");
    for (int i = 0; i <s1*s2; i++)fprintf(stderr,"%0.3lf ", test_T[i]);
    fprintf(stderr,"\n");


    exit(1);
  */

  // Arguments: ( M N K A_mem{0,1} B_mem{0,1} C_mem{0,1} alpha) (beta
  // add_device) M_split N_split K_split cblas_mem{0,1}

  double alpha, beta;

  size_t M, N, K, itterations = 1;

  predict_p main_pred = (predict_p)malloc(sizeof(struct pred_str));
  control_p main_ctrl = (control_p)malloc(sizeof(struct control_str));

  main_ctrl->num_devices = 2;

  M = 100;
  K = 200;
  N = 300;
  main_ctrl->A_mem = ROW_MAJOR;
  main_ctrl->B_mem = ROW_MAJOR;
  main_ctrl->C_mem = ROW_MAJOR;
  alpha = 1.1;
  beta = 0;
  main_pred->Cadd_device_id = -1;
  main_pred->asynch_trans = 0;

  int ctr = 1;
  switch (argc) {
    case (14):
      M = atoi(argv[ctr++]);
      N = atoi(argv[ctr++]);
      K = atoi(argv[ctr++]);
      if (atoi(argv[ctr++]))
        main_ctrl->A_mem = COL_MAJOR;
      else
        main_ctrl->A_mem = ROW_MAJOR;
      if (atoi(argv[ctr++]))
        main_ctrl->B_mem = COL_MAJOR;
      else
        main_ctrl->B_mem = ROW_MAJOR;
      if (atoi(argv[ctr++]))
        main_ctrl->C_mem = COL_MAJOR;
      else
        main_ctrl->C_mem = ROW_MAJOR;
      alpha = atof(argv[ctr++]);
    case (7):
      beta = atof(argv[ctr++]);
      main_pred->Cadd_device_id = atoi(argv[ctr++]);
    case (5):
      main_pred->M_split = atoi(argv[ctr++]);
      main_pred->N_split = atoi(argv[ctr++]);
      main_pred->K_split = atoi(argv[ctr++]);
      break;
    default:
      error("Incorrect input arguments");
  }

  main_ctrl->alloc_t = main_ctrl->scatter_t = main_ctrl->transpose_t =
      main_ctrl->cpu_ex_t = main_ctrl->gpu_ex_t = main_ctrl->gather_t =
          main_ctrl->reduce_t = 0;

  // cudaStream_t stream1, stream2;
  // cudaStreamCreate(&stream1);
  // cudaStreamCreate (&stream2);

  double *A, *B, *C, *C_comp, *C_buffer, *d_A, *d_B, *d_C, *C_T;
  size_t ldA, ldB, ldC;
  cublasOperation_t gpu_op_A, gpu_op_B;  // CUBLAS_OP_N, CUBLAS_OP_T
  CBLAS_TRANSPOSE cpu_op_A, cpu_op_B;    // CblasNoTrans, CblasTrans
  CBLAS_LAYOUT cblas_layout;

  if (main_ctrl->A_mem == COL_MAJOR)
    ldA = M;
  else
    ldA = K;
  if (main_ctrl->B_mem == COL_MAJOR)
    ldB = K;
  else
    ldB = N;

  if (main_ctrl->C_mem == COL_MAJOR) {
    cblas_layout = CblasColMajor;
    ldC = M;
    if (main_ctrl->A_mem == COL_MAJOR) {
      cpu_op_A = CblasNoTrans;
      gpu_op_A = CUBLAS_OP_N;
    } else {
      cpu_op_A = CblasTrans;
      gpu_op_A = CUBLAS_OP_T;
    }
    if (main_ctrl->B_mem == COL_MAJOR) {
      cpu_op_B = CblasNoTrans;
      gpu_op_B = CUBLAS_OP_N;
    } else {
      cpu_op_B = CblasTrans;
      gpu_op_B = CUBLAS_OP_T;
    }
  } else {
    cblas_layout = CblasRowMajor;
    ldC = N;
    if (main_ctrl->A_mem == ROW_MAJOR) {
      cpu_op_A = CblasNoTrans;
      gpu_op_A = CUBLAS_OP_T;
    } else {
      cpu_op_A = CblasTrans;
      gpu_op_A = CUBLAS_OP_N;
    }
    if (main_ctrl->B_mem == ROW_MAJOR) {
      cpu_op_B = CblasNoTrans;
      gpu_op_B = CUBLAS_OP_T;
    } else {
      cpu_op_B = CblasTrans;
      gpu_op_B = CUBLAS_OP_N;
    }
  }

  int devices = 0;
  cudaGetDeviceCount(&devices);
  if (main_pred->M_split + main_pred->N_split + main_pred->K_split &&
      devices < 1)
    error("Trying to execute something CUDA-related on node without CUDA GPUs");
  else
    cudaSetDevice(devices - 1);

  double transpose_timer, cpu_timer = csecond(), total_t;
  float gpu_preproc_t = 0, gpu_comp_t = 0, gpu_reduce_t = 0;
  gpu_timer_p cuda_timer = gpu_timer_init();

  A = Dvec_init_pinned(M * K, 42);
  B = Dvec_init_pinned(K * N, 42);
  C = Dvec_init_pinned(M * N, 42);
  C_comp = Dvec_init_host(M * N, 0);
  Dvec_copy(C_comp, C, M * N);

  if (!strcmp(argv[ctr], "DEBUG")) {
    fprintf(stderr,
            "\nMatrix details: A(%s) B(%s) C(%s) -> M = %d, N = %d, K = %d\n",
            print_mem(main_ctrl->A_mem), print_mem(main_ctrl->B_mem),
            print_mem(main_ctrl->C_mem), M, N, K);
    fprintf(stderr, "Constants: alpha = %lf, beta = %lf\n", alpha, beta);
    fprintf(stderr,
            "Predicted values: M_split = %d, N_split = %d, K_split = %d "
            "Cadd_device "
            "= %d\n\n",
            main_pred->M_split, main_pred->N_split, main_pred->K_split,
            main_pred->Cadd_device_id);

    C_buffer = Dvec_init_host(M * N, 0);
    Dvec_copy(C_buffer, C, M * N);

    cpu_timer = csecond() - cpu_timer;
    fprintf(stderr,
            "Initializing Arrays on host (M=%d, N=%d, K=%d) t_init = %lf ms\n",
            M, N, K, cpu_timer * 1000);
    cpu_timer = csecond();
    for (int i = 0; i < 10; i++)
      cblas_dgemm(cblas_layout, cpu_op_A, cpu_op_B, M, N, K, alpha, A, ldA, B,
                  ldB, beta, C_comp, ldC);
    cpu_timer = csecond() - cpu_timer;
    fprintf(stderr, "MKL Sgemm(M=%d, N=%d, K=%d) ", M, N, K);
    report_results(cpu_timer / 10, (long)M * K * (2 * N + 1),
                   (long)(M * K + K * N + M * N * 2) * sizeof(double));
    fprintf(stderr, "\n");

    transpose_timer = csecond();
    if (main_ctrl->C_mem == ROW_MAJOR) {
      C_T = (double *)pin_malloc(M * N * sizeof(double));
      Dtranspose(C_T, C, M, N);
    } else
      C_T = C;
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
      stat = cublasDgemm(handle, gpu_op_A, gpu_op_B, M, N, K, &alpha, d_A, ldA,
                         d_B, ldB, &beta, d_C, M);
    }

    gpu_timer_stop(cuda_timer);
    cudaCheckErrors();
    gpu_comp_t = gpu_timer_get(cuda_timer);
    gpu_timer_start(cuda_timer);
    cudaMemcpy(C_T, d_C, M * N * sizeof(double), cudaMemcpyDeviceToHost);
    gpu_timer_stop(cuda_timer);
    transpose_timer = csecond() - transpose_timer;
    if (main_ctrl->C_mem == ROW_MAJOR) {
      Dtranspose(C, C_T, N, M);
      pin_free(C_T);
    } else
      C = C_T;
    transpose_timer = csecond() - transpose_timer;
    gpu_reduce_t = gpu_timer_get(cuda_timer);

    fprintf(stderr,
            "\nDevice overhead(M=%d, N=%d, K=%d) transpose = %lf ms, t_preproc "
            "= %lf "
            "ms, t_reduce = %lf ms\n",
            M, N, K, 1000 * transpose_timer, gpu_preproc_t, gpu_reduce_t);

    fprintf(stderr, "CUDA Sgemm(M=%d, N=%d, K=%d) ", M, N, K);
    report_results((double)gpu_comp_t / 1000.0 / 10, (long)M * K * (2 * N + 1),
                   (long)(M * K + K * N + M * N * 2) *
                       sizeof(double));  //(M*N+(long)M*K*(3*N+1))
    fprintf(stderr, "\n");

    Dtest_equality(C_comp, C, M * N);
    fprintf(stderr, "\n");

    gpu_free(d_A);
    gpu_free(d_B);
    gpu_free(d_C);
    pin_free(C);

    C = Dvec_init_pinned(M * N, 0);
    Dvec_copy(C, C_buffer, M * N);
    Dvec_copy(C_comp, C_buffer, M * N);

    cblas_dgemm(cblas_layout, cpu_op_A, cpu_op_B, M, N, K, alpha, A, ldA, B,
                ldB, beta, C_comp, ldC);
  } else if (!strcmp(argv[ctr], "TEST"))
    cblas_dgemm(cblas_layout, cpu_op_A, cpu_op_B, M, N, K, alpha, A, ldA, B,
                ldB, beta, C_comp, ldC);
  else if (!strcmp(argv[ctr], "BENCHMARK")) {
    itterations = 1000;
    fprintf(stdout, "%d,%d,%d,%s,%s,%s,%lf,%lf,%d,%d,%d,%d", M, N, K,
            print_mem(main_ctrl->A_mem), print_mem(main_ctrl->B_mem),
            print_mem(main_ctrl->C_mem), alpha, beta, main_pred->Cadd_device_id,
            main_pred->M_split, main_pred->N_split, main_pred->K_split);
  } else
    error("Invalid run mode given");

  if (!main_pred->M_split + !main_pred->N_split + !main_pred->K_split < 2)
    error("split more than one dim for 2 devices.");

  if (main_pred->M_split == M || main_pred->N_split == N ||
      main_pred->K_split == K) {
    for (int it = 0; it < itterations; it++)
      C = gpu_dgemm(main_ctrl, main_pred, M, N, K, alpha, A, B, beta, C);

    total_t =
        (main_ctrl->gpu_ex_t + main_ctrl->scatter_t + main_ctrl->alloc_t +
         main_ctrl->transpose_t + main_ctrl->gather_t + main_ctrl->reduce_t) /
        itterations;

  } else if (main_pred->M_split) {
    for (int it = 0; it < itterations; it++)
      C = hybrid_dgemm_Msplit(main_ctrl, main_pred, M, N, K, alpha, A, B, beta,
                              C);

    fprintf(stderr, "Hybrid Sgemm(M_cpu=%d, N=%d, K=%d) CPU time = ",
            (M - main_pred->M_split), N, K);
    report_results(main_ctrl->cpu_ex_t / itterations,
                   (long)(M - main_pred->M_split) * K * (2 * N + 1),
                   (long)((M - main_pred->M_split) * K + K * N +
                          (M - main_pred->M_split) * N * 2) *
                       sizeof(double));  //(M*N+(long)M*K*(3*N+1))
    fprintf(stderr, "\n");

    fprintf(stderr, "Hybrid Sgemm(M_gpu=%d, N=%d, K=%d) GPU time = ",
            main_pred->M_split, N, K);
    report_results(
        main_ctrl->gpu_ex_t / itterations,
        (long)main_pred->M_split * K * (2 * N + 1),
        (long)(main_pred->M_split * K + K * N + main_pred->M_split * N * 2) *
            sizeof(double));  //(M*N+(long)M*K*(3*N+1))
    fprintf(stderr, "\n");

    total_t =
        (fmax(main_ctrl->cpu_ex_t, main_ctrl->gpu_ex_t) + main_ctrl->scatter_t +
         main_ctrl->alloc_t + main_ctrl->transpose_t + main_ctrl->gather_t +
         main_ctrl->reduce_t) /
        itterations;

  } else if (main_pred->N_split) {
    for (int it = 0; it < itterations; it++)
      C = hybrid_dgemm_Nsplit(main_ctrl, main_pred, M, N, K, alpha, A, B, beta,
                              C);

    fprintf(stderr, "Hybrid Sgemm(M=%d, N_cpu=%d, K=%d) CPU time = ", M,
            (N - main_pred->N_split), K);
    report_results(main_ctrl->cpu_ex_t / itterations,
                   (long)M * K * (2 * (N - main_pred->N_split) + 1),
                   (long)(M * K + K * (N - main_pred->N_split) +
                          M * (N - main_pred->N_split) * 2) *
                       sizeof(double));  //(M*N+(long)M*K*(3*N+1))
    fprintf(stderr, "\n");

    fprintf(stderr, "Hybrid Sgemm(M=%d, N_gpu=%d, K=%d) GPU time = ", M,
            main_pred->N_split, K);
    report_results(
        main_ctrl->gpu_ex_t / itterations,
        (long)M * K * (2 * main_pred->N_split + 1),
        (long)(M * K + K * main_pred->N_split + M * main_pred->N_split * 2) *
            sizeof(double));  //(M*N+(long)M*K*(3*N+1))
    fprintf(stderr, "\n");

    total_t =
        (fmax(main_ctrl->cpu_ex_t, main_ctrl->gpu_ex_t) + main_ctrl->scatter_t +
         main_ctrl->alloc_t + main_ctrl->transpose_t + main_ctrl->gather_t +
         main_ctrl->reduce_t) /
        itterations;

  } else if (main_pred->K_split) {
    for (int it = 0; it < itterations; it++)
      C = hybrid_dgemm_Ksplit(main_ctrl, main_pred, M, N, K, alpha, A, B, beta,
                              C);

    fprintf(stderr, "Hybrid Sgemm(M=%d, N=%d, K_cpu=%d) CPU time = ", M, N,
            K - main_pred->K_split);
    report_results(main_ctrl->cpu_ex_t / itterations,
                   (long)M * (K - main_pred->K_split) * (2 * N + 1),
                   (long)(M * (K - main_pred->K_split) +
                          (K - main_pred->K_split) * N + M * N * 2) *
                       sizeof(double));  //(M*N+(long)M*K*(3*N+1))
    fprintf(stderr, "\n");

    fprintf(stderr, "Hybrid Sgemm(M=%d, N=%d, K_gpu=%d) GPU time = ", M, N,
            main_pred->K_split);
    report_results(
        main_ctrl->gpu_ex_t / itterations,
        (long)M * main_pred->K_split * (2 * N + 1),
        (long)(M * main_pred->K_split + main_pred->K_split * N + M * N * 2) *
            sizeof(double));  //(M*N+(long)M*K*(3*N+1))
    fprintf(stderr, "\n");

    total_t =
        (fmax(main_ctrl->cpu_ex_t, main_ctrl->gpu_ex_t) + main_ctrl->scatter_t +
         main_ctrl->alloc_t + main_ctrl->transpose_t + main_ctrl->gather_t +
         main_ctrl->reduce_t) /
        itterations;

  } else {
    debug("Not spliting at all, execute the whole on host");

    cpu_timer = csecond();
    for (int it = 0; it < itterations; it++)
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A,
                  K, B, N, beta, C, N);
    total_t = (csecond() - cpu_timer) / itterations;
    fprintf(stderr, "MKL Sgemm(M=%d, N=%d, K=%d) ", M, N, K);
    report_results(total_t, (long)M * K * (2 * N + 1),
                   (long)(M * K + K * N + M * N * 2) * sizeof(double));
    fprintf(stderr, "\n");
  }

  fprintf(stderr,
          "Device overhead(M=%d, N=%d, K=%d) pin_alloc = %lf ms, scatter = %lf "
          "ms, "
          "transpose = %lf ms, gather = %lf ms, reduce = %lf ms\n",
          M, N, K, 1000 * main_ctrl->alloc_t / itterations,
          1000 * main_ctrl->scatter_t / itterations,
          1000 * main_ctrl->transpose_t / itterations,
          1000 * main_ctrl->gather_t / itterations,
          1000 * main_ctrl->reduce_t / itterations);

  fprintf(stderr, "Total Sgemm(M=%d, N=%d, K=%d) ", M, N, K);
  report_results(total_t, (long)M * K * (2 * N + 1),
                 (long)(M * K + K * N + M * N * 2) *
                     sizeof(double));  //(M*N+(long)M*K*(3*N+1))
  fprintf(stderr, "\n");

  if (1 == itterations)
    Dtest_equality(C_comp, C, M * N);
  else
    fprintf(
        stdout, ",%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n",
        main_ctrl->alloc_t / itterations, main_ctrl->scatter_t / itterations,
        main_ctrl->transpose_t / itterations, main_ctrl->gather_t / itterations,
        main_ctrl->reduce_t / itterations, main_ctrl->cpu_ex_t / itterations,
        main_ctrl->gpu_ex_t / itterations, total_t);

  return 0;
}
