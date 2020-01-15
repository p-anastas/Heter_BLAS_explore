import subprocess

machine = 'gold1'
resDir = 'Results_' + machine + '/'

with open(resDir + 'daxpy_log_' + machine + '.md_sorted', "r") as file0:
    add_db = file0.readlines()

with open(resDir + 'bandwidth_log_' + machine + '.md_sorted', "r") as file0:
    bw_db = file0.readlines()

with open(resDir + 'transpose_log_' + machine + '.md', "r") as file0:
    trans_db = file0.readlines()

with open(resDir + 'CPU_only_log_' + machine + '.md_sorted', "r") as file0:
    cpu_gemm_db = file0.readlines()

with open(resDir + 'GPU_only_log_' + machine + '.md_sorted', "r") as file0:
    gpu_gemm_db = file0.readlines()

# For now assume no overlaping
communication_overlap = 0
gpu_init_overlap = 0
cpu_init_overlap = 0
transpose_overlap = 0
add_overlap = 0


def binary_bounds(start, X):
    next = start
    while (next < X):
        next *= 2
    if next == X:
        prev = next
    else:
        prev = next / 2
    return (prev, next)


def t_init_pinned(bytes):
    return bytes / 2e10 * (1 - cpu_init_overlap)


def t_init_gpu(bytes):
    return bytes / 2e10 * (1 - gpu_init_overlap)


def t_transfer_to_gpu(bytes):
    bytes_min, bytes_max = binary_bounds(625, bytes)
    time_min = 0
    time_max = 0
    for line in bw_db:
        temp = line.split(',')
        if int(temp[0]) == bytes_min and int(temp[1]) == -1 and int(temp[2]) == 0:
            time_min = float(temp[3])
        if int(temp[0]) == bytes_max and int(temp[1]) == -1 and int(temp[2]) == 0:
            time_max = float(temp[3])
    if (time_min == 0 or time_max == 0):
        print("t_transfer_to_gpu: No DB entry found")
        return bytes / 6e9 * (1 - communication_overlap)
    else:
        if bytes_max == bytes_min:
            return (1 - communication_overlap) * time_min
        else:
            return (1 - communication_overlap) * (time_min + (time_max - time_min) /
                                                  (bytes_max - bytes_min) * (bytes - bytes_min))


def t_transfer_from_gpu(bytes):
    bytes_min, bytes_max = binary_bounds(625, bytes)
    time_min = 0
    time_max = 0
    for line in bw_db:
        temp = line.split(',')
        if int(temp[0]) == bytes_min and int(temp[1]) == 0 and int(temp[2]) == -1:
            time_min = float(temp[3])
        if int(temp[0]) == bytes_max and int(temp[1]) == 0 and int(temp[2]) == -1:
            time_max = float(temp[3])
    if (time_min == 0 or time_max == 0):
        print("t_transfer_from_gpu: No DB entry found")
        return bytes / 6e9 * (1 - communication_overlap)
    else:
        if bytes_max == bytes_min:
            return (1 - communication_overlap) * time_min
        else:
            return (1 - communication_overlap) * (time_min + (time_max - time_min) /
                                                  (bytes_max - bytes_min) * (bytes - bytes_min))


def t_dtranspose(X):
    for line in trans_db:
        temp = line.split(',')
        if int(temp[0]) * int(temp[1]) >= X:
            return float(temp[2]) * (X / (int(temp[0]) * int(temp[1]))) * (1 - transpose_overlap)
    print("No DB entry found")
    return 0.01 * X / 1e9 * (1 - transpose_overlap)


def t_dgemm_cpu(N, M, K):
    if (M == 0 or N == 0 or K == 0):
        return 0
    M_min, M_max = binary_bounds(100, M)
    N_min, N_max = binary_bounds(100, N)
    K_min, K_max = binary_bounds(100, K)
    print(M_min, M_max, N_min, N_max, K_min, K_max)
    time_min = 0
    time_max = 0
    for line in cpu_gemm_db:
        temp = line.split(',')
        if int(temp[0]) == M_min and int(temp[1]) == N_min and int(temp[2]) == K_min:
            time_min = float(temp[19])
        if int(temp[0]) == M_max and int(temp[1]) == N_min and int(temp[2]) == K_min:
            time_max = float(temp[19])
    if (time_min == 0 or time_max == 0):
        print("t_dgemm_cpu: No DB entry found")
        return 0.0064 * N * M * K * (1 - add_overlap)
    else:
        if M_max == M_min or N_max == N_min or K_max == K_min:
            return time_min
        else:
            return (time_min + (time_max - time_min) / (M_max - M_min) * (M - M_min) /
                    (N_max - N_min) * (N - N_min) / (K_max - K_min) * (K - K_min))


def t_dgemm_gpu(N, M, K):
    if (M == 0 or N == 0 or K == 0):
        return 0
    M_min, M_max = binary_bounds(100, M)
    N_min, N_max = binary_bounds(100, N)
    K_min, K_max = binary_bounds(100, K)
    print(M_min, M_max, N_min, N_max, K_min, K_max)
    time_min = 0
    time_max = 0
    for line in gpu_gemm_db:
        temp = line.split(',')
        if int(temp[0]) == M_min and int(temp[1]) == N_min and int(temp[2]) == K_min:
            time_min = float(temp[19])
        if int(temp[0]) == M_max and int(temp[1]) == N_min and int(temp[2]) == K_min:
            time_max = float(temp[19])
    if (time_min == 0 or time_max == 0):
        print("t_dgemm_cpu: No DB entry found")
        return 0.0064 * N * M * K * (1 - add_overlap)
    else:
        if M_max == M_min or N_max == N_min or K_max == K_min:
            return time_min
        else:
            return (time_min + (time_max - time_min) / (M_max - M_min) * (M - M_min) /
                    (N_max - N_min) * (N - N_min) / (K_max - K_min) * (K - K_min))


def t_dadd(X):
    N_min, N_max = binary_bounds(625, X)
    time_min = 0
    time_max = 0
    for line in add_db:
        temp = line.split(',')
        if int(temp[0]) == N_min:
            time_min = float(temp[2])
        if int(temp[0]) == N_max:
            time_max = float(temp[2])
    if (time_min == 0 or time_max == 0):
        print("t_dadd: No DB entry found")
        return 0.0064 * (1 - add_overlap)
    else:
        if N_max == N_min:
            return (1 - communication_overlap) * time_min
        else:
            return (1 - communication_overlap) * (time_min +
                                                  (time_max - time_min) / (N_max - N_min) * (N - N_min))


def t_dbuffer_init_gpu(N, M, K):
    return t_init_gpu(8 * (M * N + N * K + K * M))


def t_dbuffer_init_cpu(N, M, K, buffer_flag):
    return t_init_pinned(8 * buffer_flag * M * N)


def t_dcommunication(N, M, K, Ctransfer_flag):
    return t_transfer_to_gpu(8 * (Ctransfer_flag * M * N + N * K + K * M)
                             ) + t_transfer_from_gpu(8 * M * N)


def dgemm_flops(N, M, K):
    return M * K * (2 * N + 1)


def dgemm_bytes(N, M, K):
    return (M * K + K * N + M * N * 2) * 8


def print_bench(time, flops, bytes):
    Gflops = flops / (time * 1e9)
    Gbytes = bytes / (time * 1e9)
    return str(round(1000 * time, 5)) + ' ms ( ' + str(round(Gflops, 3)) + \
        ' Gflops/s ' + str(round(Gbytes, 3)) + ' Gbytes/s)'


# Inputs = N, M, K, M_layout, N_layout, K_layout, (alpha), beta
# For all cases, t_total = t_gemm_computation + t_buffer_init + t_communication + t_transform + t_axpy
# Goal: minimize t_total
# For CPU-GPU environment choises are:


N = 10000
M = K = 10000

# CPU only ->
t_total_cpu = t_dgemm_cpu(N, M, K)
print(
    "CPU only M=%d N=%d K=%d -> " %
    (M,
     N,
     K) +
    print_bench(
        t_total_cpu,
        dgemm_flops(
            N,
            M,
            K),
        dgemm_bytes(
            N,
            M,
            K)))
# GPU only ->
for C_mem in [0, 1]:
    t_total_gpu = t_dgemm_gpu(N,
                              M,
                              K) + t_dbuffer_init_gpu(N,
                                                      M,
                                                      K) + t_dbuffer_init_cpu(N,
                                                                              M,
                                                                              K,
                                                                              1 - C_mem) + t_dcommunication(N,
                                                                                                            M,
                                                                                                            K,
                                                                                                            1) + t_dtranspose(2 * M * N * (1 - C_mem))
    t_total_gpu_addChost = t_dgemm_gpu(N,
                                       M,
                                       K) + t_dbuffer_init_gpu(N,
                                                               M,
                                                               K) + t_dbuffer_init_cpu(N,
                                                                                       M,
                                                                                       K,
                                                                                       1 - C_mem) + t_dcommunication(N,
                                                                                                                     M,
                                                                                                                     K,
                                                                                                                     0) + t_dtranspose(M * N * (1 - C_mem) + t_dadd(M * N))  # 1 extra t_dadd hidden in CPU while gpu_gemm
    #print("GPU Naive M=%d N=%d K=%d C_mem=%d -> " %(M,N,K,C_mem) +  print_bench(t_total_gpu, dgemm_flops(N,M,K), dgemm_bytes(N,M,K)))
    #print("GPU addChost M=%d N=%d K=%d C_mem=%d -> " %(M,N,K, C_mem) + print_bench(t_total_gpu_addChost, dgemm_flops(N,M,K), dgemm_bytes(N,M,K)))
    for A_mem in [0, 1]:
        for B_mem in [0, 1]:
            Predicted_times = [(0, 0, 0, -1, t_total_cpu)]
            for M_split in range(int(M / 5), M + 1, int(M / 5)):
                M_gpu = M_split
                M_cpu = M - M_split
                t_transfer_A = (1 - A_mem) * t_transfer_to_gpu(8 * M_gpu * K) + \
                    A_mem * K * t_transfer_to_gpu(8 * M_gpu)
                t_transfer_Β = t_transfer_to_gpu(8 * N * K)
                t_transfer_C = (1 - C_mem) * t_transfer_to_gpu(8 * M_gpu * N) + \
                    C_mem * N * t_transfer_to_gpu(8 * M_gpu)
                t_get_C = (1 - C_mem) * t_transfer_from_gpu(8 * M_gpu * N) + \
                    C_mem * N * t_transfer_from_gpu(8 * M_gpu)
                t_gpu_overhead = t_dbuffer_init_gpu(N, M_gpu, K) + t_transfer_A + t_transfer_Β + t_transfer_C + t_get_C
                t_cpu_overhead = t_dbuffer_init_cpu(
                    N, M_gpu, K, 1 - C_mem) + t_dtranspose(2 * M_gpu * N * (1 - C_mem))
                t_total_M_split = t_gpu_overhead + t_cpu_overhead + \
                    max(t_dgemm_gpu(N, M_gpu, K), t_dgemm_cpu(N, M_cpu, K))
                Predicted_times.append((M_split, 0, 0, -1, t_total_M_split))
                #print("Hybrid M=%d N=%d K=%d C_mem=%d ( M_split=%d )-> " %(M,N,K,C_mem, M_split) +  print_bench(t_total_M_split, dgemm_flops(N,M,K), dgemm_bytes(N,M,K)))
                t_gpu_overhead_addChost = t_dbuffer_init_gpu(N, M_gpu, K) + t_transfer_A + t_transfer_Β + t_transfer_C
                t_cpu_overhead_addChost = t_dbuffer_init_cpu(N, M_gpu, K, 1) + t_dtranspose(
                    M_gpu * N * (1 - C_mem)) + C_mem * N * t_dadd(M_gpu) + (1 - C_mem) * t_dadd(N * M_gpu)
                t_total_M_split_addChost = t_gpu_overhead_addChost + t_cpu_overhead_addChost + \
                    max(t_dgemm_gpu(N, M_gpu, K), t_dgemm_cpu(N, M_cpu, K) + +
                        C_mem * N * t_dadd(M_gpu) + (1 - C_mem) * t_dadd(N * M_gpu))
                Predicted_times.append((M_split, 0, 0, 0, t_total_M_split_addChost))
                #print("Hybrid addChost M=%d N=%d K=%d A_mem=%d B_mem=%d C_mem=%d ( M_split=%d )-> " %(M,N,K,A_mem, B_mem,C_mem, M_split) +  print_bench(t_total_M_split_addChost, dgemm_flops(N,M,K), dgemm_bytes(N,M,K)))

            for N_split in range(int(N / 5), N + 1, int(N / 5)):
                N_gpu = N_split
                N_cpu = N - N_split
                t_transfer_A = t_transfer_to_gpu(8 * M * K)
                t_transfer_Β = B_mem * t_transfer_to_gpu(8 * N_gpu * K) + (1 - B_mem) * K * t_transfer_to_gpu(8 * N_gpu)
                t_transfer_C = C_mem * \
                    t_transfer_to_gpu(8 * N_gpu * M) + (1 - C_mem) * \
                    M * t_transfer_to_gpu(8 * N_gpu)
                t_get_C = C_mem * \
                    t_transfer_from_gpu(8 * N_gpu * M) + (1 - C_mem) * \
                    M * t_transfer_from_gpu(8 * N_gpu)
                t_gpu_overhead = t_dbuffer_init_gpu(N, M_gpu, K) + t_transfer_A + t_transfer_Β + t_transfer_C + t_get_C
                t_cpu_overhead = t_dbuffer_init_cpu(
                    N_gpu, M, K, 1 - C_mem) + t_dtranspose(2 * M * N_gpu * (1 - C_mem))
                t_total_N_split = t_gpu_overhead + t_cpu_overhead + \
                    max(t_dgemm_gpu(N_gpu, M, K), t_dgemm_cpu(N_cpu, M, K))
                Predicted_times.append((N_split, 0, 0, -1, t_total_N_split))
                #print("Hybrid M=%d N=%d K=%d C_mem=%d ( N_split=%d )-> " %(M,N,K,C_mem, N_split) +  print_bench(t_total_N_split, dgemm_flops(N,M,K), dgemm_bytes(N,M,K)))
                t_gpu_overhead_addChost = t_dbuffer_init_gpu(N_gpu, M, K) + t_transfer_A + t_transfer_Β + t_transfer_C
                t_cpu_overhead_addChost = t_dbuffer_init_cpu(N_gpu, M, K, 1) + t_dtranspose(
                    M * N_gpu * (1 - C_mem)) + (1 - C_mem) * M * t_dadd(N_gpu) + C_mem * t_dadd(N_gpu * M)
                t_total_N_split_addChost = t_gpu_overhead_addChost + t_cpu_overhead_addChost + \
                    max(t_dgemm_gpu(N_gpu, M, K), t_dgemm_cpu(N_cpu, M, K) +
                        (1 - C_mem) * M * t_dadd(N_gpu) + C_mem * t_dadd(N_gpu * M))
                Predicted_times.append((0, N_split, 0, -1, t_total_N_split_addChost))
                #print("Hybrid addChost M=%d N=%d K=%d A_mem=%d B_mem=%d C_mem=%d ( N_split=%d )-> " %(M,N,K,A_mem, B_mem,C_mem, N_split) +  print_bench(t_total_N_split_addChost, dgemm_flops(N,M,K), dgemm_bytes(N,M,K)))

            for K_split in range(int(K / 10), K + 1, int(K / 10)):
                K_gpu = K_split
                K_cpu = K - K_split
                t_transfer_A = A_mem * \
                    t_transfer_to_gpu(8 * M * K_gpu) + (1 - A_mem) * \
                    M * t_transfer_to_gpu(8 * K_gpu)
                t_transfer_Β = (1 - B_mem) * t_transfer_to_gpu(8 * M * K_gpu) + B_mem * M * t_transfer_to_gpu(8 * K_gpu)
                t_get_C = t_transfer_from_gpu(8 * N * M)
                t_gpu_overhead = t_dbuffer_init_gpu(N, M, K_gpu) + + t_transfer_A + t_transfer_Β + t_get_C
                t_cpu_overhead = t_dbuffer_init_cpu(
                    N, M, K_gpu, 1) + t_dtranspose(M * N * (1 - C_mem)) + t_dadd(M * N)
                t_total_K_split = t_gpu_overhead + t_cpu_overhead + \
                    max(t_dgemm_gpu(N, M, K_gpu), t_dgemm_cpu(N, M, K_cpu))
                Predicted_times.append((0, 0, K_split, 0, t_total_K_split))
                #print("Hybrid (always addChost) M=%d N=%d K=%d A_mem=%d B_mem=%d C_mem=%d ( K_split=%d )-> " %(M,N,K,A_mem, B_mem,C_mem, K_split) +  print_bench(t_total_K_split, dgemm_flops(N,M,K), dgemm_bytes(N,M,K)))
            sorted_times = sorted(Predicted_times, key=lambda tup: tup[4])
            for ctr in range(3):
                best_option = sorted_times[ctr]
                if best_option[0] > 0:
                    split = 'M_split=' + str(best_option[0])
                elif best_option[1] > 0:
                    split = 'N_split=' + str(best_option[1])
                elif best_option[2] > 0:
                    split = 'K_split=' + str(best_option[2])
                else:
                    split = 'CPU only'
                if best_option[3] == -1:
                    addition = 'add=Normal '
                elif best_option[3] == 0:
                    addition = 'add=Host '

                print(
                    "M=%d N=%d K=%d A_mem=%d B_mem=%d C_mem=%d -> " %
                    (M,
                     N,
                     K,
                     A_mem,
                     B_mem,
                     C_mem) +
                    addition +
                    split +
                    ' ' +
                    print_bench(
                        best_option[4],
                        dgemm_flops(
                            N,
                            M,
                            K),
                        dgemm_bytes(
                            N,
                            M,
                            K)))
            print()
