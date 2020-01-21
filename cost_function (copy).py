import subprocess
import sys
import numpy as np
import scipy.optimize as scipop
import scipy as sc
machine = 'gold1'
resDir = 'Results_' + machine + '/'

with open(resDir + 'bandwidth_log_' + machine + '.md_sorted', "r") as file0:
    bw_db = file0.readlines()

with open(resDir + 'daxpy_log_' + machine + '.md_sorted', "r") as file0:
    add_db = file0.readlines()

with open(resDir + 'transpose_log_' + machine + '.md', "r") as file0:
    trans_db = file0.readlines()

with open(resDir + 'CPU_only_log_' + machine + '.md_sorted', "r") as file0:
    cpu_gemm_db = file0.readlines()

with open(resDir + 'GPU_only_log_' + machine + '.md_sorted', "r") as file0:
    gpu_gemm_db = file0.readlines()


def report_bandwidth(bytes):
    cpu_to_gpu_time = t_memcpy(bytes, -1, 0)
    gpu_to_cpu_time = t_memcpy(bytes, 0, -1)
    print('CPU to GPU %d bytes t = %.5lf ms, bw =  %.3lf Gb/s' %
          (bytes, 1000 * cpu_to_gpu_time, GigaVal_per_s(bytes, cpu_to_gpu_time)))
    print('GPU to CPU %d bytes t = %.5lf ms, bw =  %.3lf Gb/s\n' %
          (bytes, 1000 * gpu_to_cpu_time, GigaVal_per_s(bytes, gpu_to_cpu_time)))


def report_flops(N):
    t_add = t_add_vec_1d(N, 8)
    print('Add (%d)  t = %.5lf ms, flops =  %.3lf Gfops/s\n' %
          (N, 1000 * t_add, GigaVal_per_s(N, t_add)))
    t_trans = t_dtranspose(N, N)
    print('Transpose (%d, %d)  t = %.5lf ms, flops =  %.3lf Gfops/s\n' %
          (N, N, 1000 * t_trans, GigaVal_per_s(N * N, t_trans)))
    t_gemm_cpu = t_dgemm_cpu(N, N / 2, N * 2)
    print('DGEMM CPU (%d, %d, %d)  t = %.5lf ms, flops =  %.3lf Gfops/s\n' %
          (N, N / 2, N * 2, 1000 * t_gemm_cpu, GigaVal_per_s(N * N / 2 * N * 2, t_gemm_cpu)))
    t_gemm_gpu = t_dgemm_gpu(N, N / 2, N * 2)
    print('DGEMM GPU (%d, %d, %d)  t = %.5lf ms, flops =  %.3lf Gfops/s\n' %
          (N, N / 2, N * 2, 1000 * t_gemm_gpu, GigaVal_per_s(N * N / 2 * N * 2, t_gemm_gpu)))


def GigaVal_per_s(val, time):
    return val * 1e-9 / time


def t_transfer_to_gpu(bytes):
    prev = []
    next = []
    ctr = len(bw_db)
    for line in bw_db:
        ctr -= 1
        temp = line.split(',')
        if int(temp[1]) == -1 and int(temp[2]) == 0:
            prev = next
            next = temp
            if int(temp[0]) >= bytes:
                break
    if next == []:
        sys.exit('Error: CPU to GPU benchmark not found')
    elif int(next[0]) == bytes:
        return float(next[3])
    if prev == []:
        print("t_transfer_to_gpu: No DB entry found, %d bytes < min benchmark value" % bytes)
        return float(next[3]) * bytes / int(next[0])
    elif ctr == 0:
        print("t_transfer_to_gpu: No DB entry found, %d bytes > max benchmark value" % bytes)
        return float(next[3]) * bytes / int(next[0])
    else:
        return (float(prev[3]) + (float(next[3]) - float(prev[3])) /
                (int(next[0]) - int(prev[0])) * (bytes - int(prev[0])))


def t_transfer_from_gpu():
    if  len(bw_db) == 0:
        sys.exit('Error: Bandwidth benchmark not found')
    time = [0]
    bytes = [0]
    for line in bw_db:
        temp = line.split(',')
        if int(temp[1]) == 0 and int(temp[2]) == -1 and temp[-1] != 'synth\n':
           if bytes[-1] != int(temp[0]):
                 time.append(float(temp[3]))
                 bytes.append(int(temp[0]))
           else:
                 print('Duplicate entry found for %d bytes' % bytes[-1])
    print (bytes, time)
    from scipy.interpolate import interp1d
    #f = interp1d(bytes, time)
    f2 = interp1d(bytes, time, kind='cubic')
    return f2

    prev = []
    next = []
    ctr = len(bw_db)
    for line in bw_db:
        ctr -= 1
        temp = line.split(',')
        if int(temp[1]) == 0 and int(temp[2]) == -1:
            prev = next
            next = temp
            if int(temp[0]) >= bytes:
                break

    elif int(next[0]) == bytes:
        return float(next[3])
    if prev == []:
        print("t_transfer_from_gpu: No DB entry found, %d bytes < min benchmark value" % bytes)
        return float(next[3]) * bytes / int(next[0])
    elif ctr == 0:
        print("t_transfer_from_gpu: No DB entry found, %d bytes > max benchmark value" % bytes)
        return float(next[3]) * bytes / int(next[0])
    else:
        return (float(prev[3]) + (float(next[3]) - float(prev[3])) /
                (int(next[0]) - int(prev[0])) * (bytes - int(prev[0])))


def t_dadd(X):
    prev = []
    next = []
    ctr = len(add_db)
    for line in add_db:
        ctr -= 1
        temp = line.split(',')
        prev = next
        next = temp
        if int(temp[0]) >= X:
            break
    if next == []:
        sys.exit('Error: dadd benchmark not found')
    elif int(next[0]) == X:
        return float(next[4])
    if prev == []:
        print("t_dadd: No DB entry found, %d < min benchmark value" % X)
        return float(next[4]) * X / int(next[0])
    elif ctr == 0:
        print("t_dadd: No DB entry found, %d > max benchmark value" % X)
        return float(next[4]) * X / int(next[0])
    else:
        return (float(prev[4]) + (float(next[4]) - float(prev[4])) /
                (int(next[0]) - int(prev[0])) * (X - int(prev[0])))


def t_dtranspose(X, Y):
    if (X == 0 or Y == 0):
        return 0
    ctr = len(trans_db)
    if ctr == 0:
        sys.exit('Error: transpose benchmark not found')
    var_list = [X, Y]
    var_bot = [0, 0]
    var_top = [0, 0]
    for dim in range(len(var_list)):
        for line in trans_db:
            ctr -= 1
            temp = line.split(',')
            var_bot[dim] = var_top[dim]
            var_top[dim] = int(temp[dim])
            if int(temp[dim]) >= var_list[dim]:
                break
        if ctr == 0:
            var_top[dim] = 0
        if (var_top[dim] == var_list[dim]):
            var_bot[dim] = var_list[dim]
    # print(var_bot)
    # print(var_list)
    # print(var_top)
    prev = []
    next = []
    for line in trans_db:
        temp = line.split(',')
        if var_bot[0] == int(temp[0]) and var_bot[1] == int(temp[1]):
            prev = temp
        if var_top[0] == int(temp[0]) and var_top[1] == int(temp[1]):
            next = temp
            break
    # print(prev)
    # print(next)
    if prev == []:
        print("t_transpose: No DB entry found, %d %d < min benchmark value" % (X, Y))
        return float(next[2]) * X / int(next[0]) * Y / int(next[1])
    elif next == []:
        print("t_transpose: No DB entry found, %d %d > max benchmark value" % (X, Y))
        return float(prev[2]) * X / int(prev[0]) * Y / int(prev[1])
    elif int(next[0]) == X and int(next[1]) == Y:
        return float(next[2])
    else:
        normalizer = 1.0
        if int(next[0]) != X:
            normalizer *= (X - float(prev[0])) / (int(next[0]) - int(prev[0]))
        if int(next[1]) != Y:
            normalizer *= (Y - float(prev[1])) / (int(next[1]) - int(prev[1]))
        return (float(prev[2]) + (float(next[2]) - float(prev[2])) * normalizer)


def t_dgemm_cpu(M, N, K):
    ctr = len(cpu_gemm_db)
    if ctr == 0:
        sys.exit('Error: t_dgemm_cpu benchmark not found')
    var_list = [M, N, K]
    var_bot = [0, 0, 0]
    var_top = [0, 0, 0]
    for dim in range(len(var_list)):
        for line in cpu_gemm_db:
            ctr -= 1
            temp = line.split(',')
            var_bot[dim] = var_top[dim]
            var_top[dim] = int(temp[dim])
            if int(temp[dim]) >= var_list[dim]:
                break
        if ctr == 0:
            var_top[dim] = 0
        if (var_top[dim] == var_list[dim]):
            var_bot[dim] = var_list[dim]
    # print(var_bot)
    # print(var_list)
    # print(var_top)
    prev = []
    next = []
    for line in cpu_gemm_db:
        temp = line.split(',')
        if var_bot[0] == int(temp[0]) and var_bot[1] == int(temp[1]) and var_bot[2] == int(temp[2]):
            prev = temp
        if var_top[0] == int(temp[0]) and var_top[1] == int(temp[1]) and var_top[2] == int(temp[2]):
            next = temp
            break
    # print(prev)
    # print(next)
    if prev == []:
        print("t_dgemm_cpu: No DB entry found, %d %d %d < min benchmark value" % (M, N, K))
        return float(temp[19]) * M / int(next[0]) * N / int(next[1]) * K / int(next[2])
    elif next == []:
        print("t_dgemm_cpu: No DB entry found, %d %d %d > max benchmark value" % (M, N, K))
        return float(prev[19]) * M / int(prev[0]) * N / int(prev[1]) * K / int(prev[2])
    elif int(next[0]) == M and int(next[1]) == N and int(next[2]) == K:
        return float(next[19])
    else:
        normalizer = 1.0
        if int(next[0]) != M:
            normalizer *= (M - float(prev[0])) / (int(next[0]) - int(prev[0]))
        if int(next[1]) != N:
            normalizer *= (N - float(prev[1])) / (int(next[1]) - int(prev[1]))
        if int(next[2]) != K:
            normalizer *= (K - float(prev[2])) / (int(next[2]) - int(prev[2]))
        return (float(prev[19]) + (float(next[19]) - float(prev[19])) * normalizer)


def t_dgemm_gpu(M, N, K):
    ctr = len(gpu_gemm_db)
    if ctr == 0:
        sys.exit('Error: t_dgemm_cpu benchmark not found')
    var_list = [M, N, K]
    var_bot = [0, 0, 0]
    var_top = [0, 0, 0]
    for dim in range(len(var_list)):
        for line in gpu_gemm_db:
            ctr -= 1
            temp = line.split(',')
            var_bot[dim] = var_top[dim]
            var_top[dim] = int(temp[dim])
            if int(temp[dim]) >= var_list[dim]:
                break
        if ctr == 0:
            var_top[dim] = 0
        if (var_top[dim] == var_list[dim]):
            var_bot[dim] = var_list[dim]
    # print(var_bot)
    # print(var_list)
    # print(var_top)
    prev = []
    next = []
    for line in gpu_gemm_db:
        temp = line.split(',')
        if var_bot[0] == int(temp[0]) and var_bot[1] == int(temp[1]) and var_bot[2] == int(temp[2]):
            prev = temp
        if var_top[0] == int(temp[0]) and var_top[1] == int(temp[1]) and var_top[2] == int(temp[2]):
            next = temp
            break
    # print(prev)
    # print(next)
    if prev == []:
        print("t_dgemm_cpu: No DB entry found, %d %d %d < min benchmark value" % (M, N, K))
        return float(temp[19]) * M / int(next[0]) * N / int(next[1]) * K / int(next[2])
    elif next == []:
        print("t_dgemm_cpu: No DB entry found, %d %d %d > max benchmark value" % (M, N, K))
        return float(prev[19]) * M / int(prev[0]) * N / int(prev[1]) * K / int(prev[2])
    elif int(next[0]) == M and int(next[1]) == N and int(next[2]) == K:
        return float(next[19])
    else:
        normalizer = 1.0
        if int(next[0]) != M:
            normalizer *= (M - float(prev[0])) / (int(next[0]) - int(prev[0]))
        if int(next[1]) != N:
            normalizer *= (N - float(prev[1])) / (int(next[1]) - int(prev[1]))
        if int(next[2]) != K:
            normalizer *= (K - float(prev[2])) / (int(next[2]) - int(prev[2]))
        return (float(prev[19]) + (float(next[19]) - float(prev[19])) * normalizer)


def t_memcpy(bytes, src, dest):
        # For now only for dev0 <-> host
    if (src == -1 and dest == 0):
        return t_transfer_to_gpu(bytes)
    if (src == 0 and dest == -1):
        return t_transfer_from_gpu(bytes)


def t_copy_vec_1d(N, src, dest, elem_size):
    return t_memcpy(N * elem_size, src, dest)


def t_copy_vec_2d(dim1, dim2, ldim, src, dest, elem_size):
    if ldim == dim2:
        return t_memcpy(dim1 * dim2 * elem_size, src, dest)
    elif ldim > dim2:
        return dim1 * t_memcpy(dim2 * elem_size, src, dest)
    else:
        sys.exit('Error: dim2(%d) > ldim(%d)' % (dim2, ldim))


def t_add_vec_1d(N, elem_size):
    # For now assume 8 bytes -> double
    if elem_size == 8:
        return t_dadd(N)
    else:
        sys.exit('Error: t_add_vec_1d(%d,%d) -> unknown elem_size' % (N, elem_size))


def t_add_vec_2d(dim1, dim2, ldim, elem_size):
    if ldim == dim2:
        return t_add_vec_1d(dim1 * dim2, elem_size)
    elif ldim > dim2:
        return dim1 * t_add_vec_1d(dim2, elem_size)
    else:
        sys.exit('Error: dim2(%d) > ldim(%d)' % (dim2, ldim))


sizes = [3200]  # ,100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000, 20000000, 50000000, 100000000, 200000000, 500000000, 1000000000 ]


def f_M_split(M_split, M, N, K, elem_size):
    for A_mem in [0, 1]:
        for B_mem in [0, 1]:
            for C_mem in [0, 1]:
                temp = Impl_M_split(M_split, M, N, K, A_mem, B_mem, C_mem, elem_size)
                if (temp > min_fun_t):
                    min_fun_t = temp
    return min_fun_t


def Impl_M_split(M_split, M, N, K, A_mem, B_mem, C_mem, elem_size):
    M_gpu = M_split
    M_cpu = M - M_split
    if A_mem:
        A_dim1, A_dim2, ldim = K, M_gpu, M
    else:
        A_dim1, A_dim2, ldim = M_gpu, K, K
    t_transfer_A = t_copy_vec_2d(A_dim1, A_dim2, ldim, -1, 0, elem_size)
    if B_mem:
        B_dim1, B_dim2, ldim = N, K, K
    else:
        B_dim1, B_dim2, ldim = K, N, N
    t_transfer_B = t_copy_vec_2d(B_dim1, B_dim2, ldim, -1, 0, elem_size)
    if C_mem:
        C_dim1, C_dim2, ldim = N, M_gpu, M
    else:
        C_dim1, C_dim2, ldim = M_gpu, N, N
    t_get_C = t_copy_vec_2d(C_dim1, C_dim2, ldim, 0, -1, elem_size)
    t_transfer_C = t_copy_vec_2d(C_dim1, C_dim2, ldim, -1, 0, elem_size)

    t_gpu_overhead = t_transfer_A + t_transfer_B + t_transfer_C + t_get_C
    t_cpu_overhead = 1 / 10 * (1 - C_mem) * (t_dtranspose(M_gpu, N) + t_dtranspose(N, M_gpu))
    t_total_M_split = t_gpu_overhead + t_cpu_overhead + \
        max(t_dgemm_gpu(M_gpu, N, K), t_dgemm_cpu(M_cpu, N, K))
    print(M_split, t_total_M_split)
    return t_total_M_split


def Impl_N_split(N_split, M, N, K, A_mem, B_mem, C_mem, elem_size):
    N_gpu = N_split
    N_cpu = N - N_split
    if A_mem:
        A_dim1, A_dim2, ldim = K, M, M
    else:
        A_dim1, A_dim2, ldim = M, K, K
    t_transfer_A = t_copy_vec_2d(A_dim1, A_dim2, ldim, -1, 0, elem_size)
    if B_mem:
        B_dim1, B_dim2, ldim = N_gpu, K, K
    else:
        B_dim1, B_dim2, ldim = K, N_gpu, N
    t_transfer_B = t_copy_vec_2d(B_dim1, B_dim2, ldim, -1, 0, elem_size)
    if C_mem:
        C_dim1, C_dim2, ldim = N_gpu, M, M
    else:
        C_dim1, C_dim2, ldim = M, N_gpu, N
    t_get_C = t_copy_vec_2d(C_dim1, C_dim2, ldim, 0, -1, elem_size)
    t_transfer_C = t_copy_vec_2d(C_dim1, C_dim2, ldim, -1, 0, elem_size)

    t_gpu_overhead = t_transfer_A + t_transfer_B + t_transfer_C + t_get_C
    t_cpu_overhead = 1 / 10 * (1 - C_mem) * (t_dtranspose(M, N_gpu) + t_dtranspose(N_gpu, M))
    t_total_N_split = t_gpu_overhead + t_cpu_overhead + \
        max(t_dgemm_gpu(M, N_gpu, K), t_dgemm_cpu(M, N_cpu, K))
    print(N_split, t_total_N_split)
    return t_total_N_split


def Impl_K_split(K_split, M, N, K, A_mem, B_mem, C_mem, elem_size):
    K_gpu = K_split
    K_cpu = K - K_split
    if A_mem:
        A_dim1, A_dim2, ldim = K_gpu, M, M
    else:
        A_dim1, A_dim2, ldim = M, K_gpu, K
    t_transfer_A = t_copy_vec_2d(A_dim1, A_dim2, ldim, -1, 0, elem_size)
    if B_mem:
        B_dim1, B_dim2, ldim = N, K_gpu, K
    else:
        B_dim1, B_dim2, ldim = K_gpu, N, N
    t_transfer_B = t_copy_vec_2d(B_dim1, B_dim2, ldim, -1, 0, elem_size)
    if C_mem:
        C_dim1, C_dim2, ldim = N, M, M
    else:
        C_dim1, C_dim2, ldim = M, N, N
    t_get_C = t_copy_vec_2d(C_dim1, C_dim2, ldim, 0, -1, elem_size)
    t_transfer_C = t_copy_vec_2d(C_dim1, C_dim2, ldim, -1, 0, elem_size)

    t_gpu_overhead = t_transfer_A + t_transfer_B + t_transfer_C + t_get_C
    t_cpu_overhead = 1 / 10 * (1 - C_mem) * (t_dtranspose(M, N) +
                                             t_dtranspose(N, M)) + t_add_vec_1d(M * N, elem_size)
    t_total_K_split = t_gpu_overhead + t_cpu_overhead + \
        max(t_dgemm_gpu(M, N, K_gpu), t_dgemm_cpu(M, N, K_cpu))
    print(K_split, t_total_K_split)
    return t_total_K_split


def Wrap_any_split(ratio, M, N, K, A_mem, B_mem, C_mem, elem_size):
   return min([Impl_K_split(int(ratio*K), M, N, K, A_mem, B_mem, C_mem, elem_size), Impl_M_split(int(ratio*M), M, N, K, A_mem, B_mem, C_mem, elem_size), Impl_N_split(int(ratio*N), M, N, K, A_mem, B_mem, C_mem, elem_size)])

dtype_size = 8
t_min = 100

N = M = K = 100000
x = np.ndarray(1)
# x[0]=int(0.5*M)
# answer = scipop.minimize_scalar(Impl_M_split, args=(M, N, K, 0, 0, 0, 8), method='Bounded', bounds=(0,M))
# answer1 = scipop.minimize_scalar(Impl_N_split, args=(M, N, K, 0, 0, 0, 8), method='Bounded', bounds=(0,N))
# answer2 = scipop.minimize_scalar(Impl_K_split, args=(M, N, K, 0, 0, 0, 8), method='Bounded', bounds=(0,K))
# print(answer)
# print(answer1)
# print(answer2)
# report_bandwidth(N*dtype_size)
# report_flops(N)

f = t_transfer_from_gpu()
print(f)
plot_down_bound = bytes[0]
plot_up_bound = bytes[-1]
xnew = np.linspace(plot_down_bound, plot_up_bound, num=41, endpoint=True)
times_bound = []
bytes_bound = []
for elem,t in zip(bytes,time):
    temp = line.split(',')
    if elem <= plot_up_bound and elem >= plot_down_bound:
        times_bound.append(t)
        bytes_bound.append(elem)
import matplotlib.pyplot as plt
plt.plot(bytes_bound, times_bound, 'o', xnew, f(xnew), '-',  xnew, xnew/(16e9), '--', xnew, xnew/(8e9), '--')
# plt.xscale('log')
plt.legend(['data', 'linear', 'theoretical(16Gb/s)', 'theoretical(8Gb/s)'], loc='best')
# plt.show()




