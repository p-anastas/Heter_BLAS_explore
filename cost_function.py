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

def t_transfer_to_gpu():
    if  len(bw_db) == 0:
        sys.exit('Error: Bandwidth benchmark not found')
    time = [0]
    bytes = [0]
    for line in bw_db:
        temp = line.split(',')
        if int(temp[1]) == -1 and int(temp[2]) == 0 and True: #temp[-1] != 'synth\n':
           if bytes[-1] != int(temp[0]):
                 time.append(float(temp[3]))
                 bytes.append(int(temp[0]))
           else:
                 print('Duplicate entry found for %d bytes' % bytes[-1])
    from scipy.interpolate import interp1d
    f = interp1d(bytes, time)
    #f = interp1d(bytes, time, kind='cubic')
    return f


def t_transfer_from_gpu():
    if  len(bw_db) == 0:
        sys.exit('Error: Bandwidth benchmark not found')
    time = [0]
    bytes = [0]
    for line in bw_db:
        temp = line.split(',')
        if int(temp[1]) == 0 and int(temp[2]) == -1 and True: #temp[-1] != 'synth\n':
           if bytes[-1] != int(temp[0]):
                 time.append(float(temp[3]))
                 bytes.append(int(temp[0]))
           else:
                 print('Duplicate entry found for %d bytes' % bytes[-1])
    from scipy.interpolate import interp1d
    f = interp1d(bytes, time)
    #f = interp1d(bytes, time, kind='cubic')
    return f

def t_dadd():
    if  len(add_db) == 0:
        sys.exit('Error: daxpy benchmark not found')
    time = [0]
    elems = [0]
    for line in add_db:
        temp = line.split(',')
        if True: #temp[-1] != 'synth\n':
           if elems[-1] != int(temp[0]):
                 time.append(float(temp[4]))
                 elems.append(int(temp[0]))
           else:
                 print('Duplicate entry found for N=%d ' % elems[-1])
    from scipy.interpolate import interp1d
    f = interp1d(elems, time)
    #f = interp1d(elems, time, kind='cubic')
    return f


def t_transpose():
    if  len(trans_db) == 0:
        sys.exit('Error: daxpy benchmark not found')
    time = [0]
    elems = [[0,0]]
    for line in trans_db:
        temp = line.split(',')
        if True: #temp[-1] != 'synth\n':
           if not (elems[-1][0] == int(temp[0]) and elems[-1][1] == int(temp[1])):
                 time.append(float(temp[2]))
                 elems.append([int(temp[0]),int(temp[1])] )
           else:
                 print('Duplicate entry found for N=%d M=%d' % (elems[-1][0], elems[-1][1]))
    from scipy.interpolate import LinearNDInterpolator
    f = LinearNDInterpolator(elems, time)
    #f = interp1d(elems, time, kind='cubic')
    return f

f_trans = t_transpose()
def t_dtranspose(X,Y):
    if (X <1  or Y < 1):
        return 0
    else:
        return f_trans([X,Y])

def t_cpu_gemm():
    if  len(cpu_gemm_db) == 0:
        sys.exit('Error: t_dgemm_cpu benchmark not found')
    time = [0]
    elems = [[0,0,0]]
    for line in cpu_gemm_db:
        temp = line.split(',')
        if True: #temp[-1] != 'synth\n':
           if not (elems[-1][0] == int(temp[0]) and elems[-1][1] == int(temp[1]) and elems[-1][2] == int(temp[2])):
                 time.append(float(temp[19]))
                 elems.append([int(temp[0]),int(temp[1]), int(temp[2])] )
           else:
                 print('Duplicate entry found for N=%d M=%d K=%d' % (elems[-1][0], elems[-1][1], elems[-1][2]))
    from scipy.interpolate import LinearNDInterpolator
    f = LinearNDInterpolator(elems, time)
    #f = interp1d(elems, time, kind='cubic')
    return f

f_cpu_gemm = t_cpu_gemm()
def t_dgemm_cpu(M, N, K):
    if (M < 1 or N <1 or K <1):
        return 0
    else:
        return f_cpu_gemm([M,N,K])


def t_gpu_gemm():
    if  len(gpu_gemm_db) == 0:
        sys.exit('Error: t_dgemm_cpu benchmark not found')
    time = [0]
    elems = [[0,0,0]]
    for line in gpu_gemm_db:
        temp = line.split(',')
        if True: #temp[-1] != 'synth\n':
           if not (elems[-1][0] == int(temp[0]) and elems[-1][1] == int(temp[1]) and elems[-1][2] == int(temp[2])):
                 time.append(float(temp[19]))
                 elems.append([int(temp[0]),int(temp[1]), int(temp[2])] )
           else:
                 print('Duplicate entry found for N=%d M=%d K=%d' % (elems[-1][0], elems[-1][1], elems[-1][2]))
    from scipy.interpolate import LinearNDInterpolator
    f = LinearNDInterpolator(elems, time)
    #f = interp1d(elems, time, kind='cubic')
    return f

f_gpu_gemm = t_gpu_gemm()
def t_dgemm_gpu(M, N, K):
    if (M < 1 or N <1 or K <1):
        return 0
    else:
        return f_gpu_gemm([M,N,K])

f_send = t_transfer_to_gpu()
f_recv = t_transfer_from_gpu()
def t_memcpy(bytes, src, dest):
    if(bytes < 0):
        return float("inf")
        # For now only for dev0 <-> host
    if (src == -1 and dest == 0):
        return f_send(bytes)
    if (src == 0 and dest == -1):
        return f_recv(bytes)


def t_copy_vec_1d(N, src, dest, elem_size):
    if N < 1:
        return 0
    return t_memcpy(N * elem_size, src, dest)


def t_copy_vec_2d(dim1, dim2, ldim, src, dest, elem_size):
    if ldim == dim2:
        return t_memcpy(dim1 * dim2 * elem_size, src, dest)
    elif ldim > dim2:
        return dim1 * t_memcpy(dim2 * elem_size, src, dest)
    else:
        sys.exit('Error: dim2(%d) > ldim(%d)' % (dim2, ldim))

f_daxpy = t_dadd()
def t_add_vec_1d(N, elem_size):
    if N < 1:
        return 0
    # For now assume 8 bytes -> double
    if elem_size == 8:
        return f_daxpy(N)
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
    M_cpu = M - M_gpu
    if A_mem:
        A_dim1, A_dim2, ldim = K, M_gpu, M
    else:
        A_dim1, A_dim2, ldim = M_gpu, K, K
    t_transfer_A = t_copy_vec_2d(A_dim1, A_dim2, ldim, -1, 0, elem_size)
    if B_mem:
        B_dim1, B_dim2, ldim = N, K, K
    else:
        B_dim1, B_dim2, ldim = K, N, N
    if M_gpu > 0:
        t_transfer_B = t_copy_vec_2d(B_dim1, B_dim2, ldim, -1, 0, elem_size)
    else:
        t_transfer_B = 0
    if C_mem:
        C_dim1, C_dim2, ldim = N, M_gpu, M
    else:
        C_dim1, C_dim2, ldim = M_gpu, N, N
    t_get_C = t_copy_vec_2d(C_dim1, C_dim2, ldim, 0, -1, elem_size)
    t_transfer_C = t_copy_vec_2d(C_dim1, C_dim2, ldim, -1, 0, elem_size)

    t_gpu_overhead = t_transfer_A + t_transfer_B + t_transfer_C + t_get_C
    t_cpu_overhead =  (1 - C_mem) * (t_dtranspose(M_gpu, N) + t_dtranspose(N, M_gpu))
    t_total_M_split = t_gpu_overhead + t_cpu_overhead + \
        max(t_dgemm_gpu(M_gpu, N, K), t_dgemm_cpu(M_cpu, N, K))
    print(M_split, t_total_M_split)
    return t_total_M_split


def Impl_N_split(N_split, M, N, K, A_mem, B_mem, C_mem, elem_size):
    N_gpu = N_split
    N_cpu = N - N_gpu
    if A_mem:
        A_dim1, A_dim2, ldim = K, M, M
    else:
        A_dim1, A_dim2, ldim = M, K, K
    if N_gpu > 0:
        t_transfer_A = t_copy_vec_2d(A_dim1, A_dim2, ldim, -1, 0, elem_size)
    else:
        t_transfer_A = 0
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
    t_cpu_overhead =  (1 - C_mem) * (t_dtranspose(M, N_gpu) + t_dtranspose(N_gpu, M))
    t_total_N_split = t_gpu_overhead + t_cpu_overhead + \
        max(t_dgemm_gpu(M, N_gpu, K), t_dgemm_cpu(M, N_cpu, K))
    print(N_split, t_total_N_split)
    return t_total_N_split


def Impl_K_split(K_split, M, N, K, A_mem, B_mem, C_mem, elem_size):
    K_gpu = K_split
    K_cpu = K - K_gpu
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
    if K_gpu > 0:
        t_get_C = t_copy_vec_2d(C_dim1, C_dim2, ldim, 0, -1, elem_size)
        t_transfer_C = t_copy_vec_2d(C_dim1, C_dim2, ldim, -1, 0, elem_size)
        t_cpu_overhead = (1 - C_mem) * (t_dtranspose(M, N) +
                                             t_dtranspose(N, M)) + t_add_vec_1d(M * N, elem_size)
    else:
        t_transfer_C = t_get_C = t_cpu_overhead = 0
    t_gpu_overhead = t_transfer_A + t_transfer_B + t_transfer_C + t_get_C
    
    t_total_K_split = t_gpu_overhead + t_cpu_overhead + \
        max(t_dgemm_gpu(M, N, K_gpu), t_dgemm_cpu(M, N, K_cpu))
    print(K_split, t_total_K_split)
    return t_total_K_split


def Wrap_any_split(ratio, M, N, K, A_mem, B_mem, C_mem, elem_size):
   return min([Impl_K_split(int(ratio*K), M, N, K, A_mem, B_mem, C_mem, elem_size), Impl_M_split(int(ratio*M), M, N, K, A_mem, B_mem, C_mem, elem_size), Impl_N_split(int(ratio*N), M, N, K, A_mem, B_mem, C_mem, elem_size)])

def plot_daxpy(plot_up_bound):
    plot_down_bound = f_daxpy(0)
    xnew = np.linspace(plot_down_bound, plot_up_bound, num=41, endpoint=True)
    import matplotlib.pyplot as plt
    plt.plot(xnew,GigaVal_per_s(xnew, f_daxpy(xnew)), '-')
    # plt.xscale('log')
    plt.legend([ 'CPU daxpy' ], loc='best')
    plt.ylabel('Gflops/s')
    plt.xlabel('Size')
    plt.title('Daxpy (0-' + str(plot_up_bound) + ')')
    plt.savefig('daxpy_gold1_' + str(plot_up_bound) + '.eps', format='eps')
    plt.close()

def plot_bw(plot_up_bound):
    plot_down_bound = 0
    xnew = np.linspace(plot_down_bound, plot_up_bound, num=41, endpoint=True)
    import matplotlib.pyplot as plt
    plt.plot(xnew,GigaVal_per_s(xnew, f_send(xnew)), '-', xnew,GigaVal_per_s(xnew, f_recv(xnew)), '-',  xnew, GigaVal_per_s(xnew, xnew/(16e9)), '--', xnew, GigaVal_per_s(xnew, xnew/(8e9)), '--')
    # plt.xscale('log')
    plt.legend([ 'Host->Device', 'Device->Host', 'theoretical 2-way(16Gb/s)', 'theoretical(8Gb/s)'], loc='best')
    plt.ylabel('Gb/s')
    plt.xlabel('Bytes')
    plt.title('PCI-e Bandwidth (0-' + str(plot_up_bound) + ' bytes)')
    plt.savefig('Bandwidth_gold1_' + str(plot_up_bound) + '_bytes.eps', format='eps')
    plt.close()

def plot_transpose(plot_up_bound):
    plot_down_bound = 0
    xnew = np.linspace(plot_down_bound, plot_up_bound, 40)
    ynew = np.linspace(plot_down_bound, plot_up_bound, 40)
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    X, Y = np.meshgrid(xnew, ynew)
    ax.plot_wireframe(X,Y,GigaVal_per_s(xnew*ynew, f_trans(X,Y)), color = 'c')
    # plt.xscale('log')
    #plt.legend('CPU transpose', loc='best')
    ax.set_xlabel('M')
    ax.set_ylabel('N')
    ax.set_zlabel('Gflops/s')
    ax.view_init(60, 35)
    plt.title('Transpose (0-' + str(plot_up_bound) + '/'+ str(plot_up_bound) + ' )')
    plt.savefig('Transpose_gold1__3d' + str(plot_up_bound) + '.eps', format='eps')
    plt.close()
    plt.plot(xnew,GigaVal_per_s(xnew*plot_up_bound, f_trans(plot_up_bound,xnew)), '-', xnew,GigaVal_per_s(xnew*plot_up_bound, f_trans(xnew,plot_up_bound)), '-',)
    plt.legend([ 'Transpose Host (X=' + str(plot_up_bound) + ',Y=Size)' , 'Transpose Host (Y=' + str(plot_up_bound) + ',X=Size)'], loc='best')
    plt.ylabel('Gflops/s')
    plt.xlabel('Size')
    plt.title('Transpose 1d-fied (0-' + str(plot_up_bound) + '/'+ str(plot_up_bound) + ' )')
    plt.savefig('Transpose_gold1__1d' + str(plot_up_bound) + '.eps', format='eps')
    plt.close()




dtype_size = 8
t_min = 100

N = M = K = 10000
x = np.ndarray(1)
x[0]=M/2
plot_bw(1e5)
plot_bw(1e6)
plot_daxpy(1e5)
plot_daxpy(1e6)
plot_daxpy(1e7)
plot_transpose(1e3)
plot_transpose(1e4)
answer = scipop.minimize(Impl_N_split, x, args=(M, N, K, 0, 0, 1, 8),  bounds = [(0,M)])
print(f_cpu_gemm(M,N,K))
print(f_cpu_gemm(M,int(M-answer.x[0]),K))
print(f_gpu_gemm(M, N - int(M-answer.x[0]),K))
print(answer)



