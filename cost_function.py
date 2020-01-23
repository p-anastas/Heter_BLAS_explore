import subprocess
import sys
import numpy as np
import scipy.optimize as scipop
import scipy as sc
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from bandwidth_functions import t_copy_vec_1d, t_copy_vec_2d
from transpose_functions import t_dtranspose

machine = 'dungani'
resDir = 'Results_' + machine + '/'

# For now use artificial bound to check
bounds = [1e4,1e5,1e6,8e9]
flop_bounds = [1e6,1e7,1e8,1e9,8e9]

with open(resDir + 'daxpy_log_' + machine + '.md_sorted', "r") as file0:
    add_db = file0.readlines()

with open(resDir + 'CPU_only_log_' + machine + '.md_sorted', "r") as file0:
    cpu_gemm_db = file0.readlines()

with open(resDir + 'GPU_only_log_' + machine + '.md_sorted', "r") as file0:
    gpu_gemm_db = file0.readlines()

def report_bandwidth(bytes):
    cpu_to_gpu_time = t_copy_vec_1d(bytes, -1, 0,8)
    gpu_to_cpu_time = t_copy_vec_1d(bytes, 0, -1,8)
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

def GigaVal_per_s_l(l_val, l_time):	
    answer = []
    for val, time in zip(l_val, l_time):
        if time!= 0:
            answer.append(val * 1e-9 / time)
        elif val!=0:
            sys.exit('Error: GigaVal_per_s_l called with elem val=%d time=%d' % (val,time))
        else:
            answer.append(0)		
    return answer

def linearize_dadd():
    if  len(add_db) == 0:
        sys.exit('Error: daxpy benchmark not found')
    time = []
    elems = []
    for line in add_db:
        temp = line.split(',')
        if True: #temp[-1] != 'synth\n':
             time.append(float(temp[4]))
             elems.append(int(temp[0]))
    prev_bound = 0
    linear_subparts=[]
    for bound in bounds:
        bounded_times = []
        bounded_N = []
        for qN, qtime in zip(elems, time):
            if (qN < bound and qN >= prev_bound):
                bounded_N.append(qN)
                bounded_times.append(qtime)
        x = np.array(bounded_N).reshape(-1, 1)
        y = np.array(bounded_times)
        model = LinearRegression(n_jobs=-1).fit(x,y)
        r_sq = model.score(x, y)
        print('coefficient of determination:', r_sq)
        if model.intercept_ < 0:
             print('Negative intercept:', model.intercept_)
             model.intercept_ = 0
        if model.coef_ < 0:
             print('Negative coef:', model.coef_)
             sys.exit('Error: Negative coef too much for now')
        print('slope:', model.coef_)
        linear_subparts.append(model.predict)
        prev_bound = bound
    return linear_subparts

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

def t_cpu_gemm_lin():
    if  len(cpu_gemm_db) == 0:
        sys.exit('Error: t_dgemm_cpu benchmark not found')
    time = []
    flops = []
    for line in cpu_gemm_db:
        temp = line.split(',')
        if True: #temp[-1] != 'synth\n':
           if not (int(temp[0])*int(temp[1])*int(temp[2]) in flops):
                 time.append(float(temp[19]))
                 flops.append(int(temp[0])*int(temp[1])*int(temp[2]))
           else:
                 print('Duplicate entry found for N=%d M=%d K=%d' % (int(temp[0]),  int(temp[1]), int(temp[2])))
    prev_bound = 0
    linear_subparts=[]
    for bound in flop_bounds:
        bounded_times = []
        bounded_flops = []
        for qflop, qtime in zip(flops, time):
            if (qflop < bound and qflop >= prev_bound):
                bounded_flops.append(qflop)
                bounded_times.append(qtime)
        x = np.array(bounded_flops).reshape(-1, 1)
        y = np.array(bounded_times)
        if len(x) != 0:
            model = LinearRegression(n_jobs=-1).fit(x,y)
            r_sq = model.score(x, y)
            print('coefficient of determination:', r_sq)
            if model.intercept_ < 0:
                 print('Negative intercept:', model.intercept_)
                 model.intercept_ = 0
            if model.coef_ < 0:
                 print('Coef is Negattive !!!!')
            print('coef:', model.coef_)
            print('slope:', model.coef_)
            linear_subparts.append(model.predict)
        else:
            print('No benchmarks found for bound %d' %bound)
            linear_subparts.append(lambda x: float('inf'))
        prev_bound = bound
    return linear_subparts

f_cpu_gemm = t_cpu_gemm()
def t_dgemm_cpu(M, N, K):
    if (M < 1 or N <1 or K <1):
        return 0
    else:
        return f_cpu_gemm([M,N,K])

f_cpu_gemm_lin = t_cpu_gemm_lin()
def t_dgemm_cpu_lin(flops):
    if (flops < 1):
        return 0
    else:
        ctr = 0
        for bound in flop_bounds:
            if flops < bound:
                return f_cpu_gemm_lin[ctr](np.array(flops).reshape(-1, 1))
            ctr +=1
        return f_cpu_gemm_lin[ctr-1](np.array(flops).reshape(-1, 1))


def t_gpu_gemm():
    if  len(gpu_gemm_db) == 0:
        sys.exit('Error: t_dgemm_cpu benchmark not found')
    time = [0]
    elems = [[0,0,0]]
    for line in gpu_gemm_db:
        temp = line.split(',')
        if True: #temp[-1] != 'synth\n':
           if not (elems[-1][0] == int(temp[0]) and elems[-1][1] == int(temp[1]) and elems[-1][2] == int(temp[2])):
                 time.append(float(temp[18]))
                 elems.append([int(temp[0]),int(temp[1]), int(temp[2])] )
           else:
                 print('Duplicate entry found for N=%d M=%d K=%d' % (elems[-1][0], elems[-1][1], elems[-1][2]))
    from scipy.interpolate import LinearNDInterpolator
    f = LinearNDInterpolator(elems, time)
    #f = interp1d(elems, time, kind='cubic')
    return f

def t_gpu_gemm_lin():
    if  len(gpu_gemm_db) == 0:
        sys.exit('Error: t_dgemm_cpu benchmark not found')
    time = []
    flops = []
    for line in gpu_gemm_db:
        temp = line.split(',')
        if True: #temp[-1] != 'synth\n':
           if not (int(temp[0])*int(temp[1])*int(temp[2]) in flops):
                 time.append(float(temp[18]))
                 flops.append(int(temp[0])*int(temp[1])*int(temp[2]))
           else:
                 print('Duplicate entry found for N=%d M=%d K=%d' % (int(temp[0]),  int(temp[1]), int(temp[2])))
    prev_bound = 0
    linear_subparts=[]
    for bound in flop_bounds:
        bounded_times = []
        bounded_flops = []
        for qflop, qtime in zip(flops, time):
            if (qflop < bound and qflop >= prev_bound):
                bounded_flops.append(qflop)
                bounded_times.append(qtime)
        x = np.array(bounded_flops).reshape(-1, 1)
        y = np.array(bounded_times)
        if len(x) != 0:
            model = LinearRegression(n_jobs=-1).fit(x,y)
            r_sq = model.score(x, y)
            print('coefficient of determination:', r_sq)
            if model.intercept_ < 0:
                 print('Negative intercept:', model.intercept_)
                 model.intercept_ = 0
            if model.coef_ < 0:
                 print('Coef is Negattive !!!!')
            print('coef:', model.coef_)
            print('slope:', model.coef_)
            linear_subparts.append(model.predict)
        else:
            print('No benchmarks found for bound %d' %bound)
            linear_subparts.append(lambda x: float('inf'))
        prev_bound = bound
    return linear_subparts

f_gpu_gemm = t_gpu_gemm()
def t_dgemm_gpu(M, N, K):
    if (M < 1 or N <1 or K <1):
        return 0
    else:
        return f_gpu_gemm([M,N,K])

f_gpu_gemm_lin = t_gpu_gemm_lin()
def t_dgemm_gpu_lin(flops):
    if (flops < 1):
        return 0
    else:
        ctr = 0
        for bound in flop_bounds:
            if flops < bound:
                return f_gpu_gemm_lin[ctr](np.array(flops).reshape(-1, 1))
            ctr +=1
        return f_gpu_gemm_lin[ctr-1](np.array(flops).reshape(-1, 1))

f_daxpy = linearize_dadd()
def t_add_vec_1d(N, elem_size):
    if N < 1:
        return 0
    # For now assume 8 bytes -> double
    if elem_size == 8:
        ctr = 0
        for bound in bounds:
            if N < bound:
                return f_daxpy[ctr](np.array(N).reshape(-1, 1))
            ctr +=1
        return f_daxpy[-1](np.array(N).reshape(-1, 1))
    else:
        sys.exit('Error: t_add_vec_1d(%d,%d) -> unknown elem_size' % (N, elem_size))


def t_add_vec_2d(dim1, dim2, ldim, elem_size):
    if ldim == dim2:
        return t_add_vec_1d(dim1 * dim2, elem_size)
    elif ldim > dim2:
        return dim1 * t_add_vec_1d(dim2, elem_size)
    else:
        # This is wrong but functions hate nans
        return t_add_vec_1d(dim1 * dim2, elem_size)
        #return float("inf")
        #sys.exit('Error: dim2(%d) > ldim(%d)' % (dim2, ldim))


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
        max(t_dgemm_gpu(M_gpu, N, K), t_dgemm_cpu(M_cpu, N, K))# max(t_dgemm_gpu_lin(M_gpu*N*K), t_dgemm_cpu_lin(M_cpu*N*K)) #
    #print(M_split, t_total_M_split)
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
    t_cpu_overhead =  (1 - C_mem) *  (t_dtranspose(M, N_gpu) + t_dtranspose(N_gpu, M))
    t_total_N_split = t_gpu_overhead + t_cpu_overhead +  max(t_dgemm_gpu(M, N_gpu, K), t_dgemm_cpu(M, N_cpu, K))#max(t_dgemm_gpu_lin(M*N_gpu*K), t_dgemm_cpu_lin(M*N_cpu*K))#
    #print(N_split, t_total_N_split)
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
        t_cpu_overhead = t_add_vec_1d(M * N, elem_size) + (1 - C_mem) * (t_dtranspose(M, N) + t_dtranspose(N, M))  
    else:
        t_transfer_C = t_get_C = t_cpu_overhead = 0
    t_gpu_overhead = t_transfer_A + t_transfer_B + t_transfer_C + t_get_C
    
    t_total_K_split = t_gpu_overhead + t_cpu_overhead + \
        max(t_dgemm_gpu(M, N, K_gpu), t_dgemm_cpu(M, N, K_cpu)) #max(t_dgemm_gpu_lin(M*N*K_gpu), t_dgemm_cpu_lin(M*N*K_cpu))# 
    #print(K_split, t_total_K_split)
    return t_total_K_split


def Wrap_any_split(split_val, M, N, K, A_mem, B_mem, C_mem, elem_size):
   return min([Impl_K_split(split_val, M, N, K, A_mem, B_mem, C_mem, elem_size), Impl_M_split(split_val, M, N, K, A_mem, B_mem, C_mem, elem_size), Impl_N_split(split_val, M, N, K, A_mem, B_mem, C_mem, elem_size)])

def plot_daxpy(plot_up_bound):
    plot_down_bound = 0
    xnew = np.linspace(plot_down_bound, plot_up_bound, num=200, endpoint=True)
    time = [0]
    N = [0]
    for line in add_db:
        temp = line.split(',')
        if int(temp[0])<plot_up_bound: #temp[-1] != 'synth\n':
           if N[-1] != int(temp[0]):
                 time.append(float(temp[4]))
                 N.append(int(temp[0]))
    import matplotlib.pyplot as plt
    #plt.plot(N, GigaVal_per_s_l(N,time), 'o', xnew, GigaVal_per_s_l(xnew, list(map(lambda x: t_add_vec_1d(x,8), xnew))), '-')
    plt.plot(N, time, 'o', N, list(map(lambda x: t_add_vec_1d(x,8), N)), '-')
    # plt.xscale('log')
    plt.legend([ 'CPU daxpy(Samples)', 'CPU daxpy(Hybrid LR)'], loc='best')
    plt.ylabel('Gflops/s')
    plt.xlabel('Size')
    plt.title('Daxpy (0-' + str(plot_up_bound) + ')')
    plt.savefig('daxpy_' + machine +'_' + str(plot_up_bound) + '.eps', format='eps')
    plt.close()

def plot_bw(plot_up_bound):
    plot_down_bound = 0
    xnew = np.linspace(plot_down_bound, plot_up_bound, num=41, endpoint=True)
    time = [[0],[0]]
    bytes = [[0],[0]]
    for line in bw_db:
        temp = line.split(',')
        if int(temp[1]) == -1 and int(temp[2]) == 0 and int(temp[0])<plot_up_bound: #temp[-1] != 'synth\n':
           if bytes[0][-1] != int(temp[0]):
                 time[0].append(float(temp[3]))
                 bytes[0].append(int(temp[0]))
        elif int(temp[1]) == 0 and int(temp[2]) == -1 and int(temp[0])<plot_up_bound:
           if bytes[1][-1] != int(temp[0]):
                 time[1].append(float(temp[3]))
                 bytes[1].append(int(temp[0]))
    import matplotlib.pyplot as plt
    plt.plot(bytes[0], GigaVal_per_s_l(bytes[0],time[0]), 'o', bytes[0], GigaVal_per_s_l(bytes[0], list(map(lambda x: t_copy_vec_1d(x,-1,0,8), bytes[0]))), '-', bytes[1], GigaVal_per_s_l(bytes[1],time[1]), 'o', bytes[1], GigaVal_per_s_l(bytes[1],list(map(lambda x: t_copy_vec_1d(x,0,-1,8), bytes[1]))), '-')
    # plt.xscale('log')
    plt.legend(['Host->Dev(Samples)', 'Host->Dev(Hybrid LR)', 'Dev->Host(Samples)','Dev->Host(Hybrid LR)'], loc='best')
    plt.ylabel('Gb/s')
    plt.xlabel('Bytes')
    plt.title('PCI-e Bandwidth (0-' + str(plot_up_bound) + ' bytes)')
    plt.savefig('Bandwidth_gold1_' + str(plot_up_bound) + '_bytes.eps', format='eps')
    plt.close()

def plot_transpose(plot_up_bound):
    plot_down_bound = 0
    xnew = np.linspace(plot_down_bound, plot_up_bound, 40)
    time = [0]
    flops= [0]
    for line in trans_db:
        temp = line.split(',')
        if int(temp[0])*int(temp[1])< plot_up_bound: #temp[-1] != 'synth\n':
           if not (int(temp[0])*int(temp[1]) in flops):
                time.append(float(temp[2]))
                flops.append(int(temp[0])*int(temp[1]))
    flops, time = zip(*sorted(zip(flops, time)))
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    #fig = plt.figure()
    #ax = plt.axes(projection='3d')
    #X, Y = np.meshgrid(xnew, ynew)
    #ax.plot_wireframe(X,Y,GigaVal_per_s_l(xnew*ynew, list(map(lambda x,y: t_dtranspose(x,y), X,Y))), color = 'c')
    # plt.xscale('log')
    #plt.legend('CPU transpose', loc='best')
    #ax.set_xlabel('M')
    #ax.set_ylabel('N')
    #ax.set_zlabel('Gflops/s')
    #ax.view_init(60, 35)
    #plt.title('Transpose (0-' + str(plot_up_bound) + '/'+ str(plot_up_bound) + ' )')
    #plt.savefig('Transpose_'+machine+'_3d' + str(plot_up_bound) + '.eps', format='eps')
    #plt.close()
    plt.plot(flops, GigaVal_per_s_l(flops,time), 'o', flops, GigaVal_per_s_l(flops,list(map(lambda x: t_dtranspose_lin(x),flops))) , '-' )
    plt.legend(['Transpose Host (Samples)', 'Transpose Host (LR total flops)' ], loc='best')
    plt.ylabel('Gflops/s')
    plt.xlabel('Size')
    plt.title('Transpose 1d-fied (0-' + str(plot_up_bound) + '/'+ str(plot_up_bound) + ' )')
    plt.savefig('Transpose_'+machine+'_1d' + str(plot_up_bound) + '.eps', format='eps')
    plt.close()

dtype_size = 8
t_min = 100

N = M = K = 10000
A_mem = 0
B_mem = 0
C_mem = 0
x = np.ndarray(1)
x[0]=M-1
#plot_bw(1e4)
#plot_bw(1e5)
#plot_bw(1e6)
#plot_daxpy(1e5)
#plot_daxpy(1e6)
#plot_daxpy(1e7)
#plot_transpose(1e5)
#plot_transpose(1e6)
#plot_transpose(1e7)

answer = scipop.minimize(Impl_M_split, x, args=(M, N, K, A_mem, B_mem, C_mem, 8),  bounds = [(0,M)])
answer1 = scipop.minimize(Impl_N_split, x, args=(M, N, K, A_mem, B_mem, C_mem, 8),  bounds = [(0,N)])
answer2 = scipop.minimize(Impl_K_split, x, args=(M, N, K, A_mem, B_mem, C_mem, 8),  bounds = [(0,K)])
outfile = resDir + 'Predicted_' +str(M) + '_' + str(N) + '_' + str(K) + '_' + str(A_mem) + '_' + str(B_mem) + '_' + str(C_mem) + '.out'
Runner = './build/dgemm_runner ' +  str(M) + ' ' + str(N) + ' ' + str(K) + ' ' + str(A_mem) + ' ' + str(B_mem) + ' ' + str(C_mem) + ' 1.345 1.234 0'
print('Predicted M_split(%d,%d,%d): %d -> t = %.5lf ms (gemm_cpu=%.5lf ms, gemm_gpu=%.5lf ms)' % (M,N,K, int(answer.x[0]) , answer.fun[0], t_dgemm_cpu(M -int(answer.x[0]),N,K), t_dgemm_gpu(int(answer.x[0]),N,K)))
proc = 'echo "' + Runner + ' ' + str(int(answer.x[0])) + ' 0 0 BENCHMARK" > ' + outfile
if (M * N + N * K + M * K) * 8 < 8e9:
    process = subprocess.call(proc, shell=True)

print('Predicted N_split(%d,%d,%d): %d -> t = %.5lf ms (gemm_cpu=%.5lf ms, gemm_gpu=%.5lf ms)' % (M,N,K, int(answer1.x[0]) , answer1.fun[0], t_dgemm_cpu(M,N - int(answer1.x[0]),K), t_dgemm_gpu(M,int(answer1.x[0]),K)))
proc = 'echo "' + Runner + ' 0 ' + str(int(answer1.x[0])) + ' 0 BENCHMARK" >> ' + outfile
if (M * N + N * K + M * K) * 8 < 8e9:
    process = subprocess.call(proc, shell=True)

print('Predicted K_split(%d,%d,%d): %d -> t = %.5lf ms (gemm_cpu=%.5lf ms, gemm_gpu=%.5lf ms)' % (M,N,K, int(answer2.x[0]) , answer2.fun[0], t_dgemm_cpu(M,N, K - int(answer2.x[0])), t_dgemm_gpu(M,N,int(answer2.x[0]))))
proc = 'echo "' + Runner + ' 0 0 ' + str(int(answer2.x[0])) + ' BENCHMARK" >> ' + outfile
if (M * N + N * K + M * K) * 8 < 8e9:
    process = subprocess.call(proc, shell=True)



