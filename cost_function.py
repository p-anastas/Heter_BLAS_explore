import subprocess
import sys
import numpy as np
import scipy.optimize as scipop
import scipy as sc
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from bandwidth_functions import t_copy_vec_1d, t_copy_vec_2d
from transpose_functions import t_dtranspose
from dgemm_functions import t_dgemm_cpu, t_dgemm_gpu

machine = 'dungani'
resDir = 'Results_' + machine + '/'

# For now use artificial bound to check
bounds = [1e4,1e5,1e6,8e9]
flop_bounds = [1e6,1e7,1e8,1e9,8e9]

with open(resDir + 'daxpy_log_' + machine + '.md_sorted', "r") as file0:
    add_db = file0.readlines()

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


def Impl_M_split(M_split, M, N, K, A_mem, B_mem, C_mem, elem_size,printflag):
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
    if printflag:
        print('Output M_split Split=%d Μ=%d Ν=%d Κ=%d A_mem=%d B_mem=%d C_mem=%d elem_sz=%d:' % ( M_split, M, N, K, A_mem, B_mem, C_mem, elem_size))
        print('t_transfer = %.5lf (t_transfer_A = %.5lf, t_transfer_B = %.5lf, t_transfer_C = %.5lf, t_get_C = %.5lf)' % (t_gpu_overhead, t_transfer_A, t_transfer_B,t_transfer_C,t_get_C))
        print('t_transform = %.5lf' % t_cpu_overhead)
        print('t_exec = %.5lf(t_dgemm_gpu = %.5lf, t_dgemm_cpu = %.5lf)\n' % (max(t_dgemm_gpu(M_gpu, N, K), t_dgemm_cpu(M_cpu, N, K)), t_dgemm_gpu(M_gpu, N, K), t_dgemm_cpu(M_cpu, N, K)))
    return t_total_M_split


def Impl_N_split(N_split, M, N, K, A_mem, B_mem, C_mem, elem_size,printflag):
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
    if printflag:
        print('Output N_split Split=%d Μ=%d Ν=%d Κ=%d A_mem=%d B_mem=%d C_mem=%d elem_sz=%d:' % ( N_split, M, N, K, A_mem, B_mem, C_mem, elem_size))
        print('t_transfer = %.5lf (t_transfer_A = %.5lf, t_transfer_B = %.5lf, t_transfer_C = %.5lf, t_get_C = %.5lf)' % (t_gpu_overhead, t_transfer_A, t_transfer_B,t_transfer_C,t_get_C))
        print('t_transform = %.5lf' % t_cpu_overhead)
        print('t_exec = %.5lf(t_dgemm_gpu = %.5lf, t_dgemm_cpu = %.5lf)\n' % (max(t_dgemm_gpu(M, N_gpu, K), t_dgemm_cpu(M, N_cpu, K)), t_dgemm_gpu(M, N_gpu, K), t_dgemm_cpu(M, N_cpu, K)))
    return t_total_N_split


def Impl_K_split(K_split, M, N, K, A_mem, B_mem, C_mem, elem_size,printflag):
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
    if printflag:
        print('Output K_split Split=%d Μ=%d Ν=%d Κ=%d A_mem=%d B_mem=%d C_mem=%d elem_sz=%d:' % ( K_split, M, N, K, A_mem, B_mem, C_mem, elem_size))
        print('t_transfer = %.5lf (t_transfer_A = %.5lf, t_transfer_B = %.5lf, t_transfer_C = %.5lf, t_get_C = %.5lf)' % (t_gpu_overhead, t_transfer_A, t_transfer_B,t_transfer_C,t_get_C))
        print('t_transform = %.5lf' % t_cpu_overhead)
        print('t_exec = %.5lf(t_dgemm_gpu = %.5lf, t_dgemm_cpu = %.5lf)\n' % (max(t_dgemm_gpu(M, N, K_gpu), t_dgemm_cpu(M, N, K_cpu)), t_dgemm_gpu(M, N, K_gpu), t_dgemm_cpu(M, N, K_cpu)))
    return t_total_K_split

dtype_size = 8
N = M = K = 10000
A_mem = 1
B_mem = 1
C_mem = 1
x = np.ndarray(1)
x[0]=0

# Get past local minimums for now:
answer = scipop.minimize(Impl_M_split, x, args=(M, N, K, A_mem, B_mem, C_mem, 8,False),  bounds = [(0,M)])
for x0 in range(int(M/10)-1, M, int(M/10)):
    x[0] = x0
    temp =  scipop.minimize(Impl_M_split, x, args=(M, N, K, A_mem, B_mem, C_mem, 8,False),  bounds = [(0,M)])
    if temp.fun[0]< answer.fun[0]:
        answer = temp
        print('Predicted M_split(%d,%d,%d): x0=%d -> Split = %d , t = %.5lf s (gemm_cpu=%.5lf s, gemm_gpu=%.5lf s)' % (M,N,K,x0, int(answer.x[0]) , answer.fun[0], t_dgemm_cpu(M -int(answer.x[0]),N,K), t_dgemm_gpu(int(answer.x[0]),N,K)))
print('Predicted M_split(%d,%d,%d) -> Split = %d , t = %.5lf ms (gemm_cpu=%.5lf s, gemm_gpu=%.5lf s)\n' % (M,N,K, int(answer.x[0]) , answer.fun[0], t_dgemm_cpu(M -int(answer.x[0]),N,K), t_dgemm_gpu(int(answer.x[0]),N,K)))
Impl_M_split(int(answer.x[0]), M, N, K, A_mem, B_mem, C_mem, 8,True)
x[0]=0
answer1 = scipop.minimize(Impl_N_split, x, args=(M, N, K, A_mem, B_mem, C_mem, 8,False),  bounds = [(0,N)])
for x0 in range(int(N/10)-1, N, int(N/10)):
    x[0] = x0
    temp =  scipop.minimize(Impl_N_split, x, args=(M, N, K, A_mem, B_mem, C_mem, 8,False),  bounds = [(0,N)])
    if temp.fun[0]< answer1.fun[0]:
        answer1 = temp
        print('Predicted N_split(%d,%d,%d): x0=%d -> Split = %d , t = %.5lf s (gemm_cpu=%.5lf s, gemm_gpu=%.5lf s)' % (M,N,K,x0, int(answer1.x[0]) , answer1.fun[0], t_dgemm_cpu(M,N - int(answer1.x[0]),K), t_dgemm_gpu(M,int(answer1.x[0]),K)))
print('Predicted N_split(%d,%d,%d) -> Split = %d , t = %.5lf s (gemm_cpu=%.5lf s, gemm_gpu=%.5lf s)\n' % (M,N,K, int(answer1.x[0]) , answer1.fun[0], t_dgemm_cpu(M,N - int(answer1.x[0]),K), t_dgemm_gpu(M,int(answer1.x[0]),K)))
Impl_N_split(int(answer1.x[0]), M, N, K, A_mem, B_mem, C_mem, 8,True)

x[0]=0
answer2 = scipop.minimize(Impl_K_split, x, args=(M, N, K, A_mem, B_mem, C_mem, 8,False),  bounds = [(0,K)])
for x0 in range(int(K/10)-1, K, int(K/10)):
    x[0] = x0
    temp =  scipop.minimize(Impl_K_split, x, args=(M, N, K, A_mem, B_mem, C_mem, 8,False),  bounds = [(0,K)])
    if temp.fun[0]< answer2.fun[0]:
        answer2 = temp
        print('Predicted K_split(%d,%d,%d): x0=%d -> Split = %d , t = %.5lf s (gemm_cpu=%.5lf s, gemm_gpu=%.5lf s)' % (M,N,K,x0, int(answer2.x[0]) , answer2.fun[0], t_dgemm_cpu(M,N, K - int(answer2.x[0])), t_dgemm_gpu(M,N,int(answer2.x[0]))))
print('Predicted K_split(%d,%d,%d) -> Split = %d , t = %.5lf s (gemm_cpu=%.5lf s, gemm_gpu=%.5lf s)\n' % (M,N,K, int(answer2.x[0]) , answer2.fun[0], t_dgemm_cpu(M,N, K - int(answer2.x[0])), t_dgemm_gpu(M,N,int(answer2.x[0]))))
Impl_K_split(int(answer2.x[0]), M, N, K, A_mem, B_mem, C_mem, 8,True)

#outfile = resDir + 'Predicted_' +str(M) + '_' + str(N) + '_' + str(K) + '_' + str(A_mem) + '_' + str(B_mem) + '_' + str(C_mem) + '.out'
#Runner = './build/dgemm_runner ' +  str(M) + ' ' + str(N) + ' ' + str(K) + ' ' + str(A_mem) + ' ' + str(B_mem) + ' ' + str(C_mem) + ' 1.345 1.234 0'
#proc = 'echo "' + Runner + ' ' + str(int(answer.x[0])) + ' 0 0 BENCHMARK" > ' + outfile
#if (M * N + N * K + M * K) * 8 < 8e9:
#    process = subprocess.call(proc, shell=True)
#proc = 'echo "' + Runner + ' 0 ' + str(int(answer1.x[0])) + ' 0 BENCHMARK" >> ' + outfile
#if (M * N + N * K + M * K) * 8 < 8e9:
#    process = subprocess.call(proc, shell=True)
#proc = 'echo "' + Runner + ' 0 0 ' + str(int(answer2.x[0])) + ' BENCHMARK" >> ' + outfile
#if (M * N + N * K + M * K) * 8 < 8e9:
#    process = subprocess.call(proc, shell=True)



