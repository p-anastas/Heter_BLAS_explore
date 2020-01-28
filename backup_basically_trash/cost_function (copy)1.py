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

def t_bus_recv(M1, N1, M, N, C_mem, elem_size, printflag):
    if C_mem:
        C_dim1, C_dim2, ldim = N1, M1, M
    else:
        C_dim1, C_dim2, ldim = M1, N1, N
    t_get_C = t_copy_vec_2d(C_dim1, C_dim2, ldim, 0, -1, elem_size)
    if printflag:
        print('t_bus_recv(%d,%d) of (%d,%d) with dim(%d) elem_sz=%d: t_total = %.5lf' % ( M1, N1, M, N, C_mem, elem_size, t_get_C))
    return t_get_C

def elems_send(M1, N1, K1, M, N, K, A_mem, B_mem, C_mem, elem_size, printflag):
    elems_send_A = M1*K1
    elems_send_B = K1*N1
    return (elems_send_A, elems_send_B)

def t_bus_send(M1, N1, K1, M, N, K, A_mem, B_mem, C_mem, elem_size, printflag):
    if A_mem:
        A_dim1, A_dim2, ldim = K1, M1, M
    else:
        A_dim1, A_dim2, ldim = M1, K1, K
    t_transfer_A = t_copy_vec_2d(A_dim1, A_dim2, ldim, -1, 0, elem_size)
    if B_mem:
        B_dim1, B_dim2, ldim = N1, K1, K
    else:
        B_dim1, B_dim2, ldim = K1, N1, N
    t_transfer_B = t_copy_vec_2d(B_dim1, B_dim2, ldim, -1, 0, elem_size)
    #if C_mem:
    #    C_dim1, C_dim2, ldim = N, M_gpu, M
    #else:
    #    C_dim1, C_dim2, ldim = M_gpu, N, N
    #t_get_C = t_copy_vec_2d(C_dim1, C_dim2, ldim, 0, -1, elem_size)
    #t_transfer_C = t_copy_vec_2d(C_dim1, C_dim2, ldim, -1, 0, elem_size)
    t_transfer_C = 0 
    t_transfer_total  = t_transfer_A + t_transfer_B + t_transfer_C
    if printflag:
        print('t_bus_send(%d,%d,%d) of (%d,%d,%d) with dims(%d,%d,%d) elem_sz=%d: t_total = %.5lf (A=%.5lf s , B=%.5lf s,C=%.5lfs )' % ( M1, N1, K1, M, N, K, A_mem, B_mem, C_mem, elem_size, t_transfer_total, t_transfer_A, t_transfer_B,t_transfer_C))
    return t_transfer_total

def t_cpu_reduce(M1, N1, M, N, C_mem, elem_size, printflag):
    if C_mem:
        C_dim1, C_dim2, ldim = N1, M1, M
        t_transpose = 0 
    else:
        C_dim1, C_dim2, ldim = M1, N1, N
        t_transpose = t_dtranspose(N1, M1)
    t_add = t_add_vec_2d(C_dim1, C_dim2, ldim, elem_size)
    t_reduce_total = t_add + t_transpose
    if printflag:
        print('t_cpu_reduce(%d,%d) of (%d,%d) with dim(%d) elem_sz=%d: t_total = %.5lf (add=%.5lf, trans=%.5lf)' % ( M1, N1, M, N, C_mem, elem_size, t_reduce_total, t_add, t_transpose))
    return t_reduce_total

def Impl_split(kernel_dims, M, N, K, A_mem, B_mem, C_mem, elem_size, printflag):
    N_kernels = len(kernel_dims)
    t_bus = t_reduce = t_ker_gpu = remainder = sent_A = sent_B = 0   
    for kernel_id in range(N_kernels):
        M1, N1, K1 = kernel_dims[kernel_id]
        temp_A, temp_B = elems_send(M1, N1, K1, M, N, K, A_mem, B_mem, C_mem, elem_size, printflag)
        sent_A += temp_A
        sent_B += temp_B
        if sent_A > M * K and sent_B > M * K:
            t_bus += t_bus_recv(M1, N1, M, N, C_mem, elem_size,printflag)
        elif sent_A > M * K:
            t_bus += t_bus_send(0, N1, K1, M, N, K, A_mem, B_mem, C_mem, elem_size, printflag) + t_bus_recv(M1, N1, M, N, C_mem, elem_size,printflag)
        elif sent_B > M * K:
            t_bus += t_bus_send(M1, 0, K1, M, N, K, A_mem, B_mem, C_mem, elem_size, printflag) + t_bus_recv(M1, N1, M, N, C_mem, elem_size,printflag)
        else:
            t_bus += t_bus_send(M1, N1, K1, M, N, K, A_mem, B_mem, C_mem, elem_size, printflag) + t_bus_recv(M1, N1, M, N, C_mem, elem_size,printflag)
        t_reduce += t_cpu_reduce(M1, N1, M, N, C_mem, elem_size, printflag)
        t_ker_gpu +=  t_dgemm_gpu(M1, N1, K1)
        if ( kernel_id == 0 ): 
            t_ker_gpu +=  t_bus_send(M1, N1, K1, M, N, K, A_mem, B_mem, C_mem, elem_size, printflag)
            t_reduce += t_ker_gpu
        if ( kernel_id ==N_kernels -1 ): 
            t_last = t_cpu_reduce(M1, N1, M, N, C_mem, elem_size, printflag)
    t_total_split = max (t_bus, t_reduce, t_ker_gpu) + t_last
    #print(M_split, t_total_M_split)
    if printflag:
        print('Impl_split of (%d,%d,%d) with dims(%d,%d,%d) elem_sz=%d: t_total = %.5lf (t_bus = %.5lf, t_reduce = %.5lf, t_ker_gpu = %.5lf)' % ( M, N, K, A_mem, B_mem, C_mem, elem_size, t_total_split, t_bus, t_reduce, t_ker_gpu))
    return t_total_split


def Impl_M_split(M_split, M, N, K, A_mem, B_mem, C_mem, elem_size,printflag):
    t_bus = t_bus_send(M_split, N, K, M, N, K, A_mem, B_mem, C_mem, elem_size,printflag) + t_bus_recv(M_split, N, M, N, C_mem, elem_size,printflag)
    t_reduce = t_cpu_reduce(M_split, N, M, N, C_mem, elem_size, printflag)
    t_total_M_split = t_bus + t_reduce + max(t_dgemm_gpu(M_split, N, K), t_dgemm_cpu(M - M_split, N, K))
    #print(M_split, t_total_M_split)
    if printflag:
        print('Output M_split Split=%d Μ=%d Ν=%d Κ=%d A_mem=%d B_mem=%d C_mem=%d elem_sz=%d: t = %.5lf' % ( M_split, M, N, K, A_mem, B_mem, C_mem, elem_size,t_total_M_split))
        print('t_exec = %.5lf(t_dgemm_gpu = %.5lf, t_dgemm_cpu = %.5lf)\n' % (max(t_dgemm_gpu(M - M_split, N, K), t_dgemm_cpu(M - M_split, N, K)), t_dgemm_gpu(M_split, N, K), t_dgemm_cpu(M - M_split, N, K)))
    return t_total_M_split


def Impl_N_split(N_split, M, N, K, A_mem, B_mem, C_mem, elem_size,printflag):
    t_bus = t_bus_send(M, N_split, K, M, N, K, A_mem, B_mem, C_mem, elem_size,printflag) + t_bus_recv(M, N_split, M, N, C_mem, elem_size,printflag)
    t_reduce = t_cpu_reduce(M, N_split, M, N, C_mem, elem_size, printflag)
    t_total_N_split = t_bus + t_reduce +  max(t_dgemm_gpu(M, N_split, K), t_dgemm_cpu(M, N - N_split, K))
    #print(N_split, t_total_N_split)
    if printflag:
        print('Output N_split Split=%d Μ=%d Ν=%d Κ=%d A_mem=%d B_mem=%d C_mem=%d elem_sz=%d: t = %.5lf' % ( N_split, M, N, K, A_mem, B_mem, C_mem, elem_size,t_total_N_split))
        print('t_exec = %.5lf(t_dgemm_gpu = %.5lf, t_dgemm_cpu = %.5lf)\n' % (max(t_dgemm_gpu(M, N_split, K), t_dgemm_cpu(M, N - N_split, K)), t_dgemm_gpu(M, N_split, K), t_dgemm_cpu(M, N - N_split, K)))
    return t_total_N_split


def Impl_K_split(K_split, M, N, K, A_mem, B_mem, C_mem, elem_size,printflag):
    t_bus = t_bus_send(M, N, K_split, M, N, K, A_mem, B_mem, C_mem, elem_size,printflag) + t_bus_recv(M, N, M, N, C_mem, elem_size,printflag)
    t_reduce = t_cpu_reduce(M, N, M, N, C_mem, elem_size, printflag)
    t_total_K_split = t_bus + t_reduce + max(t_dgemm_gpu(M, N, K_split), t_dgemm_cpu(M, N, K - K_split))
    #print(K_split, t_total_K_split)
    if printflag:
        print('Output K_split Split=%d Μ=%d Ν=%d Κ=%d A_mem=%d B_mem=%d C_mem=%d elem_sz=%d: t = %.5lf' % ( K_split, M, N, K, A_mem, B_mem, C_mem, elem_size,t_total_K_split))
        print('t_exec = %.5lf(t_dgemm_gpu = %.5lf, t_dgemm_cpu = %.5lf)\n' % (max(t_dgemm_gpu(M, N, K_split), t_dgemm_cpu(M, N, K - K_split)), t_dgemm_gpu(M, N, K_split), t_dgemm_cpu(M, N, K - K_split)))
    return t_total_K_split


def int_Brute_solution(func, M, N, K, A_mem, B_mem, C_mem, dtype_size,bound_min, bound_max):
    answer = bound_min
    time = func(bound_min, M, N, K, A_mem, B_mem, C_mem, dtype_size, False)
    for split in range(bound_min+1,bound_max+1):
        temp = func(split, M, N, K, A_mem, B_mem, C_mem, dtype_size, False)
        if temp < time:
             time = temp
             answer = split
    func(answer, M, N, K, A_mem, B_mem, C_mem, dtype_size,True)

def list_Brute_solution(func, M, N, K, A_mem, B_mem, C_mem, dtype_size,bound_min, bound_max):
    answer = bound_min
    time = func(bound_min, M, N, K, A_mem, B_mem, C_mem, dtype_size, False)
    for split in range(bound_min+1,bound_max+1):
        temp = func(split, M, N, K, A_mem, B_mem, C_mem, dtype_size, False)
        if temp < time:
             time = temp
             answer = split
    func(answer, M, N, K, A_mem, B_mem, C_mem, dtype_size,True)


# Do something that makes more sense/is light
def recomended_split_dim(A_mem, B_mem, C_mem):
    N, M, K = 0, 0, 0
    if A_mem == 0:
        M +=1
        K -=1
    elif A_mem == 1:
        K +=1
        M -=1
    if B_mem == 0:
        K +=1
        N -=1
    elif B_mem == 1:
        N +=1
        K -=1
    if C_mem == 0:
        M +=1
        N -=1
    elif C_mem == 1:
        N +=1
        M -=1
    print( 'Split dim recommendations are M_split=%d, N_split=%d, K_Split=%d' % (M,N,K))



dtype_size = 8
N = M = K = 10000
A_mem = 0
B_mem = 1
C_mem = 0
x = np.ndarray(1)
x[0]=0

# Get past local minimums for now:
answer = scipop.minimize(Impl_M_split, x, args=(M, N, K, A_mem, B_mem, C_mem, 8,False),  bounds = [(0,M)])
for x0 in range(int(M/10)-1, M, int(M/10)):
    x[0] = x0
    temp =  scipop.minimize(Impl_M_split, x, args=(M, N, K, A_mem, B_mem, C_mem, 8,False),  bounds = [(0,M)])
    if temp.fun[0]< answer.fun[0]:
        answer = temp
        #print('Predicted M_split(%d,%d,%d): x0=%d -> Split = %d , t = %.5lf s (gemm_cpu=%.5lf s, gemm_gpu=%.5lf s)' % (M,N,K,x0, int(answer.x[0]) , answer.fun[0], t_dgemm_cpu(M -int(answer.x[0]),N,K), t_dgemm_gpu(int(answer.x[0]),N,K)))
#print('Predicted:')
#Impl_M_split(int(answer.x[0]), M, N, K, A_mem, B_mem, C_mem, 8,True)
print('Brute:')
int_Brute_solution(Impl_M_split, M, N, K, A_mem, B_mem, C_mem, dtype_size,0, M)
x[0]=0
answer1 = scipop.minimize(Impl_N_split, x, args=(M, N, K, A_mem, B_mem, C_mem, 8,False),  bounds = [(0,N)])
for x0 in range(int(N/10)-1, N, int(N/10)):
    x[0] = x0
    temp =  scipop.minimize(Impl_N_split, x, args=(M, N, K, A_mem, B_mem, C_mem, 8,False),  bounds = [(0,N)])
    if temp.fun[0]< answer1.fun[0]:
        answer1 = temp
        #print('Predicted N_split(%d,%d,%d): x0=%d -> Split = %d , t = %.5lf s (gemm_cpu=%.5lf s, gemm_gpu=%.5lf s)' % (M,N,K,x0, int(answer1.x[0]) , answer1.fun[0], t_dgemm_cpu(M,N - int(answer1.x[0]),K), t_dgemm_gpu(M,int(answer1.x[0]),K)))
#print('Predicted:')
#Impl_N_split(int(answer1.x[0]), M, N, K, A_mem, B_mem, C_mem, 8,True)
print('Brute:')
int_Brute_solution(Impl_N_split, M, N, K, A_mem, B_mem, C_mem, dtype_size,0, N)

x[0]=0
answer2 = scipop.minimize(Impl_K_split, x, args=(M, N, K, A_mem, B_mem, C_mem, 8,False),  bounds = [(0,K)])
for x0 in range(int(K/10)-1, K, int(K/10)):
    x[0] = x0
    temp =  scipop.minimize(Impl_K_split, x, args=(M, N, K, A_mem, B_mem, C_mem, 8,False),  bounds = [(0,K)])
    if temp.fun[0]< answer2.fun[0]:
        answer2 = temp
        #print('Predicted K_split(%d,%d,%d): x0=%d -> Split = %d , t = %.5lf s (gemm_cpu=%.5lf s, gemm_gpu=%.5lf s)' % (M,N,K,x0, int(answer2.x[0]) , answer2.fun[0], t_dgemm_cpu(M,N, K - int(answer2.x[0])), t_dgemm_gpu(M,N,int(answer2.x[0]))))
#print('Predicted:')
#Impl_K_split(int(answer2.x[0]), M, N, K, A_mem, B_mem, C_mem, 8,True)
print('Brute:')
int_Brute_solution(Impl_K_split, M, N, K, A_mem, B_mem, C_mem, dtype_size,0, K)

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

recomended_split_dim(A_mem, B_mem, C_mem)
M = 9500
Impl_split([(M/4,N/4,K),(M/4,N*3/4,K),(M/4,N,K),(M/2,N/2,K),(M/2,N/2,K)], M, N, K, A_mem, B_mem, C_mem, dtype_size, True)


