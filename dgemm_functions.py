import subprocess
import sys
import numpy as np
import scipy.optimize as scipop
import scipy as sc
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.interpolate import interp1d,LinearNDInterpolator
from general_functions import initialize,LinearRegression_1d,LinearRegression_3d

# For now use artificial bound to check
flop_bounds = [1e6,1e7,1e8,8e9]

resDir, _, _, cpugemmfile, gpugemmfile, _ = initialize()

with open(cpugemmfile, "r") as file0:
    cpu_gemm_db = file0.readlines()

with open(gpugemmfile, "r") as file0:
    gpu_gemm_db = file0.readlines()

def read_cpu_file(cpugemmfile):
    if  len(cpu_gemm_db) == 0:
        sys.exit('Error: t_dgemm_cpu benchmark not found')
    time = [0]
    elems = [[0,0,0]]
    for line in cpu_gemm_db:
        temp = line.split(',')
        if temp[-1]!='synth\n':
            time.append(float(temp[19]))
            elems.append([int(temp[0]),int(temp[1]), int(temp[2])] )
    xysorted = sorted(zip(elems,time), key= lambda x: x[0])
    return  ([x for x,_ in xysorted], [y for _,y in xysorted])

def read_gpu_file(gpugemmfile):
    if  len(gpu_gemm_db) == 0:
        sys.exit('Error: t_dgemm_cpu benchmark not found')
    time = [0]
    elems = [[0,0,0]]
    for line in gpu_gemm_db:
        temp = line.split(',')
        if temp[-1]!='synth\n':
            time.append(float(temp[18]))
            elems.append([int(temp[0]),int(temp[1]), int(temp[2])] )
    xysorted = sorted(zip(elems,time), key= lambda x: x[0])
    return  ([x for x,_ in xysorted], [y for _,y in xysorted])

def interpolate3d_cpu_gemm():
    (elems, time) = read_cpu_file(cpugemmfile)
    return LinearNDInterpolator(elems, time)

def interpolate3d_cpu_dgemm(M,N,K):
    f_cpu_gemm_interpolated = interpolate3d_cpu_gemm()
    return f_cpu_gemm_interpolated([M,N,K])

def linearize1d_cpu_gemm():
    (elems, time) = read_cpu_file(cpugemmfile)
    flops = list(map(lambda x: x[0]*x[1]*x[2], elems))
    return LinearRegression_1d(flops,time,flop_bounds)

def linearize1d_cpu_dgemm(flops):
    f_cpu_gemm_bound_regression1d = linearize1d_cpu_gemm()
    ctr = 0
    for bound in flop_bounds:
        if flops < bound:
            return f_cpu_gemm_bound_regression1d[ctr](np.array(flops).reshape(-1, 1))
        ctr +=1
    return f_cpu_gemm_bound_regression1d[ctr-1](np.array(flops).reshape(-1, 1))

def linearize3d_cpu_gemm():
    (elems, time) = read_cpu_file(cpugemmfile)
    return LinearRegression_3d(elems,time,flop_bounds)

def linearize3d_cpu_dgemm(M,N,K):
    f_cpu_gemm_bound_regression3d = linearize3d_cpu_gemm()
    ctr = 0
    for bound in flop_bounds:
        if M*N*K < bound:
            return f_cpu_gemm_bound_regression3d[ctr](np.array([M,N,K]).reshape(-1, 3))
        ctr +=1
    return f_cpu_gemm_bound_regression3d[ctr-1](np.array([M,N,K]).reshape(-1, 3))

def interpolate3d_gpu_gemm():
    (elems, time) = read_gpu_file(gpugemmfile)
    return LinearNDInterpolator(elems, time)

def interpolate3d_gpu_dgemm(M,N,K):
    f_gpu_gemm_interpolated = interpolate3d_gpu_gemm()
    return f_gpu_gemm_interpolated([M,N,K])

def linearize1d_gpu_gemm():
    (elems, time) = read_gpu_file(gpugemmfile)
    flops = list(map(lambda x: x[0]*x[1]*x[2], elems))
    return LinearRegression_1d(flops,time,flop_bounds)

def linearize1d_gpu_dgemm(flops):
    f_gpu_gemm_bound_regression = linearize1d_gpu_gemm()
    ctr = 0
    for bound in flop_bounds:
        if flops < bound:
            return f_gpu_gemm_bound_regression[ctr](np.array(flops).reshape(-1, 1))
        ctr +=1
    return f_gpu_gemm_bound_regression[ctr-1](np.array(flops).reshape(-1, 1))

def linearize3d_gpu_gemm():
    (elems, time) = read_cpu_file(gpugemmfile)
    return LinearRegression_3d(elems,time,flop_bounds)

def linearize3d_gpu_dgemm(M,N,K):
    f_gpu_gemm_bound_regression3d = linearize3d_gpu_gemm()
    ctr = 0
    for bound in flop_bounds:
        if M*N*K < bound:
            return f_gpu_gemm_bound_regression3d[ctr](np.array([M,N,K]).reshape(-1, 3))
        ctr +=1
    return f_gpu_gemm_bound_regression3d[ctr-1](np.array([M,N,K]).reshape(-1, 3))

def t_dgemm_cpu(M, N, K):
    if (M < 1 or N <1 or K <1):
        return 0
    else:
        #return linearize1d_cpu_dgemm(M*N*K)
        #return linearize3d_cpu_dgemm(M,N,K)
        return interpolate3d_cpu_dgemm(M,N,K)

def t_dgemm_gpu(M, N, K):
    if (M < 1 or N <1 or K <1):
        return 0
    else:
        #return linearize1d_gpu_dgemm(M*N*K)
        #return linearize3d_gpu_dgemm(M,N,K)
        return interpolate3d_gpu_dgemm(M,N,K)

