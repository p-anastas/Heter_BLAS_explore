import subprocess
import sys
import numpy as np
import scipy.optimize as scipop
import scipy as sc
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.interpolate import interp1d
from general_functions import initialize,LinearRegression_1d

# For now use artificial bound to check
bounds = [1e5,1e6,1e7,8e9]

resDir, bw_file, _, _, _, _ = initialize()

with open(bw_file, "r") as file0:
    bw_db = file0.readlines()

def read_bw_file(bw_file,src,dest):
    if  len(bw_db) == 0:
        sys.exit('Error: Bandwidth benchmark not found')
    time = [0]
    bytes = [0]
    for line in bw_db:
        temp = line.split(',')
        if int(temp[1]) == src and int(temp[2]) == dest and temp[-1]!='synth\n':
             if temp[-1]!='bound_0\n' and temp[-1]!='bound_1\n' and temp[-1]!='bound_2\n':
                 time.append(float(temp[3]))
                 bytes.append(int(temp[0]))
             else: 
                 if temp[-1]=='bound_0\n':
                     bounds[0] = int(temp[0])
                 elif temp[-1]=='bound_1\n':
                     bounds[1] = int(temp[0])
                 elif temp[-1]=='bound_2\n':
                     bounds[2] = int(temp[0])
                 #print("Bound found src=%d, dest=%d Bound=%d" % (src,dest,int(temp[0])))
    xysorted = sorted(zip(bytes,time), key= lambda x: x[0])
    return  ([x for x,_ in xysorted], [y for _,y in xysorted])

def linearize_cpu_to_gpu():
    (bytes,time) = read_bw_file(bw_file,-1,0)
    return LinearRegression_1d(bytes,time,bounds)

def linearize_gpu_to_cpu():
    (bytes,time) = read_bw_file(bw_file,0,-1)
    return LinearRegression_1d(bytes,time,bounds)

def interpolated_transfer_to_gpu():
    (bytes,time) = read_bw_file(bw_file,-1,0)
    f = interp1d(bytes, time, kind='linear')
    return f

def interpolated_transfer_from_gpu():
    (bytes,time) = read_bw_file(bw_file,0,-1)
    f = interp1d(bytes, time, kind='linear')
    return f

f_send_bound_regresion = linearize_cpu_to_gpu()
f_recv_bound_regresion =  linearize_gpu_to_cpu()
f_send_inter_linear = interpolated_transfer_to_gpu()
f_recv_inter_linear = interpolated_transfer_from_gpu()


def linearized_memcpy(bytes, src, dest):
    # For now only for dev0 <-> host
    if (src == -1 and dest == 0):
        #f_send_bound_regresion = linearize_cpu_to_gpu()
        ctr = 0
        for bound in bounds:
            if bytes < bound:
                return f_send_bound_regresion[ctr](np.array(bytes).reshape(-1, 1))
            ctr +=1
        return f_send_bound_regresion[ctr -1](np.array(bytes).reshape(-1, 1))
    elif (src == 0 and dest == -1):
        #f_recv_bound_regresion =  linearize_gpu_to_cpu()
        ctr = 0
        for bound in bounds:
            if bytes < bound:

                return f_recv_bound_regresion[ctr](np.array(bytes).reshape(-1, 1))
            ctr +=1
        return f_recv_bound_regresion[ctr -1](np.array(bytes).reshape(-1, 1))

def interpolated_memcpy(bytes, src, dest):
    # For now only for dev0 <-> host
    if (src == -1 and dest == 0):
        #f_send_inter_linear = interpolated_transfer_to_gpu()
        return f_send_inter_linear(bytes)
    elif (src == 0 and dest == -1):
        #f_recv_inter_linear = interpolated_transfer_from_gpu()
        return f_recv_inter_linear(bytes)

# Wrapper for easy use
def t_memcpy(bytes, src, dest):
    if (bytes < 1):
        return 0
    #return interpolated_memcpy(bytes, src, dest)
    return linearized_memcpy(bytes, src, dest)

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
        # This is wrong but functions hate nans
        return t_memcpy(dim1 * dim2 * elem_size, src, dest)
        #return float("inf")
        sys.exit('Error: dim2(%d) > ldim(%d)' % (dim2, ldim))


