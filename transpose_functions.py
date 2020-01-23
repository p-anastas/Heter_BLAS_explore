import subprocess
import sys
import numpy as np
import scipy.optimize as scipop
import scipy as sc
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.interpolate import interp1d
from general_functions import initialize, LinearRegression_1d, LinearRegression_2d

# For now use artificial bound to check
flop_bounds = [1e5,1e6,1e7,8e9]

resDir, _, trans_file, _, _, _ = initialize()

with open(trans_file, "r") as file0:
    trans_db = file0.readlines()

def read_trans_file(trans_file):
    if  len(trans_db) == 0:
        sys.exit('Error: Bandwidth benchmark not found')
    time = [0]
    elems = [[0,0]]
    for line in trans_db:
        temp = line.split(',')
        if temp[-1]!='synth\n':
             time.append(float(temp[2]))
             elems.append([int(temp[0]),int(temp[1])] )
    xysorted = sorted(zip(elems,time), key= lambda x: x[0])
    return  ([x for x,_ in xysorted], [y for _,y in xysorted])

def interpolate1d_transpose():
    (elems, time) = read_trans_file(trans_file)
    flops = list(map(lambda x: x[0]*x[1], elems))
    return interp1d(flops, time, kind='linear')

def linearize2d_transpose():
    (elems, time) = read_trans_file(trans_file)
    return LinearRegression_2d(elems,time,flop_bounds)

def linearize_transpose():
    (elems, time) = read_trans_file(trans_file)
    flops = list(map(lambda x: x[0]*x[1], elems))
    return LinearRegression_1d(flops,time,flop_bounds)

def linearized2d_dtranspose(X,Y):
    #Hack for our own transpose
    if X > Y:
        swap = X
        X = Y
        Y = swap
    f_transpose_bound_regresion_2d = linearize2d_transpose()
    ctr = 0
    for bound in flop_bounds:
        if X*Y < bound:
            return f_transpose_bound_regresion_2d[ctr](np.array([X,Y]).reshape(-1, 2))
        ctr +=1
    return f_transpose_bound_regresion_2d[ctr-1](np.array([X,Y]).reshape(-1, 2))

def linearized1d_dtranspose(X):
    f_transpose_bound_regresion = linearize_transpose()
    ctr = 0
    for bound in flop_bounds:
        if X < bound:
            return f_transpose_bound_regresion[ctr](np.array(X).reshape(-1, 1))
        ctr +=1
    return f_transpose_bound_regresion[ctr-1](np.array(X).reshape(-1, 1))

def interpolated1d_dtranspose(X):
    f_transpose_interpolated = interpolate1d_transpose()
    return f_transpose_interpolated(X)



# Wrapper for ease of use
def t_dtranspose(X,Y):
    if (X <1  or Y < 1):
        return 0
    return interpolated1d_dtranspose(X*Y)/10
    #return linearized2d_dtranspose(X, Y)
    #return linearized1d_dtranspose(X*Y)

