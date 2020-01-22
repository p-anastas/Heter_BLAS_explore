import subprocess
import sys
import numpy as np
import scipy.optimize as scipop
import scipy as sc
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.interpolate import interp1d

machine = 'dungani'
resDir = 'Results_' + machine + '/'

with open(resDir + 'bandwidth_log_' + machine + '.md_sorted', "r") as file0:
    bw_db = file0.readlines()

# For now use artificial bound to check
bounds = [1e5,1e6,8e9]

def predict_error(X,Y):
    answer = []
    for val,prediction in zip(X,Y):
        if val == 0 and prediction ==0:
             answer.append(0)
        elif val == 0:
             answer.append(abs(val-prediction)/prediction)
        else:
             answer.append(abs(val-prediction)/val)
    return answer

def Bound_LinearRegression_1d(X_list,Y_list,lower_bound, upper_bound):
    bounded_Y = []
    bounded_X = []
    for qX, qY in zip(X_list, Y_list):
        if (qX < upper_bound and qX >= lower_bound):
            bounded_X.append(qX)
            bounded_Y.append(qY)
    x = np.array(bounded_X).reshape(-1, 1)
    y = np.array(bounded_Y)
    model = LinearRegression(n_jobs=-1).fit(x,y)
    r_sq = model.score(x, y)
    print('coefficient of determination:', r_sq)
    if model.intercept_ < 0:
        print('Negative intercept...ignoring:', model.intercept_)
        #model.intercept_ = 0
    if model.coef_ < 0:
        print('Negative coef...danger:', model.coef_)
    error = sum(predict_error(bounded_Y, model.predict(x)))/len(bounded_Y)
    print (error)
    return (model, error)

def LinearRegression_1d(X_list,Y_list):
    linear_subparts = []
    prev_bound = 0
    err = 0
    for upper_bound in bounds:
        model,pred_err = Bound_LinearRegression_1d(X_list,Y_list,prev_bound, upper_bound)
        linear_subparts.append(model.predict) #.intercept_, model.coef_, lesser_bound, upper_bound)
        prev_bound = upper_bound
        err += pred_err
    err = err / len(bounds)
    print(err)
    return linear_subparts

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

def linearize_cpu_to_gpu():
    if  len(bw_db) == 0:
        sys.exit('Error: Bandwidth benchmark not found')
    time = []
    bytes = []
    for line in bw_db:
        temp = line.split(',')
        if int(temp[1]) == -1 and int(temp[2]) == 0:
             time.append(float(temp[3]))
             bytes.append(int(temp[0]))
    return LinearRegression_1d(bytes,time)

def linearize_gpu_to_cpu():
    if  len(bw_db) == 0:
        sys.exit('Error: Bandwidth benchmark not found')
    time = []
    bytes = []
    for line in bw_db:
        temp = line.split(',')
        if int(temp[1]) == 0 and int(temp[2]) == -1:
             time.append(float(temp[3]))
             bytes.append(int(temp[0]))
    return LinearRegression_1d(bytes,time)

def interpolated_transfer_to_gpu():
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
    f = interp1d(bytes, time, kind='linear')
    return f

def interpolated_transfer_from_gpu():
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
    f = interp1d(bytes, time, kind='linear')
    return f

def linearized_memcpy(bytes, src, dest):
    if (bytes < 1):
        return 0
    # For now only for dev0 <-> host
    elif (src == -1 and dest == 0):
        f_send_bound_regresion = linearize_cpu_to_gpu()
        ctr = 0
        for bound in bounds:
            if bytes < bound:
                return f_send_bound_regresion[ctr](np.array(bytes).reshape(-1, 1))
            ctr +=1
        return f_send_bound_regresion[-1](np.array(bytes).reshape(-1, 1))
    elif (src == 0 and dest == -1):
        f_recv_bound_regresion =  linearize_gpu_to_cpu()
        ctr = 0
        for bound in bounds:
            if bytes < bound:

                return f_recv_bound_regresion[ctr](np.array(bytes).reshape(-1, 1))
            ctr +=1
        return f_recv_bound_regresion[-1](np.array(bytes).reshape(-1, 1))

def interpolated_memcpy(bytes, src, dest):
    if (bytes < 1):
        return 0
    # For now only for dev0 <-> host
    elif (src == -1 and dest == 0):
        f_send_inter_linear = interpolated_transfer_to_gpu()
        return f_send_inter_linear(bytes)
    elif (src == 0 and dest == -1):
        f_recv_inter_linear = interpolated_transfer_from_gpu()
        return f_recv_inter_linear(bytes)

def t_copy_vec_1d(N, src, dest, elem_size):
    if N < 1:
        return 0
    return linearized_memcpy(N * elem_size, src, dest)


def t_copy_vec_2d(dim1, dim2, ldim, src, dest, elem_size):
    if ldim == dim2:
        return linearized_memcpy(dim1 * dim2 * elem_size, src, dest)
    elif ldim > dim2:
        return dim1 * linearized_memcpy(dim2 * elem_size, src, dest)
    else:
        # This is wrong but functions hate nans
        return linearized_memcpy(dim1 * dim2 * elem_size, src, dest)
        #return float("inf")
        sys.exit('Error: dim2(%d) > ldim(%d)' % (dim2, ldim))


