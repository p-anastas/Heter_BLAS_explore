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
        pred,pred_err = Bound_LinearRegression_1d(X_list,Y_list,prev_bound, upper_bound)
        linear_subparts.append(pred) #.intercept_, model.coef_, lesser_bound, upper_bound)
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
    x = np.array(bytes).reshape(-1, 1)
    y = np.array(time)
    model = LinearRegression(n_jobs=-1).fit(x,y)
    r_sq = model.score(x, y)
    print('coefficient of determination:', r_sq)
    print('intercept:', model.intercept_)
    print('slope:', model.coef_)

    f = interp1d(bytes, time, kind='linear')
    f1 = interp1d(bytes, time, kind='zero')
    f2 = interp1d(bytes, time, kind='slinear')
    f3 = interp1d(bytes, time, kind='quadratic')
    f4 = interp1d(bytes, time, kind='cubic')
    return model.predict, f, f1, f2, f3, f4

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
    x = np.array(bytes).reshape(-1, 1)
    y = np.array(time)
    model = LinearRegression(n_jobs=-1).fit(x,y)
    r_sq = model.score(x, y)
    print('coefficient of determination:', r_sq)
    print('intercept:', model.intercept_)
    print('slope:', model.coef_)
    from scipy.interpolate import interp1d
    f = interp1d(bytes, time, kind='linear')
    f1 = interp1d(bytes, time, kind='zero')
    f2 = interp1d(bytes, time, kind='slinear')
    f3 = interp1d(bytes, time, kind='quadratic')
    f4 = interp1d(bytes, time, kind='cubic')
    return model.predict, f, f1, f2, f3, f4

f_send_bound_regresion = linearize_cpu_to_gpu()
f_send_reg, f_send_inter_linear, f_send_inter_zero, f_send_inter_slinear, f_send_inter_quadratic, f_send_inter_cubic = t_transfer_to_gpu()
f_recv_bound_regresion =  linearize_gpu_to_cpu()
f_recv_reg, f_recv_inter_linear, f_recv_inter_zero, f_recv_inter_slinear, f_recv_inter_quadratic, f_recv_inter_cubic = t_transfer_from_gpu()
def t_memcpy(bytes, src, dest):
    if(bytes < 0):
        return 0 #float("inf")
    elif (bytes < 1):
        return 0
    # For now only for dev0 <-> host
    elif (src == -1 and dest == 0):
        ctr = 0
        for bound in bounds:
            if bytes < bound:
                return f_send_bound_regresion[ctr](np.array(bytes).reshape(-1, 1))
            ctr +=1
        return f_send_bound_regresion[-1](np.array(bytes).reshape(-1, 1))
    elif (src == 0 and dest == -1):
        ctr = 0
        for bound in bounds:
            if bytes < bound:
                return f_recv_bound_regresion[ctr](np.array(bytes).reshape(-1, 1))
            ctr +=1
        return f_recv_bound_regresion[-1](np.array(bytes).reshape(-1, 1))

def plot_transfers(plot_up_bound):
    plot_down_bound = 0
    xnew = np.linspace(plot_down_bound, plot_up_bound, num=200, endpoint=True)
    time = [[0],[0]]
    bytes = [[0],[0]]
    for line in bw_db:
        temp = line.split(',')
        if int(temp[1]) == -1 and int(temp[2]) == 0 and int(temp[0])<=plot_up_bound: #temp[-1] != 'synth\n':
           if bytes[0][-1] != int(temp[0]):
                 time[0].append(float(temp[3]))
                 bytes[0].append(int(temp[0]))
        elif int(temp[1]) == 0 and int(temp[2]) == -1 and int(temp[0])<=plot_up_bound:
           if bytes[1][-1] != int(temp[0]):
                 time[1].append(float(temp[3]))
                 bytes[1].append(int(temp[0]))
    import matplotlib.pyplot as plt
    plt.plot(bytes[0], time[0], 'o', bytes[0],f_send_reg(np.array(bytes[0]).reshape(-1, 1)), '-', bytes[0], f_send_inter_linear(bytes[0]), '-', bytes[0], list(map(lambda x: t_memcpy(x,-1,0), bytes[0])), '-')
    # plt.xscale('log')
    plt.legend(['Host->Device(Samples)', 'Host->Device(Single Linear regression)', 'Host->Device(interpolate linear)', 'Host->Device(Mixed Linear regression(4))'], loc='best')
    plt.ylabel('Time(s)')
    plt.xlabel('Bytes')
    plt.title('PCI-e Time (0-' + str(plot_up_bound) + ' bytes)')
    plt.savefig(resDir + 'Time_'+ machine + '_' + str(plot_up_bound) + '_bytes_host_to_device.eps', format='eps')
    plt.close()
    plt.plot(bytes[1], time[1], 'o', bytes[1],f_recv_reg(np.array(bytes[1]).reshape(-1, 1)), '-', bytes[1], f_recv_inter_linear(bytes[1]), '-', bytes[1], list(map(lambda x: t_memcpy(x,0,-1), bytes[1])), '-')
    # plt.xscale('log')
    plt.legend(['Device->Host(Samples)', 'Device->Host(Single Linear regression)', 'Device->Host(interpolate linear)', 'Device->Host(Mixed Linear regression(4))'], loc='best')
    plt.ylabel('Time(s)')
    plt.xlabel('Bytes')
    plt.title('PCI-e Time (0-' + str(plot_up_bound) + ' bytes)')
    plt.savefig(resDir + 'Time_'+ machine + '_' + str(plot_up_bound) + '_bytes_device_to_host.eps', format='eps')
    plt.close()

    plt.plot( bytes[0],predict_error(time[0], f_send_reg(np.array(bytes[0]).reshape(-1, 1))), '-', bytes[0], predict_error(time[0], f_send_inter_linear(np.array(bytes[0]).reshape(-1, 1))), '-', bytes[0], predict_error(time[0], list(map(lambda x: t_memcpy(x,-1,0), bytes[0]))) , '-')
    # plt.xscale('log')
    plt.legend([ 'Host->Device(Single Linear regression)', 'Host->Device(interpolate linear)', 'Host->Device(Mixed Linear regression(4))'], loc='best')
    plt.ylabel('Error(0-1)')
    plt.xlabel('Bytes')
    plt.title('Prediction Error (0-' + str(plot_up_bound) + ' bytes)')
    plt.savefig(resDir + 'Error_'+ machine + '_' + str(plot_up_bound) + '_bytes_host_to_device.eps', format='eps')
    plt.close()
    plt.plot( bytes[1],predict_error(time[1], f_recv_reg(np.array(bytes[1]).reshape(-1, 1))), '-', bytes[1], predict_error(time[1], f_recv_inter_linear(np.array(bytes[1]).reshape(-1, 1))), '-', bytes[1], predict_error(time[1], list(map(lambda x: t_memcpy(x,0,-1), bytes[1]))) , '-')
    # plt.xscale('log')
    plt.legend(['Device->Host(Single Linear regression)', 'Device->Host(interpolate linear)', 'Device->Host(Mixed Linear regression(4))'], loc='best')
    plt.ylabel('Error(0-1)')
    plt.xlabel('Bytes')
    plt.title('Prediction Error (0-' + str(plot_up_bound) + ' bytes)')
    plt.savefig(resDir + 'Error_'+ machine + '_' + str(plot_up_bound) + '_bytes_device_to_host.eps', format='eps')
    plt.close()


plot_transfers(1e4)
plot_transfers(1e5)
plot_transfers(1e6)

