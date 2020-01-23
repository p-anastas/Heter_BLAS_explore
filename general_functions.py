import subprocess
import sys
import numpy as np
import scipy.optimize as scipop
import scipy as sc
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.interpolate import interp1d

machine = 'dungani'

def initialize():
    resDir = 'Results_' + machine + '/'
    bwfile = resDir + 'bandwidth_log_' + machine + '.md_sorted'
    transfile = resDir + 'transpose_log_' + machine + '.md'
    return (resDir,bwfile, transfile)

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
    if len(x) != 0:
        model = LinearRegression(n_jobs=-1).fit(x,y)
        #r_sq = model.score(x, y)
        #print('coefficient of determination:', r_sq)
        if model.intercept_ < 0:
            print('Negative intercept...ignoring:', model.intercept_)
            #model.intercept_ = 0
        if model.coef_ < 0:
            print('Negative coef...danger:', model.coef_)
        error = sum(predict_error(bounded_Y, model.predict(x)))/len(bounded_Y)
        #print (error)
    else:
        sys.exit('No benchmarks found for bound [%d,%d]' % (lower_bound,upper_bound))
    return (model, error)

def Bound_LinearRegression_2d(X_list,Y_list,lower_bound, upper_bound):
    bounded_Y = []
    bounded_X = []
    for qX, qY in zip(X_list, Y_list):
        if (qX[0]*qX[1] < upper_bound and qX[0]*qX[1] >= lower_bound):
            bounded_X.append(qX)
            bounded_Y.append(qY)
    x = np.array(bounded_X).reshape(-1, 2)
    y = np.array(bounded_Y)
    if len(x) != 0:
        model = LinearRegression(n_jobs=-1).fit(x,y)
        #r_sq = model.score(x, y)
        #print('coefficient of determination:', r_sq)
        if model.intercept_ < 0:
            print('Negative intercept...ignoring:', model.intercept_)
            #model.intercept_ = 0
        if model.coef_.any() < 0:
            print('Negative coef...danger:', model.coef_)
        error = sum(predict_error(bounded_Y, model.predict(x)))/len(bounded_Y)
        #print (error)
    else:
        sys.exit('No benchmarks found for bound [%d,%d]' % (lower_bound,upper_bound))
    return (model, error)

def LinearRegression_1d(X_list,Y_list,bounds):
    linear_subparts = []
    prev_bound = 0
    err = 0
    for upper_bound in bounds:
        model,pred_err = Bound_LinearRegression_1d(X_list,Y_list,prev_bound, upper_bound)
        linear_subparts.append(model.predict) #.intercept_, model.coef_, lesser_bound, upper_bound)
        prev_bound = upper_bound
        err += pred_err
    err = err / len(bounds)
    #print(err)
    return linear_subparts

def LinearRegression_2d(X_list,Y_list,bounds):
    linear_subparts = []
    prev_bound = 0
    err = 0
    for upper_bound in bounds:
        model,pred_err = Bound_LinearRegression_2d(X_list,Y_list,prev_bound, upper_bound)
        linear_subparts.append(model.predict) #.intercept_, model.coef_, lesser_bound, upper_bound)
        prev_bound = upper_bound
        err += pred_err
    err = err / len(bounds)
    #print(err)
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

