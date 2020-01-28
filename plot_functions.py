import subprocess
import sys
import numpy as np
import scipy.optimize as scipop
import scipy as sc
import matplotlib.pyplot as plt
from bandwidth_functions import *
from general_functions import *
from transpose_functions import *
from dgemm_functions import *

machine = 'dungani'
resDir = 'Results_' + machine + '/'

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
    xnew = np.linspace(plot_down_bound, plot_up_bound, num=200, endpoint=True)
    send_bytes,send_time = [0], [0]
    recv_bytes,recv_time = [0], [0]
    for line in bw_db:
        temp = line.split(',')
        if int(temp[1]) == -1 and int(temp[2]) == 0 and int(temp[0])<=plot_up_bound: #temp[-1] != 'synth\n':
           if send_bytes[-1] != int(temp[0]):
                 send_time.append(float(temp[3]))
                 send_bytes.append(int(temp[0]))
        elif int(temp[1]) == 0 and int(temp[2]) == -1 and int(temp[0])<=plot_up_bound:
           if recv_bytes[-1] != int(temp[0]):
                 recv_time.append(float(temp[3]))
                 recv_bytes.append(int(temp[0]))
    plt.plot(send_bytes, GigaVal_per_s_l(send_bytes,send_time), 'o', xnew, GigaVal_per_s_l(xnew, list(map(lambda x: linearized_memcpy(x,-1,0), xnew))), '-', xnew, GigaVal_per_s_l(xnew, list(map(lambda x: interpolated_memcpy(x,-1,0), xnew))), '-')
    # plt.xscale('log')
    plt.legend(['Host->Dev(Samples)', 'Host->Dev(Hybrid LR)', 'Host->Dev(Interpolated)'], loc='best')
    plt.ylabel('Gb/s')
    plt.xlabel('Bytes')
    plt.title('PCI-e Bandwidth (0-' + str(plot_up_bound) + ' bytes)')
    plt.savefig(resDir+'Host_to_Dev_Bw_'+ machine +'_' + str(plot_up_bound) + '_bytes.eps', format='eps')
    plt.close()
    plt.plot( recv_bytes, GigaVal_per_s_l(recv_bytes,recv_time), 'o', xnew, GigaVal_per_s_l(xnew,list(map(lambda x: linearized_memcpy(x,0,-1), xnew))), '-', xnew, GigaVal_per_s_l(xnew,list(map(lambda x: interpolated_memcpy(x,0,-1), xnew))), '-')
    # plt.xscale('log')
    plt.legend(['Dev->Host(Samples)','Dev->Host(Hybrid LR)', 'Dev->Host(Interpolated)'], loc='best')
    plt.ylabel('Gb/s')
    plt.xlabel('Bytes')
    plt.title('PCI-e Bandwidth (0-' + str(plot_up_bound) + ' bytes)')
    plt.savefig(resDir+'Dev_to_Host_Bw_'+ machine +'_' + str(plot_up_bound) + '_bytes.eps', format='eps')
    plt.close()

    plt.plot( send_bytes, predict_error(send_time, list(map(lambda x: linearized_memcpy(x,-1,0), send_bytes))), '-', send_bytes, predict_error(send_time, list(map(lambda x: interpolated_memcpy(x,-1,0), send_bytes))), '-')
    # plt.xscale('log')
    plt.legend([ 'Host->Dev(Hybrid LR)', 'Host->Dev(Interpolated)'], loc='best')
    plt.ylabel('Error(0-1)')
    plt.xlabel('Bytes')
    plt.title('Prediction Error (0-' + str(plot_up_bound) + ' bytes)')
    plt.savefig(resDir + 'Host_to_Dev_Error_'+ machine + '_' + str(plot_up_bound) + '_bytes.eps', format='eps')
    plt.close()
    plt.plot( recv_bytes,predict_error(recv_time, list(map(lambda x: linearized_memcpy(x,0,-1), recv_bytes))), '-', recv_bytes, predict_error(recv_time, list(map(lambda x: interpolated_memcpy(x,0,-1), recv_bytes))), '-')
    # plt.xscale('log')
    plt.legend(['Dev->Host(Hybrid LR)', 'Dev->Host(Interpolated)'], loc='best')
    plt.ylabel('Relative Error')
    plt.xlabel('Bytes')
    plt.title('Prediction Error (0-' + str(plot_up_bound) + ' bytes)')
    plt.savefig(resDir + 'Dev_to_Host_Error_'+ machine + '_' + str(plot_up_bound) + '_bytes.eps', format='eps')
    plt.close()

def plot_transpose(plot_up_bound):
    plot_down_bound = 0
    time = [0]
    flops= [0]
    elems_X= [0]
    elems_Y= [0]
    for line in trans_db:
        temp = line.split(',')
        if int(temp[0])*int(temp[1])< plot_up_bound:
           if not (int(temp[0])*int(temp[1]) in flops):
               time.append(float(temp[2]))
               flops.append(int(temp[0])*int(temp[1]))
               elems_X.append(int(temp[0]))
               elems_Y.append(int(temp[1]))
    flops, time_1d = zip(*sorted(zip(flops, time), key = lambda x: x[0]))
    elems_X, elems_Y, time_2d = zip(*sorted(zip(elems_X, elems_Y, time), key = lambda x: x[0]))
    print(elems_X, elems_Y, time_2d)
    #print(flops, time_1d)
    plt.plot(flops, time_1d, 'o', flops, list(map(lambda x: linearized1d_dtranspose(x),flops)) , '-' ,list(map(lambda x,y: x*y,elems_X, elems_Y)) , list(map(lambda x,y: linearized2d_dtranspose(x,y),elems_X, elems_Y)) , '+' , flops, list(map(lambda x: interpolated1d_dtranspose(x),flops)) , '-')
    plt.legend(['Transpose Host (Samples)', 'Transpose Host (LR 1D)' , 'Transpose Host (LR 2D)' , 'Transpose Host (Interoplate 1d)'  ], loc='best')
    plt.ylabel('Time(s)')
    plt.xlabel('N*M')
    plt.title('Transpose (0-' + str(plot_up_bound) + '/'+ str(plot_up_bound) + ' )')
    plt.savefig(resDir + 'Transpose_'+machine+'_' + str(plot_up_bound) + '.eps', format='eps')
    plt.close()

    plt.plot( flops, predict_error(time_1d, list(map(lambda x: linearized1d_dtranspose(x),flops))) , '-' ,list(map(lambda x,y: x*y,elems_X, elems_Y)) , predict_error(time_2d, list(map(lambda x,y: linearized2d_dtranspose(x,y),elems_X, elems_Y))) , '+' , flops, predict_error(time_1d, list(map(lambda x: interpolated1d_dtranspose(x),flops))) , '-')
    plt.legend([ 'Transpose Host (LR 1D)' , 'Transpose Host (LR 2D)' , 'Transpose Host (Interoplate 1d)'  ], loc='best')
    plt.ylabel('Error')
    plt.xlabel('N*M')
    plt.title('Error_Transpose (0-' + str(plot_up_bound) + '/'+ str(plot_up_bound) + ' )')
    plt.savefig(resDir + 'Error_Transpose_'+machine+'_' + str(plot_up_bound) + '.eps', format='eps')
    plt.close()

def plot_gemm(loc, plot_up_bound):
    if loc == -1:
        file_db = cpu_gemm_db
        timectr = 19
        locname = 'CPU'
        inter_3d_fun = interpolate3d_cpu_dgemm
        lr_3d_fun = linearize3d_cpu_dgemm
        lr_1d_fun = linearize1d_cpu_dgemm
    elif loc == 0:
        file_db = gpu_gemm_db
        timectr = 18
        locname = 'GPU'
        inter_3d_fun = interpolate3d_gpu_dgemm
        lr_3d_fun = linearize3d_gpu_dgemm
        lr_1d_fun = linearize1d_gpu_dgemm
    plot_down_bound = 0
    time = [0]
    flops= [0]
    elems_X= [0]
    elems_Y= [0]
    elems_Z= [0]
    for line in file_db:
        temp = line.split(',')
        if int(temp[0])*int(temp[1])*int(temp[2])< plot_up_bound:
           if not (int(temp[0])*int(temp[1])*int(temp[2]) in flops):
               time.append(float(temp[timectr]))
               flops.append(int(temp[0])*int(temp[1])*int(temp[2]))
               elems_X.append(int(temp[0]))
               elems_Y.append(int(temp[1]))
               elems_Z.append(int(temp[2]))
    flops, time_1d = zip(*sorted(zip(flops, time), key = lambda x: x[0]))
    elems_X, elems_Y, elems_Z, time_3d = zip(*sorted(zip(elems_X, elems_Y, elems_Z, time), key = lambda x: x[0]))
    #print(elems_X, elems_Y, time_3d)
    #print(flops, time_1d)
    plt.plot(flops, time_1d, 'o', flops, list(map(lambda x: lr_1d_fun(x),flops)) , 'x' ,list(map(lambda x,y,z: x*y*z ,elems_X, elems_Y, elems_Z)) , list(map(lambda x,y,z: lr_3d_fun(x,y,z),elems_X, elems_Y, elems_Z)) , '+' , list(map(lambda x,y,z: x*y*z ,elems_X, elems_Y, elems_Z)), list(map(lambda x,y,z: inter_3d_fun(x,y,z),elems_X, elems_Y, elems_Z)) , 'v')
    plt.legend(['Dgemm ' + locname + ' (Samples)', 'Dgemm ' + locname + ' (LR 1D)' , 'Dgemm ' + locname + ' (LR 3D)' , 'Dgemm ' + locname + ' (Interoplate 3d)'  ], loc='best')
    plt.ylabel('Time(s)')
    plt.xlabel('N*M*K')
    plt.title('Dgemm ' + locname +' (0-' + str(plot_up_bound) + '/'+ str(plot_up_bound) + ' )')
    plt.savefig(resDir + 'Dgemm ' + locname +'_'+machine+'_' + str(plot_up_bound) + '.eps', format='eps')
    plt.close()

    plt.plot( flops, predict_error(time_1d, list(map(lambda x: lr_1d_fun(x),flops))) , 'x' ,list(map(lambda x,y,z: x*y*z ,elems_X, elems_Y, elems_Z)) ,  predict_error(time_3d, list(map(lambda x,y,z: lr_3d_fun(x,y,z),elems_X, elems_Y, elems_Z))) , '+' , list(map(lambda x,y,z: x*y*z ,elems_X, elems_Y, elems_Z)),  predict_error(time_3d, list(map(lambda x,y,z: inter_3d_fun(x,y,z),elems_X, elems_Y, elems_Z))) , 'v')
    plt.legend(['Dgemm ' + locname + ' (LR 1D)' , 'Dgemm ' + locname + ' (LR 3D)' , 'Dgemm ' + locname + ' (Interoplate 3d)'  ], loc='best')
    plt.ylabel('Error')
    plt.xlabel('N*M*K')
    plt.title('Error_Dgemm ' + locname +' (0-' + str(plot_up_bound) + '/'+ str(plot_up_bound) + ' )')
    plt.savefig(resDir + 'Error_Dgemm ' + locname + '_'+machine+'_' + str(plot_up_bound) + '.eps', format='eps')
    plt.close()

plot_bw(1e4)
plot_bw(1e5)
plot_bw(1e6)
#plot_transpose(1e5)
#plot_transpose(1e6)
#plot_transpose(1e7)
#plot_transpose(1e8)
#plot_gemm(0, 1e7)
#plot_gemm(0, 1e8)
#plot_gemm(0, 1e9)




