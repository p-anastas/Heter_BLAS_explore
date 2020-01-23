import subprocess
import sys
import numpy as np
import scipy.optimize as scipop
import scipy as sc
import matplotlib.pyplot as plt
from bandwidth_functions import *
from general_functions import *
from transpose_functions import *
from dgemm_functions import t_dgemm_cpu, t_dgemm_gpu

machine = 'dungani'
resDir = 'Results_' + machine + '/'

# For now use artificial bound to check
bounds = [1e4,1e5,1e6,8e9]
flop_bounds = [1e6,1e7,1e8,1e9,8e9]

resDir, bw_file, _, _, _, _ = initialize()

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
    xnew = np.linspace(plot_down_bound, plot_up_bound, num=50, endpoint=True)
    ynew = np.linspace(plot_down_bound, plot_up_bound, num=50, endpoint=True)
    time = [0]
    flops= [0]
    for line in trans_db:
        temp = line.split(',')
        if int(temp[0])*int(temp[1])< plot_up_bound:
           if not (int(temp[0])*int(temp[1]) in flops):
                time.append(float(temp[2]))
                flops.append(int(temp[0])*int(temp[1]))
    flops, time = zip(*sorted(zip(flops, time)))
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    X, Y = np.meshgrid(xnew, ynew)
    ax.plot_wireframe(X,Y,list(map(lambda x,y: linearized1d_dtranspose(x*y), (X,Y))), color = 'c')
    # plt.xscale('log')
    plt.legend('CPU transpose', loc='best')
    ax.set_xlabel('M')
    ax.set_ylabel('N')
    ax.set_zlabel('Time(s)')
    ax.view_init(elev=20., azim=-35)
    plt.title(resDir + 'Transpose_'+ machine +  '(0-' + str(plot_up_bound) + '/'+ str(plot_up_bound) + ' )')
    plt.savefig(resDir +'Transpose_'+machine+'_3d' + str(plot_up_bound) + '.eps', format='eps')
    plt.close()
    #plt.plot(flops, GigaVal_per_s_l(flops,time), 'o', flops, GigaVal_per_s_l(flops,list(map(lambda x: t_dtranspose_lin(x),flops))) , '-' )
    #plt.legend(['Transpose Host (Samples)', 'Transpose Host (LR total flops)' ], loc='best')
    #plt.ylabel('Gflops/s')
    #plt.xlabel('Size')
    #plt.title('Transpose 1d-fied (0-' + str(plot_up_bound) + '/'+ str(plot_up_bound) + ' )')
    #plt.savefig('Transpose_'+machine+'_1d' + str(plot_up_bound) + '.eps', format='eps')
    #plt.close()

#plot_bw(1e4)
#plot_bw(1e5)
#plot_bw(1e6)
#plot_transpose(1e5)
plot_transpose(1e6)
plot_transpose(1e7)




