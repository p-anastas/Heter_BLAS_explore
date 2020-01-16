import subprocess
import sys
machine = 'gold1'
resDir = 'Results_' + machine + '/'

with open(resDir + 'bandwidth_log_' + machine + '.md_sorted', "r") as file0:
    bw_db = file0.readlines()

with open(resDir + 'daxpy_log_' + machine + '.md_sorted', "r") as file0:
    add_db = file0.readlines()

def binary_bounds(start, X):
    next = start
    while (next < X):
        next *= 2
    if next == X:
        prev = next
    else:
        prev = next / 2
    return (prev, next)

def report_bandwidth(bytes):
	cpu_to_gpu_time =  t_memcpy(bytes, -1, 0)
	gpu_to_cpu_time =  t_memcpy(bytes, 0, -1)
	print('CPU to GPU %d bytes t = %.5lf ms, bw =  %.3lf Gb/s'  % (bytes, 1000*cpu_to_gpu_time, GigaVal_per_s(bytes,cpu_to_gpu_time) ))
	print('GPU to CPU %d bytes t = %.5lf ms, bw =  %.3lf Gb/s\n'  % (bytes, 1000*gpu_to_cpu_time, GigaVal_per_s(bytes,gpu_to_cpu_time) ))

def report_flops(N):
	t_add = t_add_vec_1d(N, 8)
	print('Add (%d)  t = %.5lf ms, flops =  %.3lf Gfops/s\n'  % (N, 1000*t_add, GigaVal_per_s(N,t_add) ))

def GigaVal_per_s(val, time):
	return val*1e-9/time

def t_transfer_to_gpu(bytes):
    bytes_min, bytes_max = binary_bounds(625, bytes)
    time_min = 0
    time_max = 0
    for line in bw_db:
        temp = line.split(',')
        if int(temp[0]) == bytes_min and int(temp[1]) == -1 and int(temp[2]) == 0:
            time_min = float(temp[3])
        if int(temp[0]) == bytes_max and int(temp[1]) == -1 and int(temp[2]) == 0:
            time_max = float(temp[3])
    if (time_min == 0 or time_max == 0):
        print("t_transfer_to_gpu: No DB entry found")
        return bytes / 6e9
    else:
        if bytes_max == bytes_min:
            return  time_min
        else:
            return  (time_min + (time_max - time_min) /
                                                  (bytes_max - bytes_min) * (bytes - bytes_min))


def t_transfer_from_gpu(bytes):
    bytes_min, bytes_max = binary_bounds(625, bytes)
    time_min = 0
    time_max = 0
    for line in bw_db:
        temp = line.split(',')
        if int(temp[0]) == bytes_min and int(temp[1]) == 0 and int(temp[2]) == -1:
            time_min = float(temp[3])
        if int(temp[0]) == bytes_max and int(temp[1]) == 0 and int(temp[2]) == -1:
            time_max = float(temp[3])
    if (time_min == 0 or time_max == 0):
        print("t_transfer_from_gpu: No DB entry found")
        return bytes / 6e9
    else:
        if bytes_max == bytes_min:
            return time_min
        else:
            return (time_min + (time_max - time_min) /
                                                  (bytes_max - bytes_min) * (bytes - bytes_min))

def t_dadd(X):
    N_min, N_max = binary_bounds(625, X)
    time_min = 0
    time_max = 0
    for line in add_db:
        temp = line.split(',')
        if int(temp[0]) == N_min:
            time_min = float(temp[4])
        if int(temp[0]) == N_max:
            time_max = float(temp[4])
    if (time_min == 0 or time_max == 0):
        print("t_dadd: No DB entry found")
        return 0.0064*X
    else:
        if N_max == N_min:
            return time_min
        else:
            return (time_min + (time_max - time_min) / (N_max - N_min) * (N - N_min))

def t_memcpy(bytes, src, dest):
	## For now only for dev0 <-> host 
	if (src == -1 and dest == 0):
		return t_transfer_to_gpu(bytes)
	if (src ==  0 and dest == -1):
		return t_transfer_from_gpu(bytes)

def t_copy_vec_1d(N,src,dest,elem_size):
	return t_memcpy(N*elem_size, src, dest)

def t_copy_vec_2d(dim1, dim2, ldim,src,dest, elem_size):
	if ldim == dim2:
		return t_memcpy(dim1*dim2*elem_size, src, dest)
	elif ldim > dim2:
		return dim1*t_memcpy(dim2*elem_size, src, dest)
	else:
		sys.exit('Error: dim2(%d) > ldim(%d)' % (dim2, ldim))

def t_add_vec_1d(N,elem_size):
	## For now assume 8 bytes -> double
	if elem_size == 8:
		return t_dadd(N)
	else:
		sys.exit('Error: t_add_vec_1d(%d,%d) -> unknown elem_size' % (N, elem_size))

def t_add_vec_2d(dim1, dim2, ldim, elem_size):
	if ldim == dim2:
		return t_add_vec_1d(dim1*dim2,elem_size)
	elif ldim > dim2:
		return dim1*t_add_vec_1d(dim2,elem_size)
	else:
		sys.exit('Error: dim2(%d) > ldim(%d)' % (dim2, ldim))


sizes = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200,102400,204800,409600]

dtype_size = 8 

for N in sizes:
	report_bandwidth(N*dtype_size)
	report_flops(N)

