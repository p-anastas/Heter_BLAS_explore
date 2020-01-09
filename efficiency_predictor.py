import subprocess

#For now assume no overlaping
communication_overlap = 0
gpu_init_overlap = 0
cpu_init_overlap = 0
transpose_overlap = 0 
add_overlap = 0 
#for now use default values
def t_init_pinned(bytes):
	return bytes/2e10*(1-cpu_init_overlap)

def t_init_gpu(bytes):
	return bytes/2e10*(1-gpu_init_overlap)

def t_transfer_to_gpu(bytes):
	return bytes/8e9*(1-communication_overlap)

def t_transfer_from_gpu(bytes):
	return bytes/8e9*(1-communication_overlap)

def t_dtranspose(X):
	return 0.0064*X/1e9*(1-transpose_overlap)

def t_dgemm_cpu(N,M,K):
	return 0.0064*N*M*K/1e9

def t_dgemm_gpu(N,M,K):
	return 0.0092*N*M*K/1e9

def t_dadd(X):
	return 0.0064*X/1e9*(1-add_overlap)

def t_dbuffer_init(N,M,K,buffer_flag):
	return t_init_gpu(8*(M*N+N*K+K*M)) + t_init_pinned(8*buffer_flag*M*N)

def t_dcommunication(N,M,K,Ctransfer_flag):
	return t_transfer_to_gpu(8*(Ctransfer_flag*M*N+N*K+K*M)) + t_transfer_from_gpu(8*M*N)

def dgemm_flops(N,M,K):
	return M * K * (2 * N + 1)

def dgemm_bytes(N,M,K):
	return (M * K + K * N + M * N * 2) * 8

def print_bench(time, flops, bytes):
	Gflops = flops / (time * 1e9)
	Gbytes = bytes / (time * 1e9)
	print(str(round(1000*time,5)) + ' ms ( ' + str(round(Gflops,3)) + ' Gflops/s ' + str(round(Gbytes,3)) + ' Gbytes/s)')


# Inputs = N, M, K, M_layout, N_layout, K_layout, (alpha), beta
# For all cases, t_total = t_gemm_computation + t_buffer_init + t_communication + t_transform + t_axpy
# Goal: minimize t_total
# For CPU-GPU environment choises are:

N = M = K = 1000
# CPU only -> 
t_total_cpu = t_dgemm_cpu(N,M,K)
print(t_total_cpu)
#GPU only -> 
for C_mem in [0,1]:
	t_total_gpu = t_dgemm_gpu(N,M,K) + t_dbuffer_init(N,M,K,1-C_mem) + t_dcommunication(N,M,K,1) + t_dtranspose(2*M*N*(1-C_mem)) + t_dadd(0)
	t_total_gpu_addChost = t_dgemm_gpu(N,M,K) + t_dbuffer_init(N,M,K,1-C_mem) + t_dcommunication(N,M,K,0) + t_dtranspose(M*N*(1-C_mem)  + t_dadd(M*N)) # 1 extra t_dadd hidden in CPU while gpu_gemm
	print_bench(t_total_gpu, dgemm_flops(N,M,K), dgemm_bytes(N,M,K))
	print_bench(t_total_gpu_addChost, dgemm_flops(N,M,K), dgemm_bytes(N,M,K))
