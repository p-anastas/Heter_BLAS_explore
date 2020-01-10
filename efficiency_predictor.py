import subprocess

with open("Results/daxpy_log.md","r") as file0:
	add_db = file0.readlines()

with open("Results/CPU_only_log.md","r") as file0:
	cpu_gemm_db = file0.readlines()

with open("Results/GPU_only_log.md","r") as file0:
	gpu_gemm_db = file0.readlines()

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
	return 0.01*X/1e9*(1-transpose_overlap)

def t_dgemm_cpu(N,M,K):	
	for line in cpu_gemm_db:
		temp = line.split(',')
		if int(temp[0]) >= M and int(temp[1]) >= N and int(temp[2]) >= K:
			return float(temp[19])*(M/int(temp[0]))*(N/int(temp[1]))*(K/int(temp[2]))
		prev = line
	print("No DB entry found")
	return 0.0064*N*M*K/1e9

def t_dgemm_gpu(N,M,K):
	for line in gpu_gemm_db:
		temp = line.split(',')
		if int(temp[0]) >= M and int(temp[1]) >= N and int(temp[2]) >= K:
			return float(temp[18])*(M/int(temp[0]))*(N/int(temp[1]))*(K/int(temp[2]))
		prev = line
	print("No DB entry found")
	return 0.0092*N*M*K/1e9

def t_dadd(X):
	for line in add_db:
		temp = line.split(',')
		if int(temp[0]) >= X:
			return float(temp[4])*(X/int(temp[0]))*(1-add_overlap)

def t_dbuffer_init_gpu(N,M,K):
	return t_init_gpu(8*(M*N+N*K+K*M))

def t_dbuffer_init_cpu(N,M,K,buffer_flag):
	return t_init_pinned(8*buffer_flag*M*N)

def t_dcommunication(N,M,K,Ctransfer_flag):
	return t_transfer_to_gpu(8*(Ctransfer_flag*M*N+N*K+K*M)) + t_transfer_from_gpu(8*M*N)

def dgemm_flops(N,M,K):
	return M * K * (2 * N + 1)

def dgemm_bytes(N,M,K):
	return (M * K + K * N + M * N * 2) * 8

def print_bench(time, flops, bytes):
	Gflops = flops / (time * 1e9)
	Gbytes = bytes / (time * 1e9)
	return str(round(1000*time,5)) + ' ms ( ' + str(round(Gflops,3)) + ' Gflops/s ' + str(round(Gbytes,3)) + ' Gbytes/s)'


# Inputs = N, M, K, M_layout, N_layout, K_layout, (alpha), beta
# For all cases, t_total = t_gemm_computation + t_buffer_init + t_communication + t_transform + t_axpy
# Goal: minimize t_total
# For CPU-GPU environment choises are:

N = 10000
M = K = 10000
# CPU only -> 
t_total_cpu = t_dgemm_cpu(N,M,K)
print("CPU only M=%d N=%d K=%d -> " %(M,N,K) + print_bench(t_total_cpu, dgemm_flops(N,M,K), dgemm_bytes(N,M,K))) 

#GPU only -> 
for C_mem in [0,1]:
	t_total_gpu = t_dgemm_gpu(N,M,K) + t_dbuffer_init_gpu(N,M,K) + t_dbuffer_init_cpu(N,M,K,1-C_mem) + t_dcommunication(N,M,K,1) + t_dtranspose(2*M*N*(1-C_mem))
	t_total_gpu_addChost = t_dgemm_gpu(N,M,K) + t_dbuffer_init_gpu(N,M,K) + t_dbuffer_init_cpu(N,M,K,1-C_mem) + t_dcommunication(N,M,K,0) + t_dtranspose(M*N*(1-C_mem)  + t_dadd(M*N)) # 1 extra t_dadd hidden in CPU while gpu_gemm
	print("GPU Naive M=%d N=%d K=%d C_mem=%d -> " %(M,N,K,C_mem) +  print_bench(t_total_gpu, dgemm_flops(N,M,K), dgemm_bytes(N,M,K)))
	print("GPU addChost M=%d N=%d K=%d C_mem=%d -> " %(M,N,K, C_mem) + print_bench(t_total_gpu_addChost, dgemm_flops(N,M,K), dgemm_bytes(N,M,K)))

	#Assuming X*t_dcommunication(Y,flag) = t_dcommunication(X*Y,flag)
	for M_split in range(int(M/5),M+1,int(M/5)):
		M_gpu = M_split
		M_cpu = M - M_split
		t_gpu_overhead = t_dbuffer_init_gpu(N,M_gpu,K) + t_dcommunication(N,M_gpu,K,1)
		t_cpu_overhead = t_dbuffer_init_cpu(N,M_gpu,K,1-C_mem) + t_dtranspose(2*M_gpu*N*(1-C_mem))
		t_total_M_split = t_gpu_overhead + t_cpu_overhead +max(t_dgemm_gpu(N,M_gpu,K), t_dgemm_cpu(N,M_cpu,K))
		print("Hybrid M=%d N=%d K=%d C_mem=%d ( M_split=%d )-> " %(M,N,K,C_mem, M_split) +  print_bench(t_total_M_split, dgemm_flops(N,M,K), dgemm_bytes(N,M,K)))
		t_gpu_overhead_addChost = t_dbuffer_init_gpu(N,M_gpu,K) + t_dcommunication(N,M_gpu,K,0)
		t_cpu_overhead_addChost = t_dbuffer_init_cpu(N,M_gpu,K,1) + t_dtranspose(M_gpu*N*(1-C_mem)) + C_mem*N*t_dadd(M_gpu) + (1-C_mem)*t_dadd(N*M_gpu)
		t_total_M_split_addChost = t_gpu_overhead_addChost + t_cpu_overhead_addChost +max(t_dgemm_gpu(N,M_gpu,K), t_dgemm_cpu(N,M_cpu,K) + + C_mem*N*t_dadd(M_gpu) + (1-C_mem)*t_dadd(N*M_gpu))
		print("Hybrid addChost M=%d N=%d K=%d C_mem=%d ( M_split=%d )-> " %(M,N,K,C_mem, M_split) +  print_bench(t_total_M_split_addChost, dgemm_flops(N,M,K), dgemm_bytes(N,M,K)))


	#Assuming X*t_dcommunication(Y,flag) = t_dcommunication(X*Y,flag)
	for N_split in range(int(N/5),N+1,int(N/5)):
		N_gpu = N_split
		N_cpu = N - N_split
		t_gpu_overhead = t_dbuffer_init_gpu(N_gpu,M,K) + t_dcommunication(N_gpu,M,K,1)
		t_cpu_overhead = t_dbuffer_init_cpu(N_gpu,M,K,1-C_mem) + t_dtranspose(2*M*N_gpu*(1-C_mem))
		t_total_N_split = t_gpu_overhead + t_cpu_overhead +max(t_dgemm_gpu(N_gpu,M,K), t_dgemm_cpu(N_cpu,M,K))
		print("Hybrid M=%d N=%d K=%d C_mem=%d ( N_split=%d )-> " %(M,N,K,C_mem, N_split) +  print_bench(t_total_N_split, dgemm_flops(N,M,K), dgemm_bytes(N,M,K)))
		t_gpu_overhead_addChost = t_dbuffer_init_gpu(N_gpu,M,K) + t_dcommunication(N_gpu,M,K,0)
		t_cpu_overhead_addChost = t_dbuffer_init_cpu(N_gpu,M,K,1) + t_dtranspose(M*N_gpu*(1-C_mem)) + (1-C_mem)*M*t_dadd(N_gpu) + C_mem*t_dadd(N_gpu*M)
		t_total_N_split_addChost = t_gpu_overhead_addChost + t_cpu_overhead_addChost +max(t_dgemm_gpu(N_gpu,M,K), t_dgemm_cpu(N_cpu,M,K) + (1-C_mem)*M*t_dadd(N_gpu) + C_mem*t_dadd(N_gpu*M))
		print("Hybrid addChost M=%d N=%d K=%d C_mem=%d ( N_split=%d )-> " %(M,N,K,C_mem, N_split) +  print_bench(t_total_N_split_addChost, dgemm_flops(N,M,K), dgemm_bytes(N,M,K)))


	#Assuming X*t_dcommunication(Y,flag) = t_dcommunication(X*Y,flag)
	for K_split in range(int(K/10),K+1,int(K/10)):
		K_gpu = K_split
		K_cpu = K - K_split
		t_gpu_overhead = t_dbuffer_init_gpu(N,M,K_gpu) + t_dcommunication(N,M,K_gpu,0)
		t_cpu_overhead = t_dbuffer_init_cpu(N,M,K_gpu,1) + t_dtranspose(M*N*(1-C_mem)) + t_dadd(M*N)
		t_total_K_split = t_gpu_overhead + t_cpu_overhead +max(t_dgemm_gpu(N,M,K_gpu), t_dgemm_cpu(N,M,K_cpu))
		print("Hybrid (always addChost) M=%d N=%d K=%d C_mem=%d ( K_split=%d )-> " %(M,N,K,C_mem, K_split) +  print_bench(t_total_K_split, dgemm_flops(N,M,K), dgemm_bytes(N,M,K)))







