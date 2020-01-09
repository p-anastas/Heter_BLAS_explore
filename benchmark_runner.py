import subprocess

functions=['dgemm']
sizes=[1000,5000,10000]
layouts=[1]
alphas=[1.345]
betas=[1.234, 0]
addmodes=[-1]
for func in functions:
	print('Running benchmarks for ' + func)
	for M in sizes:
		for N in sizes:
			for K in sizes:
				for alpha in alphas:
					for beta in betas:
						for M_layout in layouts:
							for N_layout in layouts:
								for K_layout in layouts:
									for addmode in addmodes:
										Runner = './build/' + func+ '_runner ' + str(M) + ' ' + str(N) + ' ' + str(K) + ' ' + str(M_layout) + ' ' + str(N_layout) + ' ' + str(K_layout) + ' ' + str(alpha) + ' ' + str(beta) + ' ' + str(addmode) 
										CPU_only = Runner + ' 0 0 0 BENCHMARK >>Results/CPU_only_log.md 2>Results/CPU_only_error.md'
										GPU_only = Runner + ' ' + str(M) + ' 0 0 BENCHMARK >>Results/GPU_only_log.md 2>Results/GPU_only_error.md'
										if (M*N + N*K + M*K)*8 < 8e9:
											process = subprocess.run(args=CPU_only, shell=True, check=True)
											process = subprocess.run(args=GPU_only, shell=True, check=True)
