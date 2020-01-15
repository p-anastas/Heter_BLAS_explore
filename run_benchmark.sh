echo "Running benchmarks for daxpy"
echo "Running benchmarks for bandwidth"
alpha=1.2345
machine=$1
rm Results/daxpy_log_${machine}.md
rm Results/daxpy_error_${machine}.md
rm Results/bandwidth_log_${machine}.md
rm Results/bandwidth_error_${machine}.md
for N in 100 200 500 1000 2000 5000 10000 20000 50000 100000 200000 500000 1000000 2000000 5000000 10000000 20000000 50000000 100000000 200000000 500000000
do 
	./build/daxpy_benchmark ${N} ${alpha} 1 1 >>Results/daxpy_log_${machine}.md 2>>Results/daxpy_error_${machine}.md
	N_bytes=$(( 8*N))
	./build/bandwidth_benchmark ${N_bytes} -1 -1 >>Results/bandwidth_log_${machine}.md 2>>Results/bandwidth_error_${machine}.md
	./build/bandwidth_benchmark ${N_bytes} -1 0 >>Results/bandwidth_log_${machine}.md 2>>Results/bandwidth_error_${machine}.md
	./build/bandwidth_benchmark ${N_bytes} 0 -1 >>Results/bandwidth_log_${machine}.md 2>>Results/bandwidth_error_${machine}.md	
done

echo "Running benchmarks for transpose"
rm Results/transpose_log_${machine}.md
rm Results/transpose_error_${machine}.md
for N in 500 1000 2000 5000 10000 
do
	for M in 500 1000 2000 5000 10000 
	do
		./build/transpose_benchmark ${N} ${M} >>Results/transpose_log_${machine}.md 2>>Results/transpose_error_${machine}.md
	done
done

mkdir -p Results_${machine}
rm Results_${machine}/CPU_only_log_${machine}.md
rm Results_${machine}/CPU_only_error_${machine}.md
rm Results_${machine}/GPU_only_log_${machine}.md
rm Results_${machine}/GPU_only_log_${machine}.md
echo "Running benchmarks for dgemm"
for N in 100 200 400 800 1600 3200 6400
do
	for M in 100 200 400 800 1600 3200 6400
	do
		for K in 100 200 400 800 1600 3200 6400
		do
			./build/dgemm_runner ${N} ${M} ${K} 0 0 1 1.23 1.234 -1 0 0 0 BENCHMARK >>Results/CPU_only_log_${machine}.md 2>>Results/CPU_only_error_${machine}.md
			./build/dgemm_runner ${N} ${M} ${K} 0 0 1 1.23 1.234 -1 ${N} 0 0 BENCHMARK >>Results/GPU_only_log_${machine}.md 2>>Results/GPU_only_error_${machine}.md
		done
	done
done
