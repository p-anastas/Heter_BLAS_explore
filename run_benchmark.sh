echo "Running benchmarks for daxpy"
echo "Running benchmarks for bandwidth"
alpha=1.2345
rm Results/daxpy_log_gold1.md
rm Results/daxpy_error_gold1.md
rm Results/bandwidth_log_gold1.md
rm Results/bandwidth_error_gold1.md
for N in 100 200 500 1000 2000 5000 10000 20000 50000 100000 200000 500000 1000000 2000000 5000000 10000000 20000000 50000000 100000000 200000000 500000000
do 
	./build/daxpy_benchmark ${N} ${alpha} 1 1 >>Results/daxpy_log_gold1.md 2>>Results/daxpy_error_gold1.md
	N_bytes=$(( 8*N))
	./build/bandwidth_benchmark ${N_bytes} -1 -1 >>Results/bandwidth_log_gold1.md 2>>Results/bandwidth_error_gold1.md
	./build/bandwidth_benchmark ${N_bytes} -1 0 >>Results/bandwidth_log_gold1.md 2>>Results/bandwidth_error_gold1.md
	./build/bandwidth_benchmark ${N_bytes} 0 -1 >>Results/bandwidth_log_gold1.md 2>>Results/bandwidth_error_gold1.md	
done

echo "Running benchmarks for transpose"
rm Results/transpose_log_gold1.md
rm Results/transpose_error_gold1.md
for N in 500 1000 2000 5000 10000 
do
	for M in 500 1000 2000 5000 10000 
	do
		./build/transpose_benchmark ${N} ${M} >>Results/transpose_log_gold1.md 2>>Results/transpose_error_gold1.md
	done
done
