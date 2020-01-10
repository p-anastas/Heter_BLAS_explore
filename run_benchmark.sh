echo "Running benchmarks for daxpy"
alpha=1.2345
echo "" > Results/daxpy_log.md
for N in 100 200 500 1000 2000 5000 10000 20000 50000 100000 200000 500000 1000000 2000000 5000000 10000000 20000000 50000000 100000000 200000000 500000000
do
	./build/daxpy_benchmark ${N} ${alpha} 1 1 >>Results/daxpy_log.md 2>>Results/daxpy_error.md
done
