import subprocess
import os


def comp_accuracy(X, Y, error):
    if ((X >= Y * (1 - error) and X <= Y * (1 + error))
            or (Y >= X * (1 - error) and Y <= X * (1 + error))):
        return True
    return False


machine = 'gold1'
resDir = 'Results_' + machine + '/'
if not os.path.exists(resDir):
    os.mkdir(resDir)

sizes = list(range(100, 1000, 100))
sizes.extend(list(range(1000, 50000, 1000)))
sizes.extend(list(range(50000, 100000, 10000)))
sizes.extend(list(range(100000, 1000000, 100000)))
sizes.extend(list(range(1000000, 10000000, 1000000)))
sizes.extend(list(range(10000000, 100000000, 10000000)))
sizes.extend(list(range(100000000, 1000000000, 100000000)))
benchmark_skip_error = 0.1
benchmark_recursive_depth = 3
alpha = 1.0
print('\nRunning benchmarks for dgemm_runner')
logf = resDir + 'daxpy_log_' + machine + '.md'
errf = resDir + 'daxpy_err_' + machine + '.md'
for N in sizes:
    skip_benchmark = False
    with open(logf, "r") as file0:
        log = file0.readlines()
    rec = 0
    N_r = N
    for line in log:
        temp = line.split(',')
        if int(temp[0]) == N:
            print('Skipping N=' + str(N) + ' ...already benchmarked')
            skip_benchmark = True
    #N_rec = []
    #while (rec < benchmark_recursive_depth and skip_benchmark == False):
    #    N_r /= 2
    #    for line in log:
    #        temp = line.split(',')
    #        if int(temp[0]) == N_r:
    #            N_rec.append(float(temp[4]))
    #    rec += 1
    #if len(N_rec) == benchmark_recursive_depth and skip_benchmark == False:
        #print('Found ' +  str(benchmark_recursive_depth) + ' existing ancestors')
    #    ctr = 1
    #    while(ctr < benchmark_recursive_depth and comp_accuracy(2 * N_rec[ctr], N_rec[ctr - 1], benchmark_skip_error)):
    #        ctr += 1
    #    if ctr == benchmark_recursive_depth:
    #        skip_benchmark = True
    #        with open(logf, "a+") as file0:
    #            file0.write(str(N) + ',' + str(alpha) + ',1,1,' + str(N_rec[0] * 2) + ',synth\n')
    #        print('Skipping N=' + str(N) + ' ...time calculated as 2*  N=' + str(N / 2))
    if skip_benchmark == False:
        Runner = './build/daxpy_benchmark ' + str(N) + ' ' + str(alpha) + ' 1 1'
        proc = Runner + ' >>' + logf + ' 2>>' + errf
        print('Running: ' + proc)
        if 8 * N < 8e9:
            process = subprocess.call(proc, shell=True)
with open(logf, "r") as file0:
    log = file0.readlines()
log.sort(key=lambda x: (int(x.split(',')[0])))
with open(logf + '_sorted', "w+") as file0:
    file0.writelines(log)
