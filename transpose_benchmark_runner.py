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

sizes = list(range(100, 1000, 200))
sizes.extend([1000, 2000, 5000, 10000,20000])
benchmark_skip_error = 0.1
benchmark_recursive_depth = 3
print('\nRunning benchmarks for transpose_benchmark')
logf = resDir + 'transpose_log_' + machine + '.md'
errf = resDir + 'transpose_err_' + machine + '.md'
if not os.path.exists(logf):
    os.mknod(logf)
if not os.path.exists(errf):
    os.mknod(errf)

for M in sizes:
    for N in sizes:
        skip_benchmark = False
        with open(logf, "r") as file0:
            log = file0.readlines()
        rec = 0
        #M_rec = M
        #N_rec = N
        for line in log:
            temp = line.split(',')
            if int(temp[0]) == M and int(temp[1]) == N:
                print('Skipping M=' + str(M) + ' ,N=' + str(N) + ' ...already benchmarked')
                skip_benchmark = True
        #M_rec_t = []
        #N_rec_t = []
        #while (rec < benchmark_recursive_depth and skip_benchmark == False):
        #    M_rec /= 2
        #    N_rec /= 2
        #    for line in log:
        #        temp = line.split(',')
        #        if int(temp[0]) == M_rec and int(temp[1]) == N:
        #            M_rec_t.append(float(temp[2]))
        #        if int(temp[0]) == M and int(temp[1]) == N_rec:
        #            N_rec_t.append(float(temp[2]))
        #    rec += 1
        #if len(M_rec_t) == benchmark_recursive_depth and skip_benchmark == False:
            #print('Found ' +  str(benchmark_recursive_depth) + ' existing M ancestors')
        #    ctr = 1
        #    while(ctr < benchmark_recursive_depth and comp_accuracy(2 * M_rec_t[ctr], M_rec_t[ctr - 1], benchmark_skip_error)):
        #        ctr += 1
        #    if ctr == benchmark_recursive_depth:
        #        skip_benchmark = True
        #        with open(logf, "a+") as file0:
        #            file0.write(str(M) + ',' + str(N) + ',' + str(M_rec_t[0] * 2) + ',synth\n')
        #        print('Skipping M=' + str(M) + ' ,N=' + str(N) +
        #              ' ...time calculated as 2* ' + 'M=' + str(M / 2) + ' ,N=' + str(N))
        #if len(N_rec_t) == benchmark_recursive_depth and skip_benchmark == False:
            #print('Found ' +  str(benchmark_recursive_depth) + ' existing N ancestors')
        #    ctr = 1
        #    while(ctr < benchmark_recursive_depth and comp_accuracy(2 * N_rec_t[ctr], N_rec_t[ctr - 1], benchmark_skip_error)):
        #        ctr += 1
        #    if ctr == benchmark_recursive_depth:
        #        skip_benchmark = True
        #        with open(logf, "a+") as file0:
        #            file0.write(str(M) + ',' + str(N) + ',' + str(N_rec_t[0] * 2) + ',synth\n')
        #        print('Skipping M=' + str(M) + ' ,N=' + str(N) +
        #              ' ...time calculated as 2* ' + 'M=' + str(M) + ' ,N=' + str(N / 2))
        if skip_benchmark == False:
            Runner = './build/transpose_benchmark ' + str(M) + ' ' + str(N)
            proc = Runner + ' >>' + logf + ' 2>' + errf
            if (M * N) * 8 < 8e9:
                process = subprocess.call(proc, shell=True)
with open(logf, "r") as file0:
    log = file0.readlines()
log.sort(key=lambda x: (int(x.split(',')[0]), int(x.split(',')[1])))
with open(logf + '_sorted', "w+") as file0:
    file0.writelines(log)
