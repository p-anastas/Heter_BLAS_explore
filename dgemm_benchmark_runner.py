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

sizes = [100,200,500,1000, 2000, 5000, 10000]
benchmark_skip_error = 0.0
benchmark_recursive_depth = 0
print('\nRunning benchmarks for dgemm_runner')
logfile = [resDir + 'CPU_only_log_' + machine + '.md', resDir + 'GPU_only_log_' + machine + '.md']
errfile = [resDir + 'CPU_only_err_' + machine + '.md', resDir + 'GPU_only_err_' + machine + '.md']

for M in sizes:
    for N in sizes:
        for K in sizes:
            for logf, errf in zip(logfile, errfile):
                skip_benchmark = False
                if not os.path.exists(logf):
                    os.mknod(logf)
                with open(logf, "r") as file0:
                    log = file0.readlines()
                rec = 0
                M_rec = M
                N_rec = N
                K_rec = K
                for line in log:
                    temp = line.split(',')
                    if int(temp[0]) == M and int(temp[1]) == N and int(temp[2]) == K:
                        print('Skipping M=' + str(M) + ' ,N=' + str(N) +
                              ' ,K=' + str(K) + ' ...already benchmarked')
                        skip_benchmark = True
                M_rec_t = []
                N_rec_t = []
                K_rec_t = []
                while (rec < benchmark_recursive_depth and skip_benchmark == False):
                    M_rec /= 2
                    N_rec /= 2
                    K_rec /= 2
                    for line in log:
                        temp = line.split(',')
                        if int(temp[0]) == M_rec and int(temp[1]) == N and int(temp[2]) == K:
                            M_rec_t.append(float(temp[19]))
                        if int(temp[0]) == M and int(temp[1]) == N_rec and int(temp[2]) == K:
                            N_rec_t.append(float(temp[19]))
                        if int(temp[0]) == M and int(temp[1]) == N and int(temp[2]) == K_rec:
                            K_rec_t.append(float(temp[19]))
                    rec += 1
                if len(M_rec_t) == benchmark_recursive_depth and skip_benchmark == False:
                    #print('Found ' +  str(benchmark_recursive_depth) + ' existing Μ ancestors')
                    ctr = 1
                    while(ctr < benchmark_recursive_depth and comp_accuracy(2 * M_rec_t[ctr], M_rec_t[ctr - 1], benchmark_skip_error)):
                        ctr += 1
                    if ctr == benchmark_recursive_depth:
                        skip_benchmark = True
                        with open(logf, "a+") as file0:
                            file0.write(
                                str(M) +
                                ',' +
                                str(N) +
                                ',' +
                                str(K) +
                                ',Row major,Row major,Col major,1.230000,1.234000,-1,0,0,0,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,' +
                                str(
                                    M_rec_t[0] *
                                    2) +
                                ',synth\n')
                        print('Skipping M=' +
                              str(M) +
                              ' ,N=' +
                              str(N) +
                              ' ,K=' +
                              str(K) +
                              ' ...time calculated as 2* ' +
                              'M=' +
                              str(M /
                                  2) +
                              ' ,N=' +
                              str(N) +
                              ' ,K=' +
                              str(K))
                if len(N_rec_t) == benchmark_recursive_depth and skip_benchmark == False:
                    #print('Found ' +  str(benchmark_recursive_depth) + ' existing Ν ancestors')
                    ctr = 1
                    while(ctr < benchmark_recursive_depth and comp_accuracy(2 * N_rec_t[ctr], N_rec_t[ctr - 1], benchmark_skip_error)):
                        ctr += 1
                    if ctr == benchmark_recursive_depth:
                        skip_benchmark = True
                        with open(logf, "a+") as file0:
                            file0.write(
                                str(M) +
                                ',' +
                                str(N) +
                                ',' +
                                str(K) +
                                ',Row major,Row major,Col major,1.230000,1.234000,-1,0,0,0,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,' +
                                str(
                                    N_rec_t[0] *
                                    2) +
                                ',synth\n')
                        print('Skipping M=' +
                              str(M) +
                              ' ,N=' +
                              str(N) +
                              ' ,K=' +
                              str(K) +
                              ' ...time calculated as 2* ' +
                              'M=' +
                              str(M) +
                              ' ,N=' +
                              str(N /
                                  2) +
                              ' ,K=' +
                              str(K))
                if len(K_rec_t) == benchmark_recursive_depth and skip_benchmark == False:
                    #print('Found ' +  str(benchmark_recursive_depth) + ' existing Κ ancestors')
                    ctr = 1
                    while(ctr < benchmark_recursive_depth and comp_accuracy(2 * K_rec_t[ctr], K_rec_t[ctr - 1], benchmark_skip_error)):
                        ctr += 1
                    if ctr == benchmark_recursive_depth:
                        skip_benchmark = True
                        with open(logf, "a+") as file0:
                            file0.write(
                                str(M) +
                                ',' +
                                str(N) +
                                ',' +
                                str(K) +
                                ',Row major,Row major,Col major,1.230000,1.234000,-1,0,0,0,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,' +
                                str(
                                    K_rec_t[0] *
                                    2) +
                                ',synth\n')
                        print('Skipping M=' +
                              str(M) +
                              ' ,N=' +
                              str(N) +
                              ' ,K=' +
                              str(K) +
                              ' ...time calculated as 2* ' +
                              'M=' +
                              str(M) +
                              ' ,N=' +
                              str(N) +
                              ' ,K=' +
                              str(K /
                                  2))
                if skip_benchmark == False:
                    Runner = './build/dgemm_runner ' + \
                        str(M) + ' ' + str(N) + ' ' + str(K) + ' 0 0 1 1.345 1.234 -1'
                    if (logf != resDir + 'GPU_only_log_' + machine + '.md'):
                        proc = Runner + ' 0 0 0 BENCHMARK >>' + logf + ' 2>' + errf
                    else:
                        proc = Runner + ' ' + str(M) + ' 0 0 BENCHMARK >>' + logf + ' 2>' + errf
                    if (M * N + N * K + M * K) * 8 < 8e9:
                        process = subprocess.call(proc, shell=True)
                #process = subprocess.run(args=GPU_only, shell=True, check=True)
for logf in logfile:
    with open(logf, "r") as file0:
        log = file0.readlines()
    log.sort(key=lambda x: (int(x.split(',')[0]), int(x.split(',')[1]), int(x.split(',')[2])))
    with open(logf + '_sorted', "w+") as file0:
        file0.writelines(log)
