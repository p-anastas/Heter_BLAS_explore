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
sizes.extend(list(range(1000, 10000, 1000)))
sizes.extend(list(range(10000, 100000, 10000)))
sizes.extend(list(range(100000, 1000000, 100000)))
# print(sizes)
benchmark_skip_error = 0.1
benchmark_recursive_depth = 5
print('\nRunning benchmarks for dgemm_runner')
logf = resDir + 'bandwidth_log_' + machine + '.md'
errf = resDir + 'bandwidth_err_' + machine + '.md'
devnum = [-1, 0]
for src in devnum:
    for dest in devnum:
        for Bytes in sizes:
            skip_benchmark = False
            with open(logf, "r") as file0:
                log = file0.readlines()
            rec = 0
            Bytes_r = [Bytes]
            for line in log:
                temp = line.split(',')
                if (int(temp[0]) == Bytes and int(temp[1]) == src and int(temp[2]) == dest):
                    print(
                        'Skipping Bytes=' +
                        str(Bytes) +
                        ' src=' +
                        str(src) +
                        ' dest=' +
                        str(dest) +
                        ' ...already benchmarked')
                    skip_benchmark = True
            Bytes_rec = []
            while (rec < benchmark_recursive_depth and skip_benchmark == False):
                Bytes_r.append(Bytes_r[-1] / 2)
                for line in log:
                    temp = line.split(',')
                    if (int(temp[0]) == Bytes_r[-1] and int(temp[1])
                            == src and int(temp[2]) == dest):
                        Bytes_rec.append(float(temp[3]))
                rec += 1
            if len(Bytes_rec) == benchmark_recursive_depth and skip_benchmark == False:
                #print('Found ' +  str(benchmark_recursive_depth) + ' existing ancestors')
                ctr = 1
                while(ctr < benchmark_recursive_depth and comp_accuracy(Bytes_r[ctr + 1] / Bytes_r[ctr] * Bytes_rec[ctr], Bytes_rec[ctr - 1], benchmark_skip_error)):
                    ctr += 1
                if ctr == benchmark_recursive_depth:
                    skip_benchmark = True
                    with open(logf, "a+") as file0:
                        file0.write(str(Bytes) + ',' + str(src) + ',' + str(dest) +
                                    ',' + str(Bytes_r[0] / Bytes_r[1] * 2) + ',synth\n')
                    print('Skipping Bytes=' +
                          str(Bytes) +
                          ' src=' +
                          str(src) +
                          ' dest=' +
                          str(dest) +
                          ' ...time calculated as 2* ' +
                          'Bytes=' +
                          str(Bytes /
                              2) +
                          ' src=' +
                          str(src) +
                          ' dest=' +
                          str(dest))
            if skip_benchmark == False:
                Runner = './build/bandwidth_benchmark ' + \
                    str(Bytes) + ' ' + str(src) + ' ' + str(dest)
                proc = Runner + ' >>' + logf + ' 2>>' + errf
                print('Running: ' + proc)
                if Bytes < 8e9:
                    process = subprocess.run(args=proc, shell=True, check=True)
with open(logf, "r") as file0:
    log = file0.readlines()
log.sort(key=lambda x: (int(x.split(',')[0])))
with open(logf + '_sorted', "w+") as file0:
    file0.writelines(log)
